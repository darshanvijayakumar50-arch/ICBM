import numpy as np

# ==============================================================================
# 1. CORE MATHEMATICAL HELPERS
# ==============================================================================

def euler_to_quaternion(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    qxx, qyy, qzz = qx * qx, qy * qy, qz * qz
    qxy, qxz, qyz = qx * qy, qx * qz, qy * qz
    qwx, qwy, qwz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qwz), 2.0 * (qxz + qwy)],
        [ 2.0 * (qxy + qwz), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qwx)],
        [ 2.0 * (qxz - qwy), 2.0 * (qyz + qwx), 1.0 - 2.0 * (qxx + qyy)]
    ], dtype=np.float64)


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


def skew_symmetric(v):
    return np.array([
        [ 0.0, -v[2], v[1]],
        [ v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

# DATA VALIDATION ANDD REPAIR FUNNCTIONS

def validate_imu_stream(imu_stream):

    if imu_stream.ndim != 2:
        raise ValueError("IMU stream must be 2D")

    if imu_stream.shape[1] != 7:
        raise ValueError(
            "Expected [time, ax, ay, az, gx, gy, gz]"
        )

    if len(imu_stream) < 2:
        raise ValueError(
            "Insufficient IMU samples"
        )

    if np.isnan(imu_stream[:,0]).any():
        raise ValueError(
            "Timestamp column contains NaN"
        )

    return True


def repair_nan_values(data):

    data = data.copy()

    for col in range(data.shape[1]):

        signal = data[:, col]

        nan_mask = np.isnan(signal)

        if np.any(nan_mask):
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) == 0:
                raise ValueError(f"Column {col} contains only Nans")


            signal[nan_mask] = np.interp(
                np.where(nan_mask)[0],
                valid_indices,
                signal[valid_indices]
            )

            data[:, col] = signal

    return data


# ==============================================================================
# 2. ESKF PIPELINE MODULES
# ==============================================================================

def initialize_alignment(imu_data, num_samples=50):
    """Calculates T=0 orientation quaternion from static gravity vector."""
    accel_samples = imu_data[:num_samples, 1:4] # Assuming cols: [time, ax, ay, az, ...]
    mean_accel = np.median(accel_samples, axis=0)
    ax, ay, az = mean_accel

    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    yaw = 0.0 # Unobservable without magnetometer

    q0 = euler_to_quaternion(roll, pitch, yaw)
    return q0 / np.linalg.norm(q0)


def eskf_predict(state, acc_raw, gyro_raw, P, dt, g_world=np.array([0.0, 0.0, 9.81])):
    """Propagates nominal state and 15x15 error covariance matrix P."""
    p, v, q, ba, bg = state['pos'], state['vel'], state['q'], state['ba'], state['bg']

    acc_unbiased = acc_raw - ba
    gyro_unbiased = gyro_raw - bg

    # Nominal Kinematic Propagation
    R = quaternion_to_rotation_matrix(q)
    acc_world = R @ acc_unbiased - g_world

    p_next = p + v * dt + 0.5 * acc_world * (dt**2)
    v_next = v + acc_world * dt

    omega = gyro_unbiased
    omega_norm = np.linalg.norm(omega)
    angle = omega_norm * dt
    if omega_norm > 1e-8:
        axis = omega / omega_norm
        dq_vec = axis * np.sin(0.5 * angle)
        dq_w = np.cos(0.5 * angle)
    else:
        dq_vec = 0.5 * omega * dt
        dq_w = 1.0

    dq = np.array([dq_vec[0], dq_vec[1], dq_vec[2], dq_w], dtype=np.float64)
    q_next = quaternion_multiply(q, dq)
    q_next /= np.linalg.norm(q_next)

    updated_state = {
        'pos': p_next, 'vel': v_next, 'q': q_next,
        'ba': ba.copy(), 'bg': bg.copy()
    }

    # Covariance Propagation
    F_x = np.eye(15, dtype=np.float64)
    F_x[0:3, 3:6] = np.eye(3) * dt
    F_x[3:6, 6:9] = -R @ skew_symmetric(acc_unbiased) * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[6:9, 12:15] = -R * dt

    Fi = np.zeros((15, 12), dtype=np.float64)
    Fi[3:6, 0:3] = -R
    Fi[6:9, 3:6] = -np.eye(3)
    Fi[9:12, 6:9] = np.eye(3)
    Fi[12:15, 9:12] = np.eye(3)

    sigma_a, sigma_g = 0.1, 0.01
    sigma_ba, sigma_bg = 0.001, 0.0001
    Q_vec = np.array([
        sigma_a**2, sigma_a**2, sigma_a**2,
        sigma_g**2, sigma_g**2, sigma_g**2,
        sigma_ba**2, sigma_ba**2, sigma_ba**2,
        sigma_bg**2, sigma_bg**2, sigma_bg**2
    ])
    Q = np.diag(Q_vec) * dt

    P_next = F_x @ P @ F_x.T + Fi @ Q @ Fi.T
    return updated_state, P_next


def eskf_inject_error(state, delta_x, P):
    """Injects 15x1 error vector into state and resets covariance P."""
    dp, dv, dtheta = delta_x[0:3], delta_x[3:6], delta_x[6:9]
    dba, dbg = delta_x[9:12], delta_x[12:15]

    pos_c = state['pos'] + dp
    vel_c = state['vel'] + dv
    ba_c = state['ba'] + dba
    bg_c = state['bg'] + dbg

    angle = np.linalg.norm(dtheta)
    if angle > 1e-8:
        axis = dtheta / angle
        dq_vec = axis * np.sin(0.5 * angle)
        dq_w = np.cos(0.5 * angle)
    else:
        dq_vec = 0.5 * dtheta
        dq_w = 1.0

    dq = np.array([dq_vec[0], dq_vec[1], dq_vec[2], dq_w], dtype=np.float64)
    q_c = quaternion_multiply(state['q'], dq)
    q_c /= np.linalg.norm(q_c)

    corrected_state = {'pos': pos_c, 'vel': vel_c, 'q': q_c, 'ba': ba_c, 'bg': bg_c}

    G = np.eye(15, dtype=np.float64)
    G[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(dtheta)
    P_reset = G @ P @ G.T

    return corrected_state, P_reset


# ==============================================================================
# 3. MASTER EXECUTION LOOP
# ==============================================================================

def run_master_eskf_pipeline(imu_stream):
    """
    Processes entire IMU dataset through the ESKF engine.
    
    imu_stream: Nx7 numpy array [timestamp, ax, ay, az, gx, gy, gz]
    Returns: Nx8 numpy array [timestamp, x, y, z, qx, qy, qz, qw]
    """
    validate_imu_stream(imu_stream)

    imu_stream = repair_nan_values(
        imu_stream
    )
    num_frames = imu_stream.shape[0]
    
    # --------------------------------------------------------------------------
    # Step A: Initialize Trajectory Output Matrix (Memory pre-allocated)
    # --------------------------------------------------------------------------
    trajectory_output = np.zeros((num_frames, 8), dtype=np.float64)
    
    # Step B: Auto-Alignment at T=0
    q0 = initialize_alignment(imu_stream, num_samples=50)
    
    # Initialize Nominal State
    state = {
        'pos': np.zeros(3, dtype=np.float64),
        'vel': np.array([0, 51.444444355, 0]),
        'q': q0,
        'ba': np.zeros(3, dtype=np.float64),
        'bg': np.zeros(3, dtype=np.float64)
    }
    
    # Initialize 15x15 Covariance Matrix P
    P = np.eye(15, dtype=np.float64) * 0.01
    P[0:3, 0:3] *= 0.001 # Position uncertainty
    P[3:6, 3:6] *= 0.01 # Velocity uncertainty
    P[6:9, 6:9] *= 0.001 # Orientation uncertainty

    # Log initial state T=0
    trajectory_output[0, 0] = imu_stream[0, 0]
    trajectory_output[0, 1:4] = state['pos']
    trajectory_output[0, 4:8] = state['q']

    # --------------------------------------------------------------------------
    # Step C: Sequential IMU Integration Loop
    # --------------------------------------------------------------------------
    freeze_counter = 0
    EXPECTED_DT = 0.0085
    for i in range(1, num_frames):
        t_prev = imu_stream[i - 1, 0]
        t_curr = imu_stream[i, 0]
        dt = t_curr - t_prev

        dropout_detected = (
            dt > 5 * EXPECTED_DT
        )

        if dropout_detected:

            print(f"Dropout detected at frame {i}")
            P *= 2.0
        if dt > 1.0:

            print(f"Long blackout detected at frame {i}")
            P *= 10.0

            trajectory_output[i, 0] = t_curr
            trajectory_output[i, 1:4] = state['pos']
            trajectory_output[i, 4:8] = state['q']
            continue
        
        # Guard against zero or corrupted time step deltas
        if dt <= 0:
            dt = 0.005 # Default 200 Hz fallback

        acc_raw = imu_stream[i, 1:4]
        gyro_raw = imu_stream[i, 4:7]
        # Sensor Freeze detection

        if np.allclose(
            imu_stream[i, 1:7],
            imu_stream[i-1,1:7], atol=1e-10):
            freeze_counter +=1
        else:
            freeze_counter =0
        if freeze_counter > 100:
            print(f"Sensor freeze detected at frame {i}")
        # Accelerometer and gyroscope  blackout detection
        acc_norm = np.linalg.norm(acc_raw)
        if acc_norm < 0.1:
            print(f"Accelerometer blackout detected at frame {i}")
            continue
        gyro_norm = np.linalg.norm(gyro_raw)
        # if gyro_norm < 1e-5:
        #     print(f"Gyroscope blackout detected at frame {i}")

        MAX_ACCEL = 50.0
        MAX_GYRO = 20.0

        if np.linalg.norm(acc_raw) > MAX_ACCEL:
            print(f"Accelerometer outlier at frame {i}")
            continue

        if np.linalg.norm(gyro_raw) > MAX_GYRO:
            print(f"Gyroscope outlier at frame {i}")
            continue

        # 1. State & Covariance Prediction
        state, P = eskf_predict(state, acc_raw, gyro_raw, P, dt)
        #Zero Velocity Update(ZUPT) detection
        stationary= (abs(acc_norm-9.81)<0.15) and (gyro_norm<0.02)
        if stationary:
            state['vel'][:] =0.0
        
        # 3. Store Computed Pose in Hackathon CSV Output Format
        trajectory_output[i, 0] = t_curr
        trajectory_output[i, 1:4] = state['pos']
        trajectory_output[i, 4:8] = state['q']

    return trajectory_output


# ==============================================================================
# 4. SIMULATED TEST RUN
# ==============================================================================
if __name__ == "__main__":
    # Simulate 5 seconds of 200 Hz IMU data (1000 rows)
    N = 1000
    timestamps = np.linspace(0, 5, N)
    mock_accel = np.tile([0.05, 0.02, 9.80], (N, 1)) # Slightly tilted static gravity
    mock_gyro = np.tile([0.001, -0.002, 0.0], (N, 1)) # Minor gyro drift

    mock_imu_dataset = np.column_stack((timestamps, mock_accel, mock_gyro))

    # Run complete engine
    estimated_trajectory = run_master_eskf_pipeline(mock_imu_dataset)

    print("Master ESKF Pipeline Execution Successful!")
    print(f"Processed Trajectory Shape: {estimated_trajectory.shape}")
    print("Sample Output Row [timestamp, x, y, z, qx, qy, qz, qw]:")
    print(estimated_trajectory[-1])