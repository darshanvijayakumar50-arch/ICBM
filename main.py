#!/usr/bin/env python3
"""
estimate_trajectory.py - Robust ESKF IMU Trajectory Estimator for degraded IMU recordings.

Input:
    CSV file with 7 columns: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    (or EuRoC order: timestamp, w_x, w_y, w_z, a_x, a_y, a_z — auto-detected)

Output:
    CSV file with 8 columns: time,tx,ty,tz,qx,qy,qz,qw
    (where tx, ty, tz represent estimated 3D position x, y, z in meters, and qx..qw is orientation)

Usage:
    python estimate_trajectory.py <degraded_input.csv> <trajectory_output.csv> [--gyro-deg]
"""

import argparse
import sys
import numpy as np
import pandas as pd

G = 9.80665


# ==============================================================================
# 1. QUATERNION & ROTATION MATH
# ==============================================================================

def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    q = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    ], dtype=np.float64)
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    qxx, qyy, qzz = qx * qx, qy * qy, qz * qz
    qxy, qxz, qyz = qx * qy, qx * qz, qy * qz
    qwx, qwy, qwz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qwz),       2.0 * (qxz + qwy)],
        [2.0 * (qxy + qwz),       1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qwx)],
        [2.0 * (qxz - qwy),       2.0 * (qyz + qwx),       1.0 - 2.0 * (qxx + qyy)]
    ], dtype=np.float64)


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=np.float64)


def skew_symmetric(v):
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)


# ==============================================================================
# 2. INPUT DETECTION & PRE-CLEANING (MATCHING degrade_imu.py)
# ==============================================================================

def detect_header(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip().split(",")[0]
    try:
        float(first)
        return None
    except ValueError:
        return 0


def detect_ts_scale(ts, override="auto"):
    table = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if override != "auto":
        return table[override]
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    dt = np.median(diffs) if len(diffs) > 0 else 0.005
    if dt < 1.0:
        return 1.0
    if dt < 1e3:
        return 1e-3
    if dt < 1e6:
        return 1e-6
    return 1e-9


def detect_accel_scale(acc, override="auto"):
    if override == "ms2":
        return 1.0
    if override == "g":
        return G
    norms = np.linalg.norm(acc, axis=1)
    valid_norms = norms[norms > 0.1]
    norm = np.median(valid_norms) if len(valid_norms) > 0 else G
    if 0.5 < norm < 2.0:
        return G
    return 1.0


def repair_and_preprocess(imu_stream, ts_unit="auto", accel_unit="auto", gyro_deg=False):
    if imu_stream.ndim != 2 or imu_stream.shape[1] != 7:
        raise ValueError("Expected Nx7 array: [timestamp, ax, ay, az, gx, gy, gz]")
    if len(imu_stream) < 2:
        raise ValueError("Insufficient IMU samples")

    data = imu_stream.astype(np.float64).copy()

    # Interpolate any NaNs
    for col in range(data.shape[1]):
        nan_mask = np.isnan(data[:, col])
        if np.any(nan_mask):
            valid = np.where(~nan_mask)[0]
            if len(valid) == 0:
                raise ValueError(f"Column {col} contains only NaNs")
            data[nan_mask, col] = np.interp(np.where(nan_mask)[0], valid, data[valid, col])

    # Auto-swap columns if input is in EuRoC order: [ts, gx, gy, gz, ax, ay, az]
    n14 = np.linalg.norm(data[:, 1:4], axis=1)
    n47 = np.linalg.norm(data[:, 4:7], axis=1)
    med_14 = np.median(n14[n14 > 1e-6]) if np.any(n14 > 1e-6) else 0.0
    med_47 = np.median(n47[n47 > 1e-6]) if np.any(n47 > 1e-6) else 0.0
    if med_14 < 0.5 and (0.5 < med_47 < 20.0):
        data[:, [1, 2, 3, 4, 5, 6]] = data[:, [4, 5, 6, 1, 2, 3]]

    raw_ts = data[:, 0].copy()
    ts_scale = detect_ts_scale(raw_ts, ts_unit)
    acc_scale = detect_accel_scale(data[:, 1:4], accel_unit)
    gyr_scale = np.radians(1.0) if gyro_deg else 1.0

    ts_sec = (raw_ts - raw_ts[0]) * ts_scale
    data[:, 1:4] *= acc_scale
    data[:, 4:7] *= gyr_scale

    # 3-sample rolling median filter to crush single-sample spikes before ESKF
    # Only applied to non-zero regions so we don't smear true start/end values
    clean_sensors = data[:, 1:7].copy()
    for i in range(1, len(data) - 1):
        # Skip if part of total zero-out failure
        if np.linalg.norm(data[i, 1:7]) < 1e-4:
            continue
        # Only median-filter if there's a sharp jump (preserves smooth true dynamics)
        jump_prev = np.linalg.norm(data[i, 1:4] - data[i - 1, 1:4])
        jump_next = np.linalg.norm(data[i, 1:4] - data[i + 1, 1:4])
        if jump_prev > 2.5 and jump_next > 2.5:
            clean_sensors[i, 0:3] = np.median(data[i - 1:i + 2, 1:4], axis=0)

        g_jump_prev = np.linalg.norm(data[i, 4:7] - data[i - 1, 4:7])
        g_jump_next = np.linalg.norm(data[i, 4:7] - data[i + 1, 4:7])
        if g_jump_prev > 0.35 and g_jump_next > 0.35:
            clean_sensors[i, 3:6] = np.median(data[i - 1:i + 2, 4:7], axis=0)

    data[:, 1:7] = clean_sensors
    return raw_ts, ts_sec, data


# ==============================================================================
# 3. ESKF CORE ENGINE
# ==============================================================================

def initialize_alignment(data, ts_sec):
    """
    Singularity-free gravity alignment + full gyro bias initialization.
    Works for any IMU mounting orientation (e.g., EuRoC where X-axis points upward).
    """
    init_mask = ts_sec <= 1.5
    if np.sum(init_mask) < 10:
        init_mask = np.arange(min(50, len(data)))

    acc_init = data[init_mask, 1:4]
    gyr_init = data[init_mask, 4:7]

    mean_acc = np.median(acc_init, axis=0)
    a_norm = np.linalg.norm(mean_acc)
    if a_norm < 1e-6:
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        # Shortest-arc rotation aligning measured specific force u_b to world Z-up [0, 0, 1]
        u_b = mean_acc / a_norm
        u_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        dot = np.clip(np.dot(u_b, u_w), -1.0, 1.0)
        if dot > 0.999999:
            q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        elif dot < -0.999999:
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = np.cross(u_b, u_w)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(dot)
            q0 = np.append(axis * np.sin(0.5 * angle), np.cos(0.5 * angle))
            q0 /= np.linalg.norm(q0)

    # Allow initial gyro biases up to 0.25 rad/s (EuRoC Z-gyro bias is ~0.082 rad/s)
    med_gyr = np.median(gyr_init, axis=0)
    bg0 = med_gyr if np.linalg.norm(med_gyr) < 0.25 else np.zeros(3, dtype=np.float64)
    return q0, bg0


def eskf_predict(state, acc_raw, gyro_raw, P, dt):
    g_world = np.array([0.0, 0.0, G], dtype=np.float64)
    p, v, q, ba, bg = state["pos"], state["vel"], state["q"], state["ba"], state["bg"]

    acc_unbiased = acc_raw - ba
    gyro_unbiased = gyro_raw - bg

    R = quaternion_to_rotation_matrix(q)
    acc_world = R @ acc_unbiased - g_world

    p_next = p + v * dt + 0.5 * acc_world * (dt ** 2)
    v_next = v + acc_world * dt

    omega_norm = np.linalg.norm(gyro_unbiased)
    angle = omega_norm * dt
    if omega_norm > 1e-8:
        axis = gyro_unbiased / omega_norm
        dq = np.append(axis * np.sin(0.5 * angle), np.cos(0.5 * angle))
    else:
        dq = np.append(0.5 * gyro_unbiased * dt, 1.0)

    q_next = quaternion_multiply(q, dq)
    q_next /= np.linalg.norm(q_next)

    updated_state = {
        "pos": p_next,
        "vel": v_next,
        "q": q_next,
        "ba": ba.copy(),
        "bg": bg.copy(),
    }

    F_x = np.eye(15, dtype=np.float64)
    F_x[0:3, 3:6] = np.eye(3) * dt
    F_x[3:6, 6:9] = -R @ skew_symmetric(acc_unbiased) * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[6:9, 12:15] = -np.eye(3) * dt

    Fi = np.zeros((15, 12), dtype=np.float64)
    Fi[3:6, 0:3] = -R
    Fi[6:9, 3:6] = -np.eye(3)
    Fi[9:12, 6:9] = np.eye(3)
    Fi[12:15, 9:12] = np.eye(3)

    sigma_a, sigma_g = 0.22, 0.018
    sigma_ba, sigma_bg = 0.005, 0.0005
    Q_diag = np.array([
        sigma_a**2, sigma_a**2, sigma_a**2,
        sigma_g**2, sigma_g**2, sigma_g**2,
        sigma_ba**2, sigma_ba**2, sigma_ba**2,
        sigma_bg**2, sigma_bg**2, sigma_bg**2,
    ])
    Q = np.diag(Q_diag) * dt

    P_next = F_x @ P @ F_x.T + Fi @ Q @ Fi.T
    return updated_state, P_next


def eskf_inject_error(state, delta_x, P):
    dp, dv, dtheta = delta_x[0:3], delta_x[3:6], delta_x[6:9]
    dba, dbg = delta_x[9:12], delta_x[12:15]

    angle = np.linalg.norm(dtheta)
    if angle > 1e-8:
        axis = dtheta / angle
        dq = np.append(axis * np.sin(0.5 * angle), np.cos(0.5 * angle))
    else:
        dq = np.append(0.5 * dtheta, 1.0)

    q_c = quaternion_multiply(state["q"], dq)
    q_c /= np.linalg.norm(q_c)

    # Clamp biases to physically realistic MEMS bounds so transients can never cause runaway
    ba_new = np.clip(state["ba"] + dba, -1.5, 1.5)
    bg_new = np.clip(state["bg"] + dbg, -0.25, 0.25)

    corrected_state = {
        "pos": state["pos"] + dp,
        "vel": state["vel"] + dv,
        "q": q_c,
        "ba": ba_new,
        "bg": bg_new,
    }

    G_mat = np.eye(15, dtype=np.float64)
    G_mat[6:9, 6:9] = np.eye(3) - 0.5 * skew_symmetric(dtheta)
    P_reset = G_mat @ P @ G_mat.T
    P_reset = 0.5 * (P_reset + P_reset.T)
    return corrected_state, P_reset


def eskf_zupt_update(state, acc_raw, gyro_raw, P):
    """9-DOF Zero-Velocity + Gravity + Zero-Angular-Rate Update with Chi-Square Gating."""
    g_world = np.array([0.0, 0.0, G], dtype=np.float64)
    R = quaternion_to_rotation_matrix(state["q"])
    g_body_pred = R.T @ g_world

    r_vel = -state["vel"]
    r_acc = (acc_raw - state["ba"]) - g_body_pred
    r_gyr = (gyro_raw - state["bg"])
    r = np.concatenate([r_vel, r_acc, r_gyr])

    H = np.zeros((9, 15), dtype=np.float64)
    H[0:3, 3:6] = np.eye(3)
    H[3:6, 6:9] = skew_symmetric(g_body_pred)
    H[3:6, 9:12] = np.eye(3)
    H[6:9, 12:15] = np.eye(3)

    R_meas = np.diag([
        0.05**2, 0.05**2, 0.05**2,
        0.25**2, 0.25**2, 0.25**2,
        0.02**2, 0.02**2, 0.02**2,
    ])

    S = H @ P @ H.T + R_meas
    S_inv = np.linalg.inv(S)

    # Chi-Square (NIS) gate on velocity innovation (3 DOF, 99% threshold = 11.34)
    S_vel = S[0:3, 0:3]
    nis_vel = float(r_vel.T @ np.linalg.inv(S_vel) @ r_vel)
    if nis_vel > 11.34:
        return state, P

    K = P @ H.T @ S_inv
    delta_x = K @ r

    I_KH = np.eye(15) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
    return eskf_inject_error(state, delta_x, P_upd)


def eskf_gravity_tilt_update(state, acc_raw, P):
    """
    Lightweight gravity-vector roll/pitch correction during quasi-constant velocity.
    Does NOT update accelerometer bias (H[:, 9:12] = 0) so sustained flight accelerations
    cannot corrupt state['ba'] and cause quadratic position runaway.
    """
    g_world = np.array([0.0, 0.0, G], dtype=np.float64)
    R = quaternion_to_rotation_matrix(state["q"])
    g_body_pred = R.T @ g_world

    r_acc = (acc_raw - state["ba"]) - g_body_pred
    H = np.zeros((3, 15), dtype=np.float64)
    H[0:3, 6:9] = skew_symmetric(g_body_pred)

    R_meas = np.eye(3) * (0.85 ** 2)
    S = H @ P @ H.T + R_meas
    K = P @ H.T @ np.linalg.inv(S)
    delta_x = K @ r_acc

    I_KH = np.eye(15) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
    return eskf_inject_error(state, delta_x, P_upd)


def is_stationary_shoe(acc_win, gyr_win, ba, bg, sigma_a=0.15, sigma_g=0.015, gamma=4.0):
    """
    Stance Hypothesis Optimal Estimation (SHOE) detector over a sliding window.
    Distinguishes true ground rest from constant-velocity flight via signal variance.
    """
    if len(acc_win) < 10:
        return False

    a_unb = acc_win - ba
    g_unb = gyr_win - bg

    a_mean = np.mean(a_unb, axis=0)
    a_mean_norm = np.linalg.norm(a_mean)
    if a_mean_norm < 1e-6 or abs(a_mean_norm - G) > 0.25:
        return False

    g_ref = G * (a_mean / a_mean_norm)
    acc_term = np.sum((a_unb - g_ref) ** 2, axis=1) / (sigma_a ** 2)
    gyr_term = np.sum(g_unb ** 2, axis=1) / (sigma_g ** 2)

    T_shoe = np.mean(acc_term + gyr_term)
    return T_shoe < gamma


# ==============================================================================
# 4. MASTER TRAJECTORY PIPELINE
# ==============================================================================

def run_trajectory_estimator(imu_stream, ts_unit="auto", accel_unit="auto", gyro_deg=False):
    raw_ts, ts_sec, data = repair_and_preprocess(imu_stream, ts_unit, accel_unit, gyro_deg)
    n = len(data)
    diffs = np.diff(ts_sec)
    expected_dt = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.005

    trajectory_output = np.zeros((n, 8), dtype=np.float64)

    q0, bg0 = initialize_alignment(data, ts_sec)
    state = {
        "pos": np.zeros(3, dtype=np.float64),
        "vel": np.zeros(3, dtype=np.float64),
        "q": q0,
        "ba": np.zeros(3, dtype=np.float64),
        "bg": bg0,
    }

    P = np.eye(15, dtype=np.float64) * 0.01
    P[0:3, 0:3] = np.eye(3) * (0.01 ** 2)
    P[3:6, 3:6] = np.eye(3) * (0.05 ** 2)
    P[6:9, 6:9] = np.eye(3) * (0.05 ** 2)
    P[9:12, 9:12] = np.eye(3) * (0.15 ** 2)
    P[12:15, 12:15] = np.eye(3) * (0.05 ** 2)

    trajectory_output[0, 0] = raw_ts[0]
    trajectory_output[0, 1:4] = state["pos"]
    trajectory_output[0, 4:8] = state["q"]

    last_valid_acc = data[0, 1:4].copy()
    last_valid_gyr = data[0, 4:7].copy()
    consecutive_rejections = 0

    shoe_win = max(15, min(40, int(round(0.25 / expected_dt))))
    acc_buf = [data[0, 1:4].copy()]
    gyr_buf = [data[0, 4:7].copy()]

    for i in range(1, n):
        dt = ts_sec[i] - ts_sec[i - 1]

        # 1. Duplicated rows (dt == 0)
        if dt <= 1e-7:
            trajectory_output[i, 0] = raw_ts[i]
            trajectory_output[i, 1:4] = state["pos"]
            trajectory_output[i, 4:8] = state["q"]
            continue

        acc_raw = data[i, 1:4].copy()
        gyro_raw = data[i, 4:7].copy()

        # 2. Total Sensor Blackout / End-of-File Failure (exact zeros)
        if np.linalg.norm(acc_raw) < 1e-3 and np.linalg.norm(gyro_raw) < 1e-5:
            state["pos"] += state["vel"] * dt
            P[0:9, 0:9] *= 1.02
            trajectory_output[i, 0] = raw_ts[i]
            trajectory_output[i, 1:4] = state["pos"]
            trajectory_output[i, 4:8] = state["q"]
            continue

        # 3. Stuck / Frozen Sample Detection
        is_frozen = np.allclose(data[i, 1:7], data[i - 1, 1:7], atol=1e-9)
        if is_frozen:
            acc_raw = last_valid_acc
            gyro_raw = last_valid_gyr

        # 4. Residual Spike Rejection (with anti-latch counter)
        acc_jump = np.linalg.norm(acc_raw - last_valid_acc)
        gyr_jump = np.linalg.norm(gyro_raw - last_valid_gyr)
        is_spike = (
            (acc_jump > 4.5 and dt < 0.05)
            or (gyr_jump > 0.6 and dt < 0.05)
            or np.linalg.norm(acc_raw) > 60.0
        )
        if is_spike and consecutive_rejections < 4:
            acc_raw = last_valid_acc
            gyro_raw = last_valid_gyr
            consecutive_rejections += 1
        else:
            last_valid_acc = acc_raw.copy()
            last_valid_gyr = gyro_raw.copy()
            consecutive_rejections = 0

        acc_buf.append(acc_raw.copy())
        gyr_buf.append(gyro_raw.copy())
        if len(acc_buf) > shoe_win:
            acc_buf.pop(0)
            gyr_buf.pop(0)

        # 5. State & Covariance Prediction (with sub-stepping across burst dropouts)
        if dt > 4.0 * expected_dt:
            P[0:9, 0:9] *= min(5.0, dt / expected_dt)
            steps = min(25, max(1, int(np.round(dt / expected_dt))))
            sub_dt = dt / steps
            for _ in range(steps):
                state, P = eskf_predict(state, acc_raw, gyro_raw, P, sub_dt)
        else:
            state, P = eskf_predict(state, acc_raw, gyro_raw, P, dt)

        # 6. Measurement Updates (Gated ZUPT vs. In-Flight Tilt-Only Update)
        acc_unbiased_norm = np.linalg.norm(acc_raw - state["ba"])
        gyr_unbiased_norm = np.linalg.norm(gyro_raw - state["bg"])
        est_speed = np.linalg.norm(state["vel"])

        if not is_frozen:
            # ZUPT fires only when SHOE variance confirms rest AND estimated speed is < 1.5 m/s
            if est_speed < 1.5 and is_stationary_shoe(
                np.array(acc_buf), np.array(gyr_buf), state["ba"], state["bg"]
            ):
                state, P = eskf_zupt_update(state, acc_raw, gyro_raw, P)
            elif abs(acc_unbiased_norm - G) < 0.12 and gyr_unbiased_norm < 0.08:
                # Safe in-flight roll/pitch stabilization (does not touch velocity or accel bias)
                state, P = eskf_gravity_tilt_update(state, acc_raw, P)

        # 7. Log Output
        trajectory_output[i, 0] = raw_ts[i]
        trajectory_output[i, 1:4] = state["pos"]
        trajectory_output[i, 4:8] = state["q"]

    return trajectory_output


def predict_future_5s(final_state, last_timestamp, imu_dt,
                      horizon=5.0, prev_q=None):
    """
    Constant-velocity, constant-orientation prediction.
    final_state: pos [x,y,z], vel [vx,vy,vz] (same frame as pos),
                 q [qx,qy,qz,qw] (XYZW)
    Returns rows: [t, tx, ty, tz, qx, qy, qz, qw]  (TUM format)
    """
    position = np.asarray(final_state["pos"], dtype=float).reshape(-1)
    velocity = np.asarray(final_state["vel"], dtype=float).reshape(-1)
    quaternion = np.asarray(final_state["q"], dtype=float).reshape(-1)

    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("Position and velocity must have 3 elements")
    if quaternion.shape != (4,):
        raise ValueError("Quaternion must have 4 elements")
    if not (np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))
            and np.all(np.isfinite(quaternion))):
        raise ValueError("Non-finite value in final state")
    if not (np.isfinite(imu_dt) and np.isfinite(horizon)
            and np.isfinite(last_timestamp)):
        raise ValueError("Non-finite time value")
    if imu_dt <= 0 or horizon <= 0:
        raise ValueError("Invalid time interval")

    q_norm = np.linalg.norm(quaternion)
    if q_norm < 1e-12:
        raise ValueError("Invalid zero quaternion")
    quaternion = quaternion / q_norm

    # Keep sign continuous with the previous quaternion (q and -q are equal)
    if prev_q is not None and np.dot(quaternion, np.asarray(prev_q, float)) < 0:
        quaternion = -quaternion

    # Robust time grid: integer count, then make sure the endpoint is exactly horizon
    n = int(np.floor(horizon / imu_dt + 1e-9))
    times = imu_dt * np.arange(1, n + 1)
    if len(times) == 0 or horizon - times[-1] > 1e-9:
        times = np.append(times, horizon)
    else:
        times[-1] = horizon

    positions = position + np.outer(times, velocity)   # (N, 3)

    traj = np.column_stack([
        last_timestamp + times,
        positions,
        np.tile(quaternion, (len(times), 1))
    ])
    return traj.tolist()


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Estimate 3D trajectory from a degraded IMU CSV.")
    ap.add_argument("input", help="Path to degraded IMU CSV")
    ap.add_argument("output", help="Path to output trajectory CSV")
    ap.add_argument("--ts-unit", choices=["auto", "s", "ms", "us", "ns"], default="auto")
    ap.add_argument("--accel-unit", choices=["auto", "ms2", "g"], default="auto")
    ap.add_argument("--gyro-deg", action="store_true", help="Set if input gyro is in deg/s")
    args = ap.parse_args()

    header = detect_header(args.input)
    df = pd.read_csv(args.input, header=header)
    if df.shape[1] != 7:
        sys.exit(f"Expected 7 columns, got {df.shape[1]}")

    imu_array = df.to_numpy(dtype=np.float64)
    traj = run_trajectory_estimator(
        imu_array,
        ts_unit=args.ts_unit,
        accel_unit=args.accel_unit,
        gyro_deg=args.gyro_deg,
    )

    future_traj = predict_future_5s(
        final_state=final_state,
        last_timestamp=timestamps[-1],
        imu_dt=imu_dt,
        horizon=5.0,
        prev_q=final_state["q"]
    )

    # 3. Combine original and future trajectories
    future_traj = np.asarray(future_traj, dtype=np.float64)

    complete_traj = np.vstack([traj, future_traj])


    out_cols = ["time", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    out_df = pd.DataFrame(traj, columns=out_cols)
    out_df["time"] = np.round(traj[:, 0]).astype(np.int64)
    out_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved trajectory ({len(out_df)} rows) -> {args.output}")


if __name__ == "__main__":
    main()