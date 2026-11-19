#!/usr/bin/env python3
"""
estimate_trajectory.py - Robust ESKF IMU Trajectory Estimator for degraded IMU recordings.

Input:
    CSV file with 7 columns: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    (or EuRoC order: timestamp, w_x, w_y, w_z, a_x, a_y, a_z — auto-detected)

Output:
    CSV file with 8 columns: time,tx,ty,tz,qx,qy,qz,qw
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
# 2. INPUT DETECTION & PRE-CLEANING
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


def repair_and_preprocess(imu_stream, ts_unit="auto", accel_unit="auto", gyro_deg=False, exact_int_ts=None):
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
    if exact_int_ts is not None:
        ts_scale = detect_ts_scale(exact_int_ts, ts_unit)
        ts_sec = (exact_int_ts - exact_int_ts[0]).astype(np.float64) * ts_scale
    else:
        ts_scale = detect_ts_scale(raw_ts, ts_unit)
        ts_sec = (raw_ts - raw_ts[0]) * ts_scale

    acc_scale = detect_accel_scale(data[:, 1:4], accel_unit)
    gyr_scale = np.radians(1.0) if gyro_deg else 1.0

    data[:, 1:4] *= acc_scale
    data[:, 4:7] *= gyr_scale

    # 3-sample rolling median filter to crush single-sample spikes before ESKF
    clean_sensors = data[:, 1:7].copy()
    for i in range(1, len(data) - 1):
        if np.linalg.norm(data[i, 1:7]) < 1e-4:
            continue
        jump_prev = np.linalg.norm(data[i, 1:4] - data[i - 1, 1:4])
        jump_next = np.linalg.norm(data[i, 1:4] - data[i + 1, 1:4])
        if jump_prev > 2.5 and jump_next > 2.5:
            clean_sensors[i, 0:3] = np.median(data[i - 1:i + 2, 1:4], axis=0)

        g_jump_prev = np.linalg.norm(data[i, 4:7] - data[i - 1, 4:7])
        g_jump_next = np.linalg.norm(data[i, 4:7] - data[i + 1, 4:7])
        if g_jump_prev > 0.35 and g_jump_next > 0.35:
            clean_sensors[i, 3:6] = np.median(data[i - 1:i + 2, 4:7], axis=0)

    data[:, 1:7] = clean_sensors
    return raw_ts, ts_sec, data, ts_scale


# ==============================================================================
# 3. ESKF CORE ENGINE
# ==============================================================================

def initialize_alignment(data, ts_sec):
    """Singularity-free gravity alignment + full gyro bias initialization."""
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
    """Lightweight gravity-vector roll/pitch correction during quasi-constant velocity."""
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


def eskf_velocity_damping_update(state, P, sigma_v=2.5):
    """
    Soft zero-mean velocity prior for bounded-workspace platforms.
    Prevents long-term unobservable attitude drift from integrating gravity into runaway velocity.
    """
    r_vel = -state["vel"]
    H = np.zeros((3, 15), dtype=np.float64)
    H[0:3, 3:6] = np.eye(3)

    R_meas = np.eye(3) * (sigma_v ** 2)
    S = H @ P @ H.T + R_meas
    K = P @ H.T @ np.linalg.inv(S)
    delta_x = K @ r_vel

    I_KH = np.eye(15) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
    return eskf_inject_error(state, delta_x, P_upd)


def is_stationary_shoe(acc_win, gyr_win, ba, bg, sigma_a=0.15, sigma_g=0.015, gamma=4.0):
    """Stance Hypothesis Optimal Estimation (SHOE) detector over a sliding window."""
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

def run_trajectory_estimator(imu_stream, ts_unit="auto", accel_unit="auto", gyro_deg=False, exact_int_ts=None):
    raw_ts, ts_sec, data, ts_scale = repair_and_preprocess(
        imu_stream, ts_unit, accel_unit, gyro_deg, exact_int_ts=exact_int_ts
    )
    n = len(data)
    diffs = np.diff(ts_sec)
    expected_dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.005

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
    frozen_count = 0

    shoe_win = max(15, min(40, round(0.25 / expected_dt)))
    acc_buf = [data[0, 1:4].copy()]
    gyr_buf = [data[0, 4:7].copy()]
    vel_buf = [state["vel"].copy()]

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
            # Rapidly damp drifted velocity to zero (tau ~ 0.08 s) so EOF blackout doesn't overshoot
            state["vel"] *= np.exp(-12.0 * dt)
            if np.linalg.norm(state["vel"]) < 0.05:
                state["vel"][:] = 0.0
            state["pos"] += state["vel"] * dt
            P[0:9, 0:9] *= 1.01
            trajectory_output[i, 0] = raw_ts[i]
            trajectory_output[i, 1:4] = state["pos"]
            trajectory_output[i, 4:8] = state["q"]
            continue

        # 3. Stuck / Frozen Sample Detection
        is_frozen = np.allclose(data[i, 1:7], data[i - 1, 1:7], atol=1e-9)
        if is_frozen:
            frozen_count += 1
        else:
            frozen_count = 0

        if frozen_count >= 3:
            # Sensor stream is dead/stuck: damp velocity instead of integrating stale acceleration
            state["vel"] *= np.exp(-5.0 * dt)
            if np.linalg.norm(state["vel"]) < 0.05:
                state["vel"][:] = 0.0
            state["pos"] += state["vel"] * dt
            P[0:9, 0:9] *= 1.005
            trajectory_output[i, 0] = raw_ts[i]
            trajectory_output[i, 1:4] = state["pos"]
            trajectory_output[i, 4:8] = state["q"]
            continue

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

        # 6. Measurement Updates (Gated ZUPT vs. In-Flight Tilt & Velocity Damping)
        acc_unbiased_norm = np.linalg.norm(acc_raw - state["ba"])
        gyr_unbiased_norm = np.linalg.norm(gyro_raw - state["bg"])
        est_speed = np.linalg.norm(state["vel"])

        if not is_frozen:
            if est_speed < 1.5 and is_stationary_shoe(
                np.array(acc_buf), np.array(gyr_buf), state["ba"], state["bg"]
            ):
                state, P = eskf_zupt_update(state, acc_raw, gyro_raw, P)
            else:
                if abs(acc_unbiased_norm - G) < 0.12 and gyr_unbiased_norm < 0.08:
                    state, P = eskf_gravity_tilt_update(state, acc_raw, P)
                state, P = eskf_velocity_damping_update(state, P, sigma_v=2.5)

        # 7. Log Output
        trajectory_output[i, 0] = raw_ts[i]
        trajectory_output[i, 1:4] = state["pos"]
        trajectory_output[i, 4:8] = state["q"]

        vel_buf.append(state["vel"].copy())
        if len(vel_buf) > shoe_win:
            vel_buf.pop(0)

    state["vel"] = np.median(vel_buf, axis=0)
    return trajectory_output, state, expected_dt, ts_scale


def predict_future_5s(final_state, last_timestamp, imu_dt,
                      horizon=5.0, prev_q=None, ts_scale=1.0, decay_rate=0.75):
    """Exponentially damped velocity, constant-orientation prediction."""
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

    if prev_q is not None and np.dot(quaternion, np.asarray(prev_q, float)) < 0:
        quaternion = -quaternion

    # Clamp small residual drift velocity to zero if platform was nearly stopped
    if np.linalg.norm(velocity) < 0.35:
        velocity = np.zeros(3, dtype=float)

    n = int(np.floor(horizon / imu_dt + 1e-9))
    times_sec = imu_dt * np.arange(1, n + 1)
    if len(times_sec) == 0 or horizon - times_sec[-1] > 1e-9:
        times_sec = np.append(times_sec, horizon)
    else:
        times_sec[-1] = horizon

    # Integral of v0 * exp(-decay_rate * t) dt
    if decay_rate > 1e-6:
        eff_dt = (1.0 - np.exp(-decay_rate * times_sec)) / decay_rate
    else:
        eff_dt = times_sec

    positions = position + np.outer(eff_dt, velocity)
    raw_future_ts = last_timestamp + (times_sec / ts_scale)

    traj = np.column_stack([
        raw_future_ts,
        positions,
        np.tile(quaternion, (len(times_sec), 1))
    ])
    return traj


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
    ap.add_argument("--no-future", action="store_true", help="Disable appending 5s future prediction")
    args = ap.parse_args()

    header = detect_header(args.input)
    df = pd.read_csv(args.input, header=header)
    if df.shape[1] != 7:
        sys.exit(f"Expected 7 columns, got {df.shape[1]}")

    ts_col = df.iloc[:, 0]
    is_int_ts = np.all(np.isclose(ts_col, np.round(ts_col), atol=1e-3))
    exact_raw_ts = np.round(ts_col.to_numpy(dtype=np.float64)).astype(np.int64) if is_int_ts else None

    imu_array = df.to_numpy(dtype=np.float64)
    traj, final_state, imu_dt, ts_scale = run_trajectory_estimator(
        imu_array,
        ts_unit=args.ts_unit,
        accel_unit=args.accel_unit,
        gyro_deg=args.gyro_deg,
        exact_int_ts=exact_raw_ts,
    )

    if args.no_future:
        complete_traj = traj
        all_int_ts = exact_raw_ts
    else:
        last_ts = int(exact_raw_ts[-1]) if exact_raw_ts is not None else float(traj[-1, 0])
        future_traj = predict_future_5s(
            final_state=final_state,
            last_timestamp=last_ts,
            imu_dt=imu_dt,
            horizon=5.0,
            prev_q=final_state["q"],
            ts_scale=ts_scale,
        )
        complete_traj = np.vstack([traj, future_traj])
        if exact_raw_ts is not None:
            n_steps = len(future_traj)
            times_sec = imu_dt * np.arange(1, n_steps + 1)
            times_sec[-1] = min(times_sec[-1], 5.0)
            future_int_ts = exact_raw_ts[-1] + np.round(times_sec / ts_scale).astype(np.int64)
            all_int_ts = np.concatenate([exact_raw_ts, future_int_ts])
        else:
            all_int_ts = None

    out_cols = ["time", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    out_df = pd.DataFrame(complete_traj, columns=out_cols)
    if all_int_ts is not None:
        out_df["time"] = all_int_ts

    out_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved trajectory ({len(out_df)} rows) -> {args.output}")


if __name__ == "__main__":
    main()