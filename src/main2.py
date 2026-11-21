# Darshan

'''import os
import csv
import math
import time
import argparse


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def rotvec_to_quat(rv):
    """Exponential map from 3D rotation vector [rx, ry, rz] to [w, x, y, z]."""
    rx, ry, rz = rv
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return (1.0, 0.5 * rx, 0.5 * ry, 0.5 * rz)
    half = 0.5 * angle
    s = math.sin(half) / angle
    return (math.cos(half), rx * s, ry * s, rz * s)


def quat_rotate(q, v):
    """Rotates 3D vector v by unit quaternion q = [w, x, y, z]."""
    w, qx, qy, qz = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def quat_rotate_inv(q, v):
    """Rotates 3D vector v by conjugate of unit quaternion q = [w, x, y, z]."""
    return quat_rotate((q[0], -q[1], -q[2], -q[3]), v)


class LowMemoryInertialEstimator:
    """
    Zero-bloat 6-DOF IMU Trajectory Estimator using pure Python math.
    O(1) streaming memory, < 0.02 ms per sample.
    """
    def __init__(self, g_mag=9.80665):
        self.p = [0.0, 0.0, 0.0]
        self.v = [0.0, 0.0, 0.0]
        self.q = (1.0, 0.0, 0.0, 0.0)  # [w, x, y, z]

        self.bg = [0.0, 0.0, 0.0]
        self.ba = [0.0, 0.0, 0.0]
        self.g_mag = g_mag
        self.g = (0.0, 0.0, -g_mag)

        self.acc_world_bias = [0.0, 0.0, 0.0]
        self.last_acc_world = [0.0, 0.0, 0.0]
        self.last_omega = [0.0, 0.0, 0.0]

        self.acc_buf = []
        self.gyro_buf = []

    def calibrate_from_head(self, samples):
        n = len(samples)
        if n < 10:
            return

        win = min(50, n)
        best_i = 0
        min_score = float("inf")

        for i in range(n - win + 1):
            chunk = samples[i:i + win]
            inv_w = 1.0 / win
            mw = [sum(r[0][k] for r in chunk) * inv_w for k in range(3)]
            ma = [sum(r[1][k] for r in chunk) * inv_w for k in range(3)]
            vw = sum(sum((r[0][k] - mw[k]) ** 2 for r in chunk) * inv_w for k in range(3))
            va = sum(sum((r[1][k] - ma[k]) ** 2 for r in chunk) * inv_w for k in range(3))
            score = va + 15.0 * vw
            if score < min_score:
                min_score = score
                best_i = i

        chunk = samples[best_i:best_i + win]
        inv_w = 1.0 / win
        self.bg = [sum(r[0][k] for r in chunk) * inv_w for k in range(3)]
        mean_acc = [sum(r[1][k] for r in chunk) * inv_w for k in range(3)]

        acc_norm = math.sqrt(sum(c * c for c in mean_acc))
        if acc_norm > 1e-3:
            gx, gy, gz = (mean_acc[0] / acc_norm, mean_acc[1] / acc_norm, mean_acc[2] / acc_norm)
            ax, ay = gy, -gx
            an = math.sqrt(ax * ax + ay * ay)
            if an > 1e-6:
                angle = math.acos(max(-1.0, min(1.0, gz)))
                self.q = rotvec_to_quat(((ax / an) * angle, (ay / an) * angle, 0.0))

            exp_body = quat_rotate_inv(self.q, (0.0, 0.0, self.g_mag))
            self.ba = [mean_acc[k] - exp_body[k] for k in range(3)]

    def step(self, dt, omega_m=None, acc_m=None, is_blackout=False):
        if dt <= 0.0:
            return

        steps = max(1, int(math.ceil(dt / 0.005)))
        sub_dt = dt / steps

        for _ in range(steps):
            if not is_blackout and omega_m is not None and acc_m is not None:
                omega = [omega_m[0] - self.bg[0], omega_m[1] - self.bg[1], omega_m[2] - self.bg[2]]
                acc = [acc_m[0] - self.ba[0], acc_m[1] - self.ba[1], acc_m[2] - self.ba[2]]

                self.acc_buf.append(acc)
                self.gyro_buf.append(omega)
                if len(self.acc_buf) > 20:
                    self.acc_buf.pop(0)
                    self.gyro_buf.pop(0)

                acc_std = 1.0
                if len(self.acc_buf) >= 5:
                    inv_n = 1.0 / len(self.acc_buf)
                    ma = [sum(x[k] for x in self.acc_buf) * inv_n for k in range(3)]
                    acc_std = math.sqrt(sum(sum((x[k] - ma[k]) ** 2 for x in self.acc_buf) * inv_n for k in range(3)))

                omega_norm = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
                acc_norm = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
                acc_err = abs(acc_norm - self.g_mag)

                # 1. Midpoint quaternion attitude propagation
                mid_omega = (
                    0.5 * (self.last_omega[0] + omega[0]) * sub_dt,
                    0.5 * (self.last_omega[1] + omega[1]) * sub_dt,
                    0.5 * (self.last_omega[2] + omega[2]) * sub_dt,
                )
                self.q = quat_normalize(quat_multiply(self.q, rotvec_to_quat(mid_omega)))

                # 2. Gentle complementary gravity tilt correction during low-accel phases
                if acc_err < 0.15 and omega_norm < 0.25 and acc_norm > 1e-6:
                    dx, dy, dz = quat_rotate(self.q, (acc[0] / acc_norm, acc[1] / acc_norm, acc[2] / acc_norm))
                    tn = math.sqrt(dy * dy + dx * dx)
                    if tn > 1e-6:
                        alpha = 0.002 * math.asin(max(-1.0, min(1.0, tn))) / tn
                        self.q = quat_normalize(quat_multiply(rotvec_to_quat((dy * alpha, -dx * alpha, 0.0)), self.q))

                # 3. World acceleration + slow EMA bias removal
                aw = quat_rotate(self.q, acc)
                raw_aw = (aw[0] + self.g[0], aw[1] + self.g[1], aw[2] + self.g[2])

                beta = sub_dt / (10.0 + sub_dt)
                for k in range(3):
                    self.acc_world_bias[k] = (1.0 - beta) * self.acc_world_bias[k] + beta * raw_aw[k]

                acc_world = [raw_aw[k] - self.acc_world_bias[k] for k in range(3)]

                # 4. ZUPT + Online bias adaptation + Velocity damping
                is_stationary = (acc_std < 0.10) and (omega_norm < 0.03) and (acc_err < 0.20)
                if is_stationary:
                    self.v = [0.0, 0.0, 0.0]
                    acc_world = [0.0, 0.0, 0.0]
                    exp_body = quat_rotate_inv(self.q, (0.0, 0.0, self.g_mag))
                    for k in range(3):
                        self.bg[k] = 0.995 * self.bg[k] + 0.005 * omega_m[k]
                        self.ba[k] = 0.995 * self.ba[k] + 0.005 * (acc_m[k] - exp_body[k])
                else:
                    damp = math.exp(-0.55 * sub_dt)
                    for k in range(3):
                        self.v[k] *= damp

                half_dt2 = 0.5 * sub_dt * sub_dt
                for k in range(3):
                    self.p[k] += self.v[k] * sub_dt + acc_world[k] * half_dt2
                    self.v[k] += acc_world[k] * sub_dt

                self.last_acc_world = acc_world
                self.last_omega = omega
            else:
                # 2s Dropout / Last 5s Blackout smooth momentum decay
                decay_a = math.exp(-4.0 * sub_dt)
                decay_w = math.exp(-3.5 * sub_dt)
                decay_v = math.exp(-1.5 * sub_dt)
                self.last_acc_world = [self.last_acc_world[k] * decay_a for k in range(3)]
                self.last_omega = [self.last_omega[k] * decay_w for k in range(3)]

                self.q = quat_normalize(
                    quat_multiply(
                        self.q,
                        rotvec_to_quat((self.last_omega[0] * sub_dt, self.last_omega[1] * sub_dt, self.last_omega[2] * sub_dt)),
                    )
                )
                for k in range(3):
                    self.v[k] = self.v[k] * decay_v + self.last_acc_world[k] * sub_dt
                    self.p[k] += self.v[k] * sub_dt


def is_invalid_sample(omega, acc):
    for val in omega + acc:
        if math.isnan(val) or math.isinf(val):
            return True
    return (
        abs(omega[0]) < 1e-12 and abs(omega[1]) < 1e-12 and abs(omega[2]) < 1e-12 and
        abs(acc[0]) < 1e-12 and abs(acc[1]) < 1e-12 and abs(acc[2]) < 1e-12
    )


def run_trajectory_estimation(imu_csv_path, output_csv_path):
    if not os.path.exists(imu_csv_path):
        raise FileNotFoundError(f"Input IMU file not found: {imu_csv_path}")

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pass 1: Read only the first 300 valid rows for static calibration
    head_samples = []
    with open(imu_csv_path, "r", newline="") as f_in:
        reader = csv.reader(f_in)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 7:
                continue
            try:
                omega = (float(row[1]), float(row[2]), float(row[3]))
                acc = (float(row[4]), float(row[5]), float(row[6]))
                if not is_invalid_sample(omega, acc):
                    head_samples.append((omega, acc))
                    if len(head_samples) >= 300:
                        break
            except ValueError:
                continue

    estimator = LowMemoryInertialEstimator(g_mag=9.80665)
    estimator.calibrate_from_head(head_samples)
    del head_samples

    # Pass 2: True O(1) streaming from input CSV directly to output CSV
    prev_ts = None
    count = 0
    total_ms = 0.0

    with open(imu_csv_path, "r", newline="") as f_in, open(output_csv_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        next(reader, None)
        writer.writerow([
            "#timestamp",
            "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
            "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
            "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]",
        ])

        for row in reader:
            if not row or len(row) < 7:
                continue

            t0 = time.perf_counter()
            curr_ts = int(float(row[0]))
            dt = 0.0 if prev_ts is None else (curr_ts - prev_ts) * 1e-9
            prev_ts = curr_ts

            try:
                omega_m = (float(row[1]), float(row[2]), float(row[3]))
                acc_m = (float(row[4]), float(row[5]), float(row[6]))
                dead = is_invalid_sample(omega_m, acc_m)
            except ValueError:
                omega_m, acc_m, dead = None, None, True

            if dt > 0.0:
                estimator.step(dt, omega_m=omega_m, acc_m=acc_m, is_blackout=(dead or dt > 0.05))

            qw, qx, qy, qz = estimator.q
            writer.writerow([
                curr_ts,
                f"{estimator.p[0]:.9f}", f"{estimator.p[1]:.9f}", f"{estimator.p[2]:.9f}",
                f"{qw:.9f}", f"{qx:.9f}", f"{qy:.9f}", f"{qz:.9f}",
                f"{estimator.v[0]:.9f}", f"{estimator.v[1]:.9f}", f"{estimator.v[2]:.9f}",
            ])

            total_ms += (time.perf_counter() - t0) * 1e3
            count += 1

    print(f"Saved {count} poses to '{output_csv_path}' (avg latency: {total_ms / max(1, count):.4f} ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imu", help="Input degraded IMU CSV")
    parser.add_argument("output_est", help="Output estimated trajectory CSV")
    args = parser.parse_args()
    run_trajectory_estimation(args.input_imu, args.output_est)'''
import os
import csv
import math
import time
import argparse
from collections import deque


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def rotvec_to_quat(rv):
    """Exponential map from 3D rotation vector [rx, ry, rz] to [w, x, y, z]."""
    rx, ry, rz = rv
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return (1.0, 0.5 * rx, 0.5 * ry, 0.5 * rz)
    half = 0.5 * angle
    s = math.sin(half) / angle
    return (math.cos(half), rx * s, ry * s, rz * s)


def quat_rotate(q, v):
    """Rotates 3D vector v by unit quaternion q = [w, x, y, z]."""
    w, qx, qy, qz = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def quat_rotate_inv(q, v):
    """Rotates 3D vector v by conjugate of unit quaternion q = [w, x, y, z]."""
    return quat_rotate((q[0], -q[1], -q[2], -q[3]), v)


def median3(a, b, c):
    """Fast branch-minimal median of 3 floats for 1-sample spike rejection."""
    return max(min(a, b), min(max(a, b), c))


class LowMemoryInertialEstimator:
    """
    Hardened O(1) streaming 6-DOF IMU Trajectory Estimator.
    Resilient against white noise, random walk bias/scale drift,
    spikes, stuck samples, packet drops, duplicate rows, and sensor blackouts.
    """
    def __init__(self, g_mag=9.80665):
        self.p = [0.0, 0.0, 0.0]
        self.v = [0.0, 0.0, 0.0]
        self.q = (1.0, 0.0, 0.0, 0.0)  # [w, x, y, z]

        self.bg = [0.0, 0.0, 0.0]
        self.ba = [0.0, 0.0, 0.0]
        self.g_mag = g_mag
        self.g = (0.0, 0.0, -g_mag)

        self.acc_world_bias = [0.0, 0.0, 0.0]
        self.last_acc_world = [0.0, 0.0, 0.0]
        self.last_omega = [0.0, 0.0, 0.0]

        self.acc_buf = deque(maxlen=25)
        self.gyro_buf = deque(maxlen=25)

        # 3-tap history for single-sample spike filtering
        self.raw_acc_hist = deque(maxlen=3)
        self.raw_gyro_hist = deque(maxlen=3)

        # Stuck-sensor detection state
        self.prev_raw_acc = None
        self.prev_raw_gyro = None
        self.stuck_streak = 0

    def calibrate_from_head(self, samples):
        """
        Finds the lowest-variance stationary window within the protected
        initial period (first 2.0s) to estimate initial gyro/accel bias and tilt.
        """
        n = len(samples)
        if n < 10:
            return

        win = min(50, n)
        best_i = 0
        min_score = float("inf")

        for i in range(n - win + 1):
            chunk = samples[i:i + win]
            inv_w = 1.0 / win
            mw = [sum(r[0][k] for r in chunk) * inv_w for k in range(3)]
            ma = [sum(r[1][k] for r in chunk) * inv_w for k in range(3)]
            vw = sum(sum((r[0][k] - mw[k]) ** 2 for r in chunk) * inv_w for k in range(3))
            va = sum(sum((r[1][k] - ma[k]) ** 2 for r in chunk) * inv_w for k in range(3))
            score = va + 15.0 * vw
            if score < min_score:
                min_score = score
                best_i = i

        chunk = samples[best_i:best_i + win]
        inv_w = 1.0 / win
        self.bg = [sum(r[0][k] for r in chunk) * inv_w for k in range(3)]
        mean_acc = [sum(r[1][k] for r in chunk) * inv_w for k in range(3)]

        acc_norm = math.sqrt(sum(c * c for c in mean_acc))
        if acc_norm > 1e-3:
            gx, gy, gz = (mean_acc[0] / acc_norm, mean_acc[1] / acc_norm, mean_acc[2] / acc_norm)
            ax, ay = gy, -gx
            an = math.sqrt(ax * ax + ay * ay)
            if an > 1e-6:
                angle = math.acos(max(-1.0, min(1.0, gz)))
                self.q = rotvec_to_quat(((ax / an) * angle, (ay / an) * angle, 0.0))

            exp_body = quat_rotate_inv(self.q, (0.0, 0.0, self.g_mag))
            self.ba = [mean_acc[k] - exp_body[k] for k in range(3)]

    def _detect_stuck(self, omega_m, acc_m):
        """Detects frozen sensor outputs (exact repeat across consecutive rows)."""
        if self.prev_raw_acc is not None and self.prev_raw_gyro is not None:
            same_a = all(abs(acc_m[k] - self.prev_raw_acc[k]) < 1e-9 for k in range(3))
            same_g = all(abs(omega_m[k] - self.prev_raw_gyro[k]) < 1e-9 for k in range(3))
            if same_a and same_g:
                self.stuck_streak += 1
            else:
                self.stuck_streak = 0
        self.prev_raw_acc = acc_m
        self.prev_raw_gyro = omega_m
        return self.stuck_streak >= 1

    def _despike(self, omega_m, acc_m):
        """3-point median filter eliminates single-sample impulse spikes."""
        self.raw_gyro_hist.append(omega_m)
        self.raw_acc_hist.append(acc_m)
        if len(self.raw_acc_hist) < 3:
            return omega_m, acc_m

        g0, g1, g2 = self.raw_gyro_hist
        a0, a1, a2 = self.raw_acc_hist
        clean_g = (
            median3(g0[0], g1[0], g2[0]),
            median3(g0[1], g1[1], g2[1]),
            median3(g0[2], g1[2], g2[2]),
        )
        clean_a = (
            median3(a0[0], a1[0], a2[0]),
            median3(a0[1], a1[1], a2[1]),
            median3(a0[2], a1[2], a2[2]),
        )
        return clean_g, clean_a

    def _propagate_blackout(self, dt):
        """Smoothly decays velocity and angular rate during burst drops or final failure."""
        if dt <= 0.0:
            return
        steps = max(1, int(math.ceil(dt / 0.005)))
        sub_dt = dt / steps

        decay_a = math.exp(-4.0 * sub_dt)
        decay_w = math.exp(-3.5 * sub_dt)
        decay_v = math.exp(-1.5 * sub_dt)

        for _ in range(steps):
            self.last_acc_world = [self.last_acc_world[k] * decay_a for k in range(3)]
            self.last_omega = [self.last_omega[k] * decay_w for k in range(3)]
            self.q = quat_normalize(
                quat_multiply(
                    self.q,
                    rotvec_to_quat((
                        self.last_omega[0] * sub_dt,
                        self.last_omega[1] * sub_dt,
                        self.last_omega[2] * sub_dt,
                    )),
                )
            )
            for k in range(3):
                self.v[k] = self.v[k] * decay_v + self.last_acc_world[k] * sub_dt
                self.p[k] += self.v[k] * sub_dt

    def step(self, dt, omega_m=None, acc_m=None, is_dead=False):
        if dt <= 0.0:
            return

        # 1. Handle total sensor failure (last 5s zeros) or invalid rows
        if is_dead or omega_m is None or acc_m is None:
            self._propagate_blackout(dt)
            return

        # 2. Handle stuck/frozen sensor samples (treat as short sensor outage, NOT ZUPT!)
        if self._detect_stuck(omega_m, acc_m):
            self._propagate_blackout(dt)
            return

        # 3. Reject 1-sample impulse spikes before integration
        omega_clean, acc_clean = self._despike(omega_m, acc_m)

        # 4. Handle burst packet drops (dt > 50ms): coast across the missing gap,
        #    then integrate the newly arrived valid sample over a nominal step.
        if dt > 0.05:
            nominal_dt = 0.01
            self._propagate_blackout(dt - nominal_dt)
            dt = nominal_dt

        steps = max(1, int(math.ceil(dt / 0.005)))
        sub_dt = dt / steps

        for _ in range(steps):
            omega = [omega_clean[k] - self.bg[k] for k in range(3)]
            acc = [acc_clean[k] - self.ba[k] for k in range(3)]

            self.acc_buf.append(acc)
            self.gyro_buf.append(omega)

            acc_std = 1.0
            if len(self.acc_buf) >= 5:
                inv_n = 1.0 / len(self.acc_buf)
                ma = [sum(x[k] for x in self.acc_buf) * inv_n for k in range(3)]
                acc_std = math.sqrt(sum(sum((x[k] - ma[k]) ** 2 for x in self.acc_buf) * inv_n for k in range(3)))

            omega_norm = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
            acc_norm = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
            acc_err = abs(acc_norm - self.g_mag)

            # Midpoint quaternion attitude propagation
            mid_omega = (
                0.5 * (self.last_omega[0] + omega[0]) * sub_dt,
                0.5 * (self.last_omega[1] + omega[1]) * sub_dt,
                0.5 * (self.last_omega[2] + omega[2]) * sub_dt,
            )
            self.q = quat_normalize(quat_multiply(self.q, rotvec_to_quat(mid_omega)))

            # Gentle complementary gravity tilt correction during low-accel phases
            if acc_err < 0.25 and omega_norm < 0.30 and acc_norm > 1e-6:
                dx, dy, dz = quat_rotate(self.q, (acc[0] / acc_norm, acc[1] / acc_norm, acc[2] / acc_norm))
                tn = math.sqrt(dy * dy + dx * dx)
                if tn > 1e-6:
                    alpha = 0.0025 * math.asin(max(-1.0, min(1.0, tn))) / tn
                    self.q = quat_normalize(quat_multiply(rotvec_to_quat((dy * alpha, -dx * alpha, 0.0)), self.q))

            # World acceleration + slow EMA bias removal (counters random-walk drift)
            aw = quat_rotate(self.q, acc)
            raw_aw = (aw[0] + self.g[0], aw[1] + self.g[1], aw[2] + self.g[2])

            beta = sub_dt / (8.0 + sub_dt)
            for k in range(3):
                self.acc_world_bias[k] = (1.0 - beta) * self.acc_world_bias[k] + beta * raw_aw[k]

            acc_world = [raw_aw[k] - self.acc_world_bias[k] for k in range(3)]

            # ZUPT + Online bias adaptation (only when genuinely moving/noisy, never when stuck)
            is_stationary = (1e-5 < acc_std < 0.15) and (omega_norm < 0.04) and (acc_err < 0.25)
            if is_stationary:
                self.v = [0.0, 0.0, 0.0]
                acc_world = [0.0, 0.0, 0.0]
                exp_body = quat_rotate_inv(self.q, (0.0, 0.0, self.g_mag))
                for k in range(3):
                    self.bg[k] = 0.992 * self.bg[k] + 0.008 * omega_clean[k]
                    self.ba[k] = 0.992 * self.ba[k] + 0.008 * (acc_clean[k] - exp_body[k])
            else:
                damp = math.exp(-0.55 * sub_dt)
                for k in range(3):
                    self.v[k] *= damp

            half_dt2 = 0.5 * sub_dt * sub_dt
            for k in range(3):
                self.p[k] += self.v[k] * sub_dt + acc_world[k] * half_dt2
                self.v[k] += acc_world[k] * sub_dt

            self.last_acc_world = acc_world
            self.last_omega = omega


def is_invalid_sample(omega, acc):
    for val in omega + acc:
        if math.isnan(val) or math.isinf(val):
            return True
    return (
        abs(omega[0]) < 1e-12 and abs(omega[1]) < 1e-12 and abs(omega[2]) < 1e-12 and
        abs(acc[0]) < 1e-12 and abs(acc[1]) < 1e-12 and abs(acc[2]) < 1e-12
    )


def detect_csv_format(csv_path):
    """Auto-detects header presence, timestamp unit scale, and accel unit scale (g vs m/s^2)."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if not first_row:
            raise ValueError("Empty CSV file.")

        has_header = False
        try:
            float(first_row[0])
        except ValueError:
            has_header = True

        ts_vals = []
        acc_norms = []
        if not has_header and len(first_row) >= 7:
            ts_vals.append(float(first_row[0]))
            ax, ay, az = float(first_row[1]), float(first_row[2]), float(first_row[3])
            acc_norms.append(math.sqrt(ax * ax + ay * ay + az * az))

        for row in reader:
            if not row or len(row) < 7:
                continue
            try:
                ts_vals.append(float(row[0]))
                ax, ay, az = float(row[1]), float(row[2]), float(row[3])
                n = math.sqrt(ax * ax + ay * ay + az * az)
                if n > 1e-6:
                    acc_norms.append(n)
                if len(ts_vals) >= 100:
                    break
            except ValueError:
                continue

    # Match degrade_imu.py timestamp unit detection
    dts = sorted(ts_vals[i] - ts_vals[i - 1] for i in range(1, len(ts_vals)) if ts_vals[i] > ts_vals[i - 1])
    med_dt = dts[len(dts) // 2] if dts else 1e7
    if med_dt < 1.0:
        ts_scale = 1.0
    elif med_dt < 1e3:
        ts_scale = 1e-3
    elif med_dt < 1e6:
        ts_scale = 1e-6
    else:
        ts_scale = 1e-9

    # Match degrade_imu.py accel unit detection (g vs m/s^2)
    acc_norms.sort()
    med_acc = acc_norms[len(acc_norms) // 2] if acc_norms else 9.80665
    acc_to_ms2 = 9.80665 if (0.5 < med_acc < 2.0) else 1.0

    return has_header, ts_scale, acc_to_ms2


def run_trajectory_estimation(imu_csv_path, output_csv_path):
    if not os.path.exists(imu_csv_path):
        raise FileNotFoundError(f"Input IMU file not found: {imu_csv_path}")

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    has_header, ts_scale, acc_to_ms2 = detect_csv_format(imu_csv_path)

    # Pass 1: Read first ~2.0s (protected window) of unique valid rows for calibration
    head_samples = []
    first_ts = None
    last_calib_ts = None
    with open(imu_csv_path, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.reader(f_in)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row or len(row) < 7:
                continue
            try:
                ts = float(row[0])
                if first_ts is None:
                    first_ts = ts
                if ts == last_calib_ts:
                    continue  # Skip duplicate rows
                if (ts - first_ts) * ts_scale > 2.0 and len(head_samples) >= 50:
                    break  # Stay inside the 2.0s protected window

                # Column order in degrade_imu.py: [ts, ax, ay, az, gx, gy, gz]
                acc = (float(row[1]) * acc_to_ms2, float(row[2]) * acc_to_ms2, float(row[3]) * acc_to_ms2)
                omega = (float(row[4]), float(row[5]), float(row[6]))
                if not is_invalid_sample(omega, acc):
                    head_samples.append((omega, acc))
                    last_calib_ts = ts
                    if len(head_samples) >= 300:
                        break
            except ValueError:
                continue

    estimator = LowMemoryInertialEstimator(g_mag=9.80665)
    estimator.calibrate_from_head(head_samples)
    del head_samples

    # Pass 2: O(1) streaming from input CSV directly to output CSV
    prev_ts = None
    count = 0
    total_ms = 0.0

    with open(imu_csv_path, "r", encoding="utf-8-sig", newline="") as f_in, open(output_csv_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        if has_header:
            next(reader, None)

        writer.writerow([
            "#timestamp",
            "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
            "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
            "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]",
        ])

        for row in reader:
            if not row or len(row) < 7:
                continue

            t0 = time.perf_counter()
            raw_ts_str = row[0].strip()
            curr_ts = float(raw_ts_str)

            # Skip duplicate rows injected by degrade_imu.py
            if prev_ts is not None and curr_ts == prev_ts:
                continue

            dt = 0.0 if prev_ts is None else (curr_ts - prev_ts) * ts_scale
            prev_ts = curr_ts

            try:
                # Correct column mapping: accel=[1:4], gyro=[4:7]
                acc_m = (float(row[1]) * acc_to_ms2, float(row[2]) * acc_to_ms2, float(row[3]) * acc_to_ms2)
                omega_m = (float(row[4]), float(row[5]), float(row[6]))
                dead = is_invalid_sample(omega_m, acc_m)
            except ValueError:
                omega_m, acc_m, dead = None, None, True

            if dt > 0.0:
                estimator.step(dt, omega_m=omega_m, acc_m=acc_m, is_dead=dead)

            qw, qx, qy, qz = estimator.q
            writer.writerow([
                raw_ts_str,
                f"{estimator.p[0]:.9f}", f"{estimator.p[1]:.9f}", f"{estimator.p[2]:.9f}",
                f"{qw:.9f}", f"{qx:.9f}", f"{qy:.9f}", f"{qz:.9f}",
                f"{estimator.v[0]:.9f}", f"{estimator.v[1]:.9f}", f"{estimator.v[2]:.9f}",
            ])

            total_ms += (time.perf_counter() - t0) * 1e3
            count += 1

    print(f"Saved {count} poses to '{output_csv_path}' (avg latency: {total_ms / max(1, count):.4f} ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imu", help="Input degraded IMU CSV")
    parser.add_argument("output_est", help="Output estimated trajectory CSV")
    args = parser.parse_args()
    run_trajectory_estimation(args.input_imu, args.output_est)