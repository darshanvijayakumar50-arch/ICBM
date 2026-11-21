#!/usr/bin/env python3
"""
Generate a SYNTHETIC 200 Hz IMU recording + ground truth (TUM format) for pipeline testing.
This is NOT real EuRoC/TUM-VI data. Output CSV matches the organizer degrade script input:
    timestamp[ns], ax, ay, az, gx, gy, gz   (no header)

usage: python make_synthetic.py OUT_DIR [--duration 90] [--seed 1] [--slow]
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

G = 9.80665


def main():
    out = sys.argv[1]
    dur = 90.0
    seed = 1
    if "--duration" in sys.argv:
        dur = float(sys.argv[sys.argv.index("--duration") + 1])
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    hz, sub = 200, 10
    n = int(dur * hz) + 1
    tf = np.arange(n * sub) / (hz * sub)          # dense grid (2000 Hz)
    t0_still = 4.0
    x = np.clip((tf - t0_still) / 4.0, 0, 1)
    ramp = x * x * (3 - 2 * x)
    tt = np.maximum(tf - t0_still, 0)
    k = 1.0 if "--slow" not in sys.argv else 0.0   # default: includes faster, EuRoC-like manoeuvres
    p = np.stack([1.2 * np.sin(0.25 * tt) + k * 0.8 * np.sin(0.9 * tt),
                  1.0 * np.sin(0.2 * tt + 0.5) + k * 0.7 * np.sin(0.7 * tt + 0.5),
                  0.3 * np.sin(0.4 * tt) + k * 0.15 * np.sin(1.1 * tt)], 1)
    p = p * ramp[:, None]
    yaw = 0.8 * np.sin(0.12 * tt) * ramp
    pitch = 0.15 * np.sin(0.5 * tt + 1.0) * ramp
    roll = 0.15 * np.sin(0.7 * tt) * ramp
    R = Rot.from_euler("ZYX", np.stack([yaw, pitch, roll], 1))
    dtf = 1.0 / (hz * sub)
    acc_w = np.gradient(np.gradient(p, dtf, axis=0), dtf, axis=0)
    # decimate to 200 Hz
    idx = np.arange(0, n * sub, sub)
    Rs = R[idx]
    a_body = Rs.inv().apply(acc_w[idx] + np.array([0, 0, G]))
    Rn = R[idx[1:]]
    Rp = R[idx[:-1]]
    w = (Rp.inv() * Rn).as_rotvec() * hz
    w = np.vstack([w, w[-1]])
    a_body = a_body + rng.standard_normal((n, 3)) * 0.02 + np.array([0.03, -0.02, 0.05])
    w = w + rng.standard_normal((n, 3)) * 0.002 + np.array([0.002, -0.001, 0.0015])
    ts = 1403636579763555584 + (np.arange(n) * 5_000_000).astype(np.int64)
    with open(os.path.join(out, "imu_clean.csv"), "w") as f:
        for i in range(n):
            f.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (ts[i], *a_body[i], *w[i]))
    q = Rs.as_quat()
    with open(os.path.join(out, "groundtruth.txt"), "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(n):
            s, r = divmod(int(ts[i]), 10**9)
            f.write("%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (s, r, *p[idx[i]], *q[i]))
    print("wrote", os.path.join(out, "imu_clean.csv"), "and groundtruth.txt", n, "rows")


if __name__ == "__main__":
    main()
