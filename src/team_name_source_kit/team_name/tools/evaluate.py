#!/usr/bin/env python3
"""
Local scorer approximating evo (ATE RMSE with SE(3) Umeyama alignment, RPE over 1 s).
Needs numpy. This is NOT the organizers' evaluator - use real `evo_ape tum gt est -a` if you have evo.

usage: python evaluate.py groundtruth.txt estimated_trajectory.txt [--fail-seconds 5]
"""
import sys
import numpy as np


def load(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            rows.append([float(x) for x in ln.split()])
    return np.array(rows)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(S)
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    Rm = U @ Sg @ Vt
    t = mu_d - Rm @ mu_s
    return Rm, t


def rmse(x):
    return float(np.sqrt(np.mean(np.sum(x * x, 1)))) if len(x) else float("nan")


def quat_angle(q1, q2):
    d = np.abs(np.sum(q1 * q2, 1)).clip(0, 1)
    return 2 * np.degrees(np.arccos(d))


def main():
    gt = load(sys.argv[1])
    est = load(sys.argv[2])
    fail_s = 5.0
    if "--fail-seconds" in sys.argv:
        fail_s = float(sys.argv[sys.argv.index("--fail-seconds") + 1])
    gmap = {round(r[0], 6): r for r in gt}
    pairs = [(gmap[round(e[0], 6)], e) for e in est if round(e[0], 6) in gmap]
    G = np.array([a for a, _ in pairs])
    E = np.array([b for _, b in pairs])
    print("matched poses: %d of %d estimated rows (GT rows: %d)" % (len(pairs), len(est), len(gt)))
    T_end = E[-1, 0]
    pre = E[:, 0] < T_end - fail_s
    blind = ~pre

    def ate(mask_fit, mask_eval):
        R, t = umeyama(E[mask_fit, 1:4], G[mask_fit, 1:4])
        d = (E[mask_eval, 1:4] @ R.T + t) - G[mask_eval, 1:4]
        return rmse(d), R, t

    a_all, _, _ = ate(np.ones(len(E), bool), np.ones(len(E), bool))
    a_pre, R, t = ate(pre, pre)
    a_blind, _, _ = ate(pre, blind)
    print("ATE RMSE all rows (align all)        : %.3f m" % a_all)
    print("ATE RMSE pre-failure (align pre)     : %.3f m" % a_pre)
    print("ATE RMSE blind last %.0fs (align pre)  : %.3f m" % (fail_s, a_blind))
    # blind-window drift at the end
    Ea = E[:, 1:4] @ R.T + t
    print("final-pose position error            : %.3f m" % np.linalg.norm(Ea[-1] - G[-1, 1:4]))
    # orientation error (yaw is arbitrary w/o alignment; report after removing constant yaw+: use raw angle w/ best fixed rotation is overkill)
    # RPE over ~1 s
    ts = E[:, 0]
    j = np.searchsorted(ts, ts + 1.0)
    ok = j < len(ts)
    i0 = np.where(ok)[0]
    j0 = j[ok]
    de = E[j0, 1:4] - E[i0, 1:4]
    dg = G[j0, 1:4] - G[i0, 1:4]
    # rotate GT delta into estimate frame via alignment: use R^T
    de_a = de @ R.T
    print("RPE translation RMSE (1 s, aligned)  : %.3f m" % rmse(de_a - dg))
    qn = np.linalg.norm(E[:, 4:8], axis=1)
    print("quaternion norm range                : %.9f .. %.9f" % (qn.min(), qn.max()))


if __name__ == "__main__":
    main()
