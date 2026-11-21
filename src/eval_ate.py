#!/usr/bin/env python3
"""
Local evaluation + visualisation (NOT part of the submission). Needs numpy; plotting needs matplotlib.

    python eval_ate.py estimated_trajectory.txt ground_truth.csv [last_seconds] [options]

Options:
    --plot            draw ground truth vs estimate (4 panels) and save it as trajectory_plot.png
    --save FILE       plot file name (default trajectory_plot.png)
    --no-show         save the figure but do not open a window
    --zoom            zoom the XY / 3D views onto the ground-truth region (use when the estimate drifts far away)
    --no-align        plot / score without the rigid SE(3) alignment

Prints ATE RMSE (translation, after Umeyama SE(3) alignment, like `evo_ape --align`).
If last_seconds is given (e.g. 5) it also prints the error over the last N seconds.
Ground truth can be EuRoC csv (ts, px, py, pz, qw, ...) or TUM txt (ts tx ty tz ...); the time unit is auto-detected.
"""
import sys
import numpy as np


def load_est(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)


def load_gt(p):
    """Returns (raw timestamps, positions). Works for EuRoC csv and TUM txt."""
    rows = []
    with open(p) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            r = line.strip().split(',') if ',' in line else line.split()
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                continue                     # header line
    a = np.array(rows)
    return a[:, 0], a[:, 1:4]


def pick_scale(raw_ts, est_ts):
    """Choose the unit (ns/us/ms/s) that makes the GT time range overlap the estimate's best."""
    best, best_sc = -1.0, 1.0
    for sc in (1e-9, 1e-6, 1e-3, 1.0):
        lo, hi = max(raw_ts[0] * sc, est_ts[0]), min(raw_ts[-1] * sc, est_ts[-1])
        union = max(raw_ts[-1] * sc, est_ts[-1]) - min(raw_ts[0] * sc, est_ts[0])
        j = (hi - lo) / union if union > 0 else 0.0
        if j > best:
            best, best_sc = j, sc
    return best_sc


def umeyama(X, Y):
    """Rigid alignment (rotation + translation, no scale) of X onto Y."""
    mx, my = X.mean(0), Y.mean(0)
    S = (Y - my).T @ (X - mx) / len(X)
    U, _D, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    t = my - R @ mx
    return R, t


def associate(est_path, gt_path):
    est = load_est(est_path)
    raw, gtp = load_gt(gt_path)
    gts = raw * pick_scale(raw, est[:, 0])
    idx = np.searchsorted(gts, est[:, 0])
    idx = np.clip(idx, 1, len(gts) - 1)
    idx -= (np.abs(gts[idx - 1] - est[:, 0]) < np.abs(gts[idx] - est[:, 0]))
    ok = np.abs(gts[idx] - est[:, 0]) < 0.01
    return est, gts, gtp, idx, ok


def ate(est_path, gt_path, last_s=None, quiet=False, align=True):
    est, gts, gtp, idx, ok = associate(est_path, gt_path)
    X, Y, T = est[ok, 1:4], gtp[idx[ok]], est[ok, 0]
    if align:
        R, t = umeyama(X, Y)
        Xa = X @ R.T + t
    else:
        Xa = X
    err = np.linalg.norm(Xa - Y, axis=1)
    res = {'n_est': len(est), 'n_matched': int(ok.sum()), 'ate': float(np.sqrt((err ** 2).mean()))}
    if last_s:
        m = T > T[-1] - last_s
        res['ate_last'] = float(np.sqrt((err[m] ** 2).mean()))
    if not quiet:
        print(res)
    return res


def plot(est_path, gt_path, last_s=5.0, save='trajectory_plot.png', show=True, zoom=False, align=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    est, gts, gtp, idx, ok = associate(est_path, gt_path)
    X, Y, T = est[ok, 1:4], gtp[idx[ok]], est[ok, 0]
    if align:
        R, t = umeyama(X, Y)
        Xa = X @ R.T + t
    else:
        Xa = X
    err = np.linalg.norm(Xa - Y, axis=1)
    rmse = np.sqrt((err ** 2).mean())
    t0 = T[0]
    tr = T - t0
    blind = tr > tr[-1] - last_s

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Estimated vs ground-truth trajectory   (ATE RMSE = %.3f m%s)'
                 % (rmse, ', SE(3)-aligned' if align else ', not aligned'), fontsize=13)

    # span of ground truth, used for zooming
    lo, hi = Y.min(0), Y.max(0)
    pad = 0.25 * max((hi - lo).max(), 1.0)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(Y[:, 0], Y[:, 1], 'g-', lw=2, label='ground truth')
    ax1.plot(Xa[:, 0], Xa[:, 1], 'r-', lw=1, label='estimate')
    ax1.plot(Xa[blind, 0], Xa[blind, 1], 'm-', lw=2.5, label='last %.0f s (blind)' % last_s)
    ax1.plot(Y[0, 0], Y[0, 1], 'ko', label='start')
    ax1.set_title('Top view (X-Y)')
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    if zoom:
        ax1.set_xlim(lo[0] - pad, hi[0] + pad); ax1.set_ylim(lo[1] - pad, hi[1] + pad)
        ax1.set_aspect('equal', adjustable='box')
    else:
        ax1.set_aspect('equal', adjustable='datalim')

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'g-', lw=2, label='ground truth')
    ax2.plot(Xa[:, 0], Xa[:, 1], Xa[:, 2], 'r-', lw=1, label='estimate')
    ax2.set_title('3D view'); ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
    ax2.legend(fontsize=8)
    if zoom:
        ax2.set_xlim(lo[0] - pad, hi[0] + pad); ax2.set_ylim(lo[1] - pad, hi[1] + pad)
        ax2.set_zlim(lo[2] - pad, hi[2] + pad)

    ax3 = fig.add_subplot(2, 2, 3)
    for k, c in zip(range(3), ('tab:red', 'tab:green', 'tab:blue')):
        ax3.plot(tr, Y[:, k], '-', color=c, lw=2, alpha=0.5)
        ax3.plot(tr, Xa[:, k], '--', color=c, lw=1)
    ax3.set_title('x / y / z vs time  (solid = ground truth, dashed = estimate)')
    ax3.set_xlabel('time [s]'); ax3.set_ylabel('position [m]'); ax3.grid(alpha=0.3)
    ax3.axvspan(tr[-1] - last_s, tr[-1], color='m', alpha=0.1)
    if zoom:
        ax3.set_ylim(lo.min() - pad, hi.max() + pad)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(tr, err, 'k-')
    ax4.axvspan(tr[-1] - last_s, tr[-1], color='m', alpha=0.1, label='last %.0f s' % last_s)
    ax4.set_title('Position error vs time')
    ax4.set_xlabel('time [s]'); ax4.set_ylabel('error [m]'); ax4.grid(alpha=0.3); ax4.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save, dpi=130)
    print('plot saved to', save)
    if show:
        plt.show()


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    flags = {a for a in args if a.startswith('--')}
    save = 'trajectory_plot.png'
    if '--save' in args:
        i = args.index('--save')
        save = args[i + 1]
        del args[i:i + 2]
    pos = [a for a in args if not a.startswith('--')]
    if len(pos) < 2:
        print(__doc__)
        sys.exit(2)
    last = float(pos[2]) if len(pos) > 2 else None
    align = '--no-align' not in flags
    ate(pos[0], pos[1], last, align=align)
    if '--plot' in flags or '--save' in sys.argv:
        plot(pos[0], pos[1], last or 5.0, save=save, show='--no-show' not in flags,
             zoom='--zoom' in flags, align=align)