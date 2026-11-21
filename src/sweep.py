#!/usr/bin/env python3
"""
Tune the knobs on YOUR degraded EuRoC data (needs numpy). Not part of the submission.

    python sweep.py DEGRADED_INPUT GT_CSV [blind_seconds_from_end]

DEGRADED_INPUT = output of the organizers' degradation script (folder/zip/data.csv)
GT_CSV         = mav0/state_groundtruth_estimate0/data.csv of the same sequence
Prints ATE (all rows) and ATE (last 5 s) for each setting; pick the best and edit the constants at the top of src/main.py.
"""
import sys, os, subprocess, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ate import ate

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'main.py')
inp, gt = sys.argv[1], sys.argv[2]
grid = {
    'KP': [0.0, 0.05, 0.2, 0.5, 1.0],
    'VEL_LEAK_TAU': [0.0, 1.0, 3.0, 10.0, 30.0],
    'KI': [0.0, 0.005, 0.02],
    'BLIND_TAU': [3.0, 8.0, 20.0],
    'SPIKE_A': [0.0, 2.0, 3.0, 6.0],
    'BLIND_AFTER': [0.3, 0.5, 1.5],
}
base = {}
for name, vals in grid.items():
    print('--- sweeping', name, '(others at default)')
    for v in vals:
        subprocess.run([sys.executable, MAIN, inp, '-o', 'sweep_tmp.txt', '--set', '%s=%s' % (name, v)],
                       stderr=subprocess.DEVNULL, check=True)
        r = ate('sweep_tmp.txt', gt, 5, quiet=True)
        print('  %s=%-6s ATE=%.3f  ATE_last5s=%.3f  rows=%d' % (name, v, r['ate'], r['ate_last'], r['n_est']))
os.remove('sweep_tmp.txt')