'''#!/usr/bin/env python3
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
os.remove('sweep_tmp.txt')'''
#!/usr/bin/env python3
"""
Tune the ESKF & drift knobs across one or more degraded EuRoC sequences (needs numpy).
Not part of the submission.

Usage:
    python sweep.py DEGRADED_1 GT_1 [DEGRADED_2 GT_2 ...]

Example:
    python sweep.py deg_V1_01.csv gt_V1_01.csv deg_MH_03.csv gt_MH_03.csv
"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ate import ate

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'main.py')
if not os.path.exists(MAIN):
    MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')

if len(sys.argv) < 3 or (len(sys.argv) - 1) % 2 != 0:
    print(__doc__)
    sys.exit(2)

pairs = [(sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv), 2)]

grid = {
    'VEL_LEAK_TAU': [0.0, 3.0, 6.0, 10.0, 18.0],
    'BLIND_TAU': [1.5, 2.5, 4.0, 6.0],
    'R_GRAV': [0.15, 0.35, 0.70, 1.20],
    'R_ZUPT': [0.01, 0.03, 0.08, 0.20],
    'SIGMA_BA': [0.005, 0.010, 0.025],
    'GATE_GRAV': [0.50, 0.85, 1.30],
}

best_params = {}


def eval_config(params):
    tot_ate = 0.0
    tot_last = 0.0
    set_args = []
    for k, v in params.items():
        set_args.extend(['--set', f'{k}={v}'])
    for inp, gt in pairs:
        subprocess.run(
            [sys.executable, MAIN, inp, '-o', 'sweep_tmp.txt'] + set_args,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        r = ate('sweep_tmp.txt', gt, 5.0, quiet=True)
        tot_ate += r['ate']
        tot_last += r.get('ate_last', 0.0)
    return tot_ate / len(pairs), tot_last / len(pairs)


base_ate, base_last = eval_config(best_params)
print(f'Baseline ({len(pairs)} seqs): mean_ATE={base_ate:.3f}m  mean_ATE_last5s={base_last:.3f}m')

for name, vals in grid.items():
    print(f'\n--- sweeping {name} ---')
    best_v = None
    best_score = (1e9, 1e9)
    for v in vals:
        trial = dict(best_params)
        trial[name] = v
        mean_ate, mean_last = eval_config(trial)
        print(f'  {name}={v:<6} mean_ATE={mean_ate:.3f}m  mean_ATE_last5s={mean_last:.3f}m')
        # Primary objective: overall ATE; secondary: last-5s blackout ATE
        if (mean_ate, mean_last) < best_score:
            best_score = (mean_ate, mean_last)
            best_v = v
    best_params[name] = best_v
    print(f'  => Selected {name}={best_v} (mean_ATE={best_score[0]:.3f}m)')

if os.path.exists('sweep_tmp.txt'):
    os.remove('sweep_tmp.txt')

print('\n=== Best Tunables (copy to top of main.py) ===')
for k, v in best_params.items():
    print(f'{k} = {v}')