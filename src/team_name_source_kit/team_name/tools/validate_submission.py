#!/usr/bin/env python3
"""
Validate estimated_trajectory.txt against the input IMU file and the required format.
Standard library only.

usage: python validate_submission.py <input imu csv | imu0/data.csv> estimated_trajectory.txt
Checks: row count == IMU rows, 8 columns, finite numbers, timestamps equal input (ns->s) & non-decreasing,
        quaternions normalised.
"""
import sys
import math


def data_lines(path):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            s = ln.strip()
            if not s or s[0] == "#" or s[0].isalpha():
                continue
            out.append(s)
    return out


def main():
    imu = data_lines(sys.argv[1])
    est = data_lines(sys.argv[2])
    problems = []
    if len(imu) != len(est):
        problems.append("row count mismatch: input %d vs output %d" % (len(imu), len(est)))
    prev = None
    max_qdev = 0.0
    n_ts_bad = 0
    for i, ln in enumerate(est):
        c = ln.split()
        if len(c) != 8:
            problems.append("line %d has %d columns (need 8)" % (i, len(c)))
            if len(problems) > 20:
                break
            continue
        try:
            v = [float(x) for x in c]
        except ValueError:
            problems.append("line %d non-numeric" % i)
            continue
        if any(math.isnan(x) or math.isinf(x) for x in v):
            problems.append("line %d has nan/inf" % i)
        qn = math.sqrt(sum(x * x for x in v[4:8]))
        max_qdev = max(max_qdev, abs(qn - 1.0))
        if prev is not None and v[0] < prev:
            problems.append("timestamps decrease at line %d" % i)
        prev = v[0]
        if i < len(imu):
            raw = imu[i].split(",")[0].strip()
            try:
                ns = int(raw)
                s, r = divmod(ns, 10**9)
                if c[0] != "%d.%09d" % (s, r):
                    n_ts_bad += 1
            except ValueError:
                pass
    if n_ts_bad:
        problems.append("%d timestamps differ from input timestamp*1e-9" % n_ts_bad)
    if max_qdev > 1e-6:
        problems.append("quaternion norm deviates up to %.2e" % max_qdev)
    print("input rows: %d | output rows: %d | max |q|-1: %.2e" % (len(imu), len(est), max_qdev))
    if problems:
        print("VALIDATION FAILED:")
        for p in problems[:25]:
            print("  -", p)
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
