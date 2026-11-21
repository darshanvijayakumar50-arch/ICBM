#!/usr/bin/env python3
"""
Path Finder (Track 03) - IMU-only 6-DoF trajectory estimation under sensor degradation.

Pure Python, streaming, single thread, offline. No numpy on purpose (keeps RAM tiny, startup fast).

Usage:
    python main.py INPUT [-o estimated_trajectory.txt] [--blind-seconds N]
                         [--meta metadata.json] [--set NAME=VALUE ...]

INPUT = folder containing imu0/data.csv, a .zip of that folder, or the data.csv file itself.

Output: TUM format, one pose per IMU row:  timestamp[s] tx ty tz qx qy qz qw
"""
import sys
import os
import io
import time
from math import sqrt, sin, cos, exp

# ------------------------------------------------------------------ tunables
INIT_SECONDS = 1.0      # initial "at rest" window used for gyro bias + gravity direction
INIT_MAX_ROWS = 600     # safety cap on rows held in the init window
GAP_FACTOR = 3.0        # row-to-row time gap > GAP_FACTOR * nominal dt  => dropout
KP = 0.2                # tilt (roll/pitch) correction gain from accelerometer
KI = 0.005              # gyro-bias correction gain
KP_RECOVER = 1.0        # stronger tilt correction right after a dropout
RECOVER_SECONDS = 1.0
GATE = 1.0              # m/s^2: accel correction fades out as | |a|-g | approaches this
BLIND_AFTER = 0.5       # s of continuous invalid data => IMU considered dead
SPIKE_A = 3.0           # m/s^2: per-sample accel jump treated as a spike (0 = off)
SPIKE_G = 0.4           # rad/s: per-sample gyro jump treated as a spike
BLIND_TAU = 8.0         # s: velocity decay time constant while IMU is dead
VEL_LEAK_TAU = 0.0      # s: leak on velocity while IMU is alive (0 = off)
NOMINAL_G = 9.80665

ZERO6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TS_MULT = 1.0           # multiplier that converts the file's timestamps to ns (set automatically)


def parse(line):
    """Return (timestamp_ns or None, 6-tuple of floats or None if row is unusable)."""
    p = line.split(',')
    try:
        if TS_MULT == 1.0:
            ts = int(p[0])
        else:
            ts = int(float(p[0]) * TS_MULT + 0.5)
    except (ValueError, IndexError):
        try:
            ts = int(float(p[0]) * TS_MULT)
        except (ValueError, IndexError):
            ts = None
    try:
        v = (float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]))
    except (ValueError, IndexError):
        return ts, None
    s = v[0] + v[1] + v[2] + v[3] + v[4] + v[5]
    if s != s or s > 1e7 or s < -1e7:      # NaN / inf / absurd values
        return ts, None
    if v == ZERO6:                          # all-zero row = dropped frame
        return ts, None
    return ts, v


def fmt_ts(ts):
    """Exact ns -> seconds string (no float rounding)."""
    s, ns = divmod(ts, 1000000000)
    return '%d.%09d' % (s, ns)


def coast(px, py, pz, vx, vy, vz, dt, dead):
    """No usable IMU data for dt seconds: constant velocity (decaying if IMU is dead)."""
    if dead:
        k = exp(-dt / BLIND_TAU)
        g = BLIND_TAU * (1.0 - k)
        return px + vx * g, py + vy * g, pz + vz * g, vx * k, vy * k, vz * k
    return px + vx * dt, py + vy * dt, pz + vz * dt, vx, vy, vz


def open_input(path):
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            if os.path.basename(root) == 'imu0' and 'data.csv' in files:
                return open(os.path.join(root, 'data.csv'), 'r', encoding='utf-8', errors='replace'), None
        sys.exit('ERROR: imu0/data.csv not found inside folder ' + path)
    if path.lower().endswith('.zip'):
        import zipfile
        zf = zipfile.ZipFile(path)
        for n in zf.namelist():
            if n.replace('\\', '/').endswith('imu0/data.csv'):
                return io.TextIOWrapper(zf.open(n), encoding='utf-8', errors='replace'), zf
        sys.exit('ERROR: imu0/data.csv not found inside zip ' + path)
    return open(path, 'r', encoding='utf-8', errors='replace'), None


def detect_ts_mult(path):
    """Guess the timestamp unit from the median spacing of the first rows (like the organizers' script)."""
    inp, zf = open_input(path)
    vals = []
    for line in inp:
        try:
            vals.append(float(line.split(',')[0]))
        except ValueError:
            continue
        if len(vals) >= 60:
            break
    inp.close()
    if zf is not None:
        zf.close()
    d = sorted(vals[i + 1] - vals[i] for i in range(len(vals) - 1) if vals[i + 1] > vals[i])
    if not d:
        return 1.0
    med = d[len(d) // 2]
    if med < 1.0:
        return 1e9          # seconds
    if med < 1e3:
        return 1e6          # milliseconds
    if med < 1e6:
        return 1e3          # microseconds
    return 1.0              # nanoseconds


def process(inp, out, blind_extra_s):
    _sqrt, _sin, _cos = sqrt, sin, cos
    write = out.write
    lines = iter(inp)
    out.write('# timestamp tx ty tz qx qy qz qw\n')

    # ---------------- 1) initial window: gyro bias + gravity direction -------------
    init_ts = []
    first_ts = last_ts = None
    n = 0
    sgx = sgy = sgz = sax = say = saz = 0.0
    prev_raw = None
    init_ns = int(INIT_SECONDS * 1e9)
    for line in lines:
        if line[0] in '#\r\n':
            continue
        ts, v = parse(line)
        if ts is None:
            if last_ts is None:
                continue
            ts = last_ts + 5000000
        init_ts.append(ts)
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        if v is not None and v != prev_raw:
            sgx += v[0]; sgy += v[1]; sgz += v[2]
            sax += v[3]; say += v[4]; saz += v[5]
            n += 1
            prev_raw = v
        if ts - first_ts >= init_ns or len(init_ts) >= INIT_MAX_ROWS:
            break
    if not init_ts:
        sys.exit('ERROR: no IMU rows found')

    if len(init_ts) > 1:
        diffs = sorted(init_ts[i + 1] - init_ts[i] for i in range(len(init_ts) - 1))
        nominal_ns = diffs[len(diffs) // 2]
        if nominal_ns <= 0:
            nominal_ns = 5000000
    else:
        nominal_ns = 5000000
    nominal_s = nominal_ns * 1e-9
    gap_ns = GAP_FACTOR * nominal_ns

    swap = False                       # True if file has accel columns first (cols 1-3)
    if n > 0:
        m3 = _sqrt((sgx / n) ** 2 + (sgy / n) ** 2 + (sgz / n) ** 2)
        m6 = _sqrt((sax / n) ** 2 + (say / n) ** 2 + (saz / n) ** 2)
        if m3 > 5.0 and m6 < 5.0:
            swap = True
            sgx, sgy, sgz, sax, say, saz = sax, say, saz, sgx, sgy, sgz

    if n > 0:
        bx, by, bz = sgx / n, sgy / n, sgz / n
        fx, fy, fz = sax / n, say / n, saz / n
        G = _sqrt(fx * fx + fy * fy + fz * fz)
        if G < 8.0 or G > 11.5:
            G = NOMINAL_G
        fn = _sqrt(fx * fx + fy * fy + fz * fz)
        ux, uy, uz = fx / fn, fy / fn, fz / fn
    else:
        bx = by = bz = 0.0
        G = NOMINAL_G
        fx, fy, fz = 0.0, 0.0, G
        ux, uy, uz = 0.0, 0.0, 1.0
    pgx, pgy, pgz = bx, by, bz          # previous accepted sample (for spike rejection)
    pax, pay, paz = fx, fy, fz
    rej = 0

    # initial orientation: rotate measured "up" (body frame) onto world +Z, yaw = 0
    if uz > -0.999999:
        qw = _sqrt((1.0 + uz) * 0.5)
        qx = uy / (2.0 * qw)
        qy = -ux / (2.0 * qw)
        qz = 0.0
    else:
        qw, qx, qy, qz = 0.0, 1.0, 0.0, 0.0
    nn = 1.0 / _sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw *= nn; qx *= nn; qy *= nn; qz *= nn

    # rows of the init window: vehicle assumed at rest at the origin
    pose0 = ' 0.000000 0.000000 0.000000 %.8f %.8f %.8f %.8f\n' % (qx, qy, qz, qw)
    for t in init_ts:
        write(fmt_ts(t) + pose0)
    rows = len(init_ts)

    # ---------------- 2) main streaming loop ---------------------------------------
    px = py = pz = vx = vy = vz = 0.0
    prev_ts = last_ts
    invalid_t = 0.0
    recover_t = 0.0
    gate_inv = 1.0 / GATE
    leak_on = VEL_LEAK_TAU > 0.0
    leak_inv = 1.0 / VEL_LEAK_TAU if leak_on else 0.0
    kp_norm, kp_rec, ki = KP, KP_RECOVER, KI
    fmt = '%s %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n'

    for line in lines:
        if line[0] in '#\r\n':
            continue
        ts, v = parse(line)
        if ts is None:
            ts = prev_ts + nominal_ns
        dts = ts - prev_ts
        prev_ts = ts
        if v is not None and v == prev_raw:
            v = None                       # exact repeat of previous row = stale/frozen frame
        if dts <= 0:
            dt = 0.0
            if v is not None:
                prev_raw = v
            v = None                       # non-monotonic time: just repeat pose, no integration
            write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
            rows += 1
            continue
        dt = dts * 1e-9

        if v is None:
            # ---- dropped / invalid row: coast ----
            invalid_t += dt
            px, py, pz, vx, vy, vz = coast(px, py, pz, vx, vy, vz, dt, invalid_t > BLIND_AFTER)
            if invalid_t > 0.1:
                recover_t = RECOVER_SECONDS
            write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
            rows += 1
            continue

        prev_raw = v
        if dts > gap_ns:
            # ---- time gap in the file (rows missing): coast over the hole, then use this row ----
            hole = dt - nominal_s
            invalid_t += hole
            px, py, pz, vx, vy, vz = coast(px, py, pz, vx, vy, vz, hole, invalid_t > BLIND_AFTER)
            dt = nominal_s
            if hole > 0.1:
                recover_t = RECOVER_SECONDS
        invalid_t = 0.0

        if swap:
            ax, ay, az, gx, gy, gz = v
        else:
            gx, gy, gz, ax, ay, az = v
        if SPIKE_A > 0.0:
            if (rej < 3 and (abs(ax - pax) > SPIKE_A or abs(ay - pay) > SPIKE_A or abs(az - paz) > SPIKE_A
                             or abs(gx - pgx) > SPIKE_G or abs(gy - pgy) > SPIKE_G or abs(gz - pgz) > SPIKE_G)):
                rej += 1                   # spike: hold the previous accepted sample
                ax, ay, az, gx, gy, gz = pax, pay, paz, pgx, pgy, pgz
            else:
                rej = 0
                pax, pay, paz, pgx, pgy, pgz = ax, ay, az, gx, gy, gz
        wx = gx - bx
        wy = gy - by
        wz = gz - bz

        # ---- accelerometer tilt correction (Mahony), faded out when |a| != g ----
        fn = _sqrt(ax * ax + ay * ay + az * az)
        dev = fn - G
        if dev < 0.0:
            dev = -dev
        if dev < GATE and fn > 1e-6:
            k = 1.0 - dev * gate_inv
            inv = 1.0 / fn
            ux = ax * inv; uy = ay * inv; uz = az * inv
            hx = 2.0 * (qx * qz - qw * qy)
            hy = 2.0 * (qw * qx + qy * qz)
            hz = qw * qw - qx * qx - qy * qy + qz * qz
            ex = uy * hz - uz * hy
            ey = uz * hx - ux * hz
            ez = ux * hy - uy * hx
            if recover_t > 0.0:
                kk = kp_rec * k
                recover_t -= dt
            else:
                kk = kp_norm * k
            wx += kk * ex; wy += kk * ey; wz += kk * ez
            kb = ki * k * dt
            bx -= kb * ex; by -= kb * ey; bz -= kb * ez

        # ---- quaternion integration: q = q (x) exp(w dt) ----
        th2 = wx * wx + wy * wy + wz * wz
        if th2 > 1e-14:
            mag = _sqrt(th2)
            hw = 0.5 * mag * dt
            s = _sin(hw) / mag
            c = _cos(hw)
        else:
            s = 0.5 * dt
            c = 1.0
        dx = wx * s; dy = wy * s; dz = wz * s
        nw = qw * c - qx * dx - qy * dy - qz * dz
        nx = qw * dx + qx * c + qy * dz - qz * dy
        ny = qw * dy - qx * dz + qy * c + qz * dx
        nz = qw * dz + qx * dy - qy * dx + qz * c
        nn = 1.0 / _sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        qw = nw * nn; qx = nx * nn; qy = ny * nn; qz = nz * nn

        # ---- rotate specific force to world, remove gravity ----
        xx = qx * qx; yy = qy * qy; zz = qz * qz
        xy = qx * qy; xz = qx * qz; yz = qy * qz
        wxq = qw * qx; wyq = qw * qy; wzq = qw * qz
        awx = (1.0 - 2.0 * (yy + zz)) * ax + 2.0 * (xy - wzq) * ay + 2.0 * (xz + wyq) * az
        awy = 2.0 * (xy + wzq) * ax + (1.0 - 2.0 * (xx + zz)) * ay + 2.0 * (yz - wxq) * az
        awz = 2.0 * (xz - wyq) * ax + 2.0 * (yz + wxq) * ay + (1.0 - 2.0 * (xx + yy)) * az - G

        # ---- integrate ----
        hdt2 = 0.5 * dt * dt
        px += vx * dt + awx * hdt2
        py += vy * dt + awy * hdt2
        pz += vz * dt + awz * hdt2
        vx += awx * dt
        vy += awy * dt
        vz += awz * dt
        if leak_on:
            f = 1.0 - dt * leak_inv
            vx *= f; vy *= f; vz *= f

        write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
        rows += 1

    # ---------------- 3) optional blind extension after the file ends -----------------
    if blind_extra_s > 0.0:
        cnt = int(blind_extra_s / nominal_s + 0.5)
        if invalid_t < BLIND_AFTER:
            invalid_t = BLIND_AFTER
        for _ in range(cnt):
            prev_ts += nominal_ns
            invalid_t += nominal_s
            px, py, pz, vx, vy, vz = coast(px, py, pz, vx, vy, vz, nominal_s, True)
            write(fmt % (fmt_ts(prev_ts), px, py, pz, qx, qy, qz, qw))
            rows += 1

    duration_s = (prev_ts - first_ts) * 1e-9
    return rows, duration_s, nominal_s


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    inp_path = argv[1]
    out_path = 'estimated_trajectory.txt'
    blind = 0.0
    meta = None
    ts_unit = 'auto'
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == '-o':
            out_path = argv[i + 1]; i += 2
        elif a == '--blind-seconds':
            blind = float(argv[i + 1]); i += 2
        elif a == '--meta':
            meta = argv[i + 1]; i += 2
        elif a == '--ts-unit':
            ts_unit = argv[i + 1]; i += 2
        elif a == '--set':
            name, val = argv[i + 1].split('=')
            if name not in globals() or name in ('ZERO6',):
                sys.exit('unknown tunable ' + name)
            globals()[name] = float(val)
            i += 2
        else:
            print('unknown argument', a)
            return 2

    t0 = time.perf_counter()
    global TS_MULT
    if ts_unit == 'auto':
        TS_MULT = detect_ts_mult(inp_path)
    else:
        TS_MULT = {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1.0}[ts_unit]
    inp, zf = open_input(inp_path)
    with open(out_path, 'w', buffering=1 << 16) as out:
        rows, dur, nominal_s = process(inp, out, blind)
    inp.close()
    if zf is not None:
        zf.close()
    elapsed = time.perf_counter() - t0

    peak_kb = None
    try:
        import resource
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            peak_kb //= 1024
    except Exception:
        pass
    ms_per_s = 1000.0 * elapsed / dur if dur > 0 else 0.0
    sys.stderr.write('rows=%d  imu_seconds=%.1f  runtime=%.2fs  (%.2f ms per 1 s of IMU)  peak_rss_kb=%s\n'
                     % (rows, dur, elapsed, ms_per_s, peak_kb))
    if meta:
        with open(meta, 'w') as f:
            f.write('{\n  "rows": %d,\n  "imu_seconds": %.3f,\n  "nominal_dt_s": %.6f,\n'
                    '  "runtime_s": %.3f,\n  "ms_per_imu_second": %.3f,\n  "peak_rss_kb": %s,\n'
                    '  "language": "python (pure, no numpy)"\n}\n'
                    % (rows, dur, nominal_s, elapsed, ms_per_s, 'null' if peak_kb is None else peak_kb))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))