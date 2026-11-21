'''#!/usr/bin/env python3
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
    sys.exit(main(sys.argv))'''

#!/usr/bin/env python3
"""
Path Finder (Track 03) - IMU-only 6-DoF trajectory estimation under sensor degradation.

Pure Python, streaming, single thread, offline. No numpy on purpose (keeps RAM tiny, startup fast).
Uses a lightweight 15-state Error-State Kalman Filter (ESKF) with ZUPT, gravity-vector updates,
3-tap median spike rejection, and exponential velocity decay during dropouts/blackouts.

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
INIT_SECONDS = 1.8       # max stationary init window (matches degrade_imu protect_seconds=2.0)
INIT_MAX_ROWS = 1000     # safety cap on rows held in init window
INIT_MOTION_W = 0.15     # rad/s: break init window early if motion starts
INIT_MOTION_A = 0.50     # m/s^2: break init window early if |a| deviates from nominal g
GAP_FACTOR = 1.5         # row-to-row time gap > GAP_FACTOR * nominal_dt => dropped frame(s)

# Process noise densities (continuous-time)
SIGMA_ACC = 0.18         # m/s^2 / sqrt(Hz): accelerometer white noise
SIGMA_GYRO = 0.015       # rad/s / sqrt(Hz): gyro white noise
SIGMA_BA = 0.010         # m/s^3 / sqrt(Hz): accel bias random walk
SIGMA_BG = 0.0006        # rad/s^2 / sqrt(Hz): gyro bias random walk

# Measurement noise & gating
R_GRAV = 0.35            # m/s^2: gravity direction measurement noise (when near 1g)
GATE_GRAV = 0.85         # m/s^2: gate | |a| - G | for gravity tilt update
R_ZUPT = 0.03            # m/s: pseudo-zero-velocity measurement noise during hover/rest
ZUPT_W_THRESH = 0.06     # rad/s: gyro norm threshold for quasi-static detection
ZUPT_A_THRESH = 0.25     # m/s^2: | |a| - G | threshold for quasi-static detection

# Dropout / Blackout / Damping
SPIKE_A = 3.5            # m/s^2: per-sample accel jump guard
SPIKE_G = 0.45           # rad/s: per-sample gyro jump guard
BLIND_AFTER = 0.25       # s of continuous invalid data => IMU considered dead (blackout)
BLIND_TAU = 2.5          # s: velocity decay time constant during blackout
VEL_LEAK_TAU = 6.0       # s: gentle velocity damping to bound unobservable drift (0 = off)
NOMINAL_G = 9.80665

ZERO6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
TS_MULT = 1.0            # multiplier converting file timestamps to ns (auto-detected)


def median3(a, b, c):
    """Fast median of 3 floats."""
    if a > b:
        if b > c:
            return b
        return c if a > c else a
    if a > c:
        return a
    return c if b > c else b


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
    if v == ZERO6:                          # all-zero row = blackout / failure frame
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
    """Guess the timestamp unit from the median spacing of the first rows."""
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


def mat_mul_15(A, B):
    """15x15 matrix multiply (list of 15 lists)."""
    BT = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in BT] for row in A]


def mat_mul_15_T(A, B):
    """15x15 A @ B^T."""
    return [[sum(a * b for a, b in zip(rowA, rowB)) for rowB in B] for rowA in A]


def inv_3x3(M):
    """Invert a 3x3 matrix."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    if abs(det) < 1e-18:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    inv_det = 1.0 / det
    return [
        [c00 * inv_det, -(b * i - c * h) * inv_det, (b * f - c * e) * inv_det],
        [c01 * inv_det, (a * i - c * g) * inv_det, -(a * f - c * d) * inv_det],
        [c02 * inv_det, -(a * h - b * g) * inv_det, (a * e - b * d) * inv_det],
    ]


def eskf_update_3d(P, H, r, R_var):
    """
    3D Kalman measurement update on 15-state error vector dx:
      S = H P H^T + R_var * I_3
      K = P H^T S^-1
      dx = K r
      P = (I - K H) P
    H is 3x15, r is 3-list. Returns (dx_15, updated_P).
    """
    # PHt: 15x3
    PHt = [[sum(P[i][k] * H[j][k] for k in range(15)) for j in range(3)] for i in range(15)]
    # S: 3x3
    S = [[sum(H[i][k] * PHt[k][j] for k in range(15)) + (R_var if i == j else 0.0)
          for j in range(3)] for i in range(3)]
    S_inv = inv_3x3(S)
    # K: 15x3
    K = [[sum(PHt[i][k] * S_inv[k][j] for k in range(3)) for j in range(3)] for i in range(15)]
    # dx: 15
    dx = [sum(K[i][j] * r[j] for j in range(3)) for i in range(15)]
    # P = P - K (H P)
    HP = [[sum(H[i][k] * P[k][j] for k in range(15)) for j in range(15)] for i in range(3)]
    P_new = [[P[i][j] - sum(K[i][k] * HP[k][j] for k in range(3)) for j in range(15)] for i in range(15)]
    return dx, P_new


def process(inp, out, blind_extra_s):
    _sqrt, _sin, _cos = sqrt, sin, cos
    write = out.write
    lines = iter(inp)
    out.write('# timestamp tx ty tz qx qy qz qw\n')

    # ---------------- 1) motion-aware initial window: gyro bias + gravity direction -------------
    init_ts = []
    first_ts = last_ts = None
    n = 0
    sgx = sgy = sgz = sax = say = saz = 0.0
    prev_raw = None
    swap = False
    init_ns = int(INIT_SECONDS * 1e9)

    for line in lines:
        if line[0] in '#\r\n':
            continue
        ts, v = parse(line)
        if ts is None:
            if last_ts is None:
                continue
            ts = last_ts + 5000000
        if first_ts is None:
            first_ts = ts

        if v is not None and v != prev_raw:
            if n == 0:
                m3 = _sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                m6 = _sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5])
                if m3 > 5.0 and m6 < 5.0:
                    swap = True
            if swap:
                ax0, ay0, az0, gx0, gy0, gz0 = v
            else:
                gx0, gy0, gz0, ax0, ay0, az0 = v

            # Motion guard: if vehicle starts moving after at least 0.25s, stop init early
            if n > 20:
                wn = _sqrt((gx0 - sgx / n) ** 2 + (gy0 - sgy / n) ** 2 + (gz0 - sgz / n) ** 2)
                an = _sqrt(ax0 * ax0 + ay0 * ay0 + az0 * az0)
                g_est = _sqrt((sax / n) ** 2 + (say / n) ** 2 + (saz / n) ** 2)
                if wn > INIT_MOTION_W or abs(an - g_est) > INIT_MOTION_A:
                    # Don't discard this moving sample; break and let main loop process it
                    # (We emit the stationary init rows below, then process this row)
                    break

            sgx += gx0; sgy += gy0; sgz += gz0
            sax += ax0; say += ay0; saz += az0
            n += 1
            prev_raw = v

        init_ts.append(ts)
        last_ts = ts
        if ts - first_ts >= init_ns or len(init_ts) >= INIT_MAX_ROWS:
            break

    if not init_ts:
        sys.exit('ERROR: no IMU rows found')

    if len(init_ts) > 1:
        diffs = sorted(init_ts[i + 1] - init_ts[i] for i in range(len(init_ts) - 1) if init_ts[i + 1] > init_ts[i])
        nominal_ns = diffs[len(diffs) // 2] if diffs else 5000000
        if nominal_ns <= 0:
            nominal_ns = 5000000
    else:
        nominal_ns = 5000000
    nominal_s = nominal_ns * 1e-9
    gap_ns = GAP_FACTOR * nominal_ns

    if n > 0:
        bgx, bgy, bgz = sgx / n, sgy / n, sgz / n
        fx, fy, fz = sax / n, say / n, saz / n
        G = _sqrt(fx * fx + fy * fy + fz * fz)
        if G < 8.0 or G > 11.5:
            G = NOMINAL_G
        fn = _sqrt(fx * fx + fy * fy + fz * fz)
        ux, uy, uz = fx / fn, fy / fn, fz / fn
    else:
        bgx = bgy = bgz = 0.0
        G = NOMINAL_G
        fx, fy, fz = 0.0, 0.0, G
        ux, uy, uz = 0.0, 0.0, 1.0

    bax = bay = baz = 0.0

    # 3-tap median filter history buffers
    gx_h0 = gx_h1 = bgx
    gy_h0 = gy_h1 = bgy
    gz_h0 = gz_h1 = bgz
    ax_h0 = ax_h1 = fx
    ay_h0 = ay_h1 = fy
    az_h0 = az_h1 = fz

    # Initial orientation: align measured specific force "up" onto world +Z (yaw = 0)
    if uz > -0.999999:
        qw = _sqrt((1.0 + uz) * 0.5)
        qx = uy / (2.0 * qw)
        qy = -ux / (2.0 * qw)
        qz = 0.0
    else:
        qw, qx, qy, qz = 0.0, 1.0, 0.0, 0.0
    nn = 1.0 / _sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw *= nn; qx *= nn; qy *= nn; qz *= nn

    # Emit stationary init window poses
    pose0 = ' 0.000000 0.000000 0.000000 %.8f %.8f %.8f %.8f\n' % (qx, qy, qz, qw)
    for t in init_ts:
        write(fmt_ts(t) + pose0)
    rows = len(init_ts)

    # ---------------- 2) Initialize 15x15 ESKF Covariance P --------------------------
    # State order: [dp(0..2), dv(3..5), dtheta(6..8), dbg(9..11), dba(12..14)]
    P = [[0.0] * 15 for _ in range(15)]
    for i in range(3):
        P[i][i] = 1e-4          # initial position uncertainty
        P[i + 3][i + 3] = 1e-4  # initial velocity uncertainty
        P[i + 6][i + 6] = 1e-3  # initial tilt uncertainty
        P[i + 9][i + 9] = 1e-4  # initial gyro bias uncertainty
        P[i + 12][i + 12] = 1e-2  # initial accel bias uncertainty

    # ---------------- 3) Main streaming ESKF loop ----------------------------------
    px = py = pz = vx = vy = vz = 0.0
    prev_ts = last_ts
    invalid_t = 0.0
    stuck_count = 0
    leak_on = VEL_LEAK_TAU > 0.0
    leak_inv = 1.0 / VEL_LEAK_TAU if leak_on else 0.0
    fmt = '%s %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n'

    sa2 = SIGMA_ACC * SIGMA_ACC
    sg2 = SIGMA_GYRO * SIGMA_GYRO
    sba2 = SIGMA_BA * SIGMA_BA
    sbg2 = SIGMA_BG * SIGMA_BG
    rg_var = R_GRAV * R_GRAV
    rz_var = R_ZUPT * R_ZUPT

    for line in lines:
        if line[0] in '#\r\n':
            continue
        ts, v = parse(line)
        if ts is None:
            ts = prev_ts + nominal_ns
        dts = ts - prev_ts
        prev_ts = ts

        if dts <= 0:
            # Duplicated timestamp row: emit current pose without integrating
            if v is not None:
                prev_raw = v
            write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
            rows += 1
            continue
        dt = dts * 1e-9

        if v is None:
            # Blackout / all-zero / NaN row: coast & inflate covariance
            invalid_t += dt
            px, py, pz, vx, vy, vz = coast(px, py, pz, vx, vy, vz, dt, invalid_t > BLIND_AFTER)
            for i in range(3):
                P[i][i] += 0.5 * dt
                P[i + 3][i + 3] += 0.2 * dt
                P[i + 6][i + 6] += 0.02 * dt
            write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
            rows += 1
            continue

        # Stuck sensor check (frozen sample values over 2-10 frames)
        is_stuck = (v == prev_raw)
        if is_stuck:
            stuck_count += 1
        else:
            stuck_count = 0
            prev_raw = v

        if stuck_count > 10:
            invalid_t += dt
            px, py, pz, vx, vy, vz = coast(px, py, pz, vx, vy, vz, dt, invalid_t > BLIND_AFTER)
            write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
            rows += 1
            continue

        if dts > gap_ns:
            # Dropped single row or burst dropout: coast across missing interval
            hole = dt - nominal_s
            invalid_t += hole
            px, py, pz, vx, vy, vz = coast(px, py, pz, vx, vy, vz, hole, invalid_t > BLIND_AFTER)
            for i in range(3):
                P[i][i] += 0.5 * hole
                P[i + 3][i + 3] += 0.2 * hole
                P[i + 6][i + 6] += 0.02 * hole
            dt = nominal_s
        invalid_t = 0.0

        if swap:
            raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz = v
        else:
            raw_gx, raw_gy, raw_gz, raw_ax, raw_ay, raw_az = v

        # 3-tap median + jump guard for single-sample spikes
        if not is_stuck:
            gx = median3(gx_h0, gx_h1, raw_gx)
            gy = median3(gy_h0, gy_h1, raw_gy)
            gz = median3(gz_h0, gz_h1, raw_gz)
            ax = median3(ax_h0, ax_h1, raw_ax)
            ay = median3(ay_h0, ay_h1, raw_ay)
            az = median3(az_h0, az_h1, raw_az)

            if SPIKE_A > 0.0:
                if abs(ax - ax_h1) > SPIKE_A or abs(ay - ay_h1) > SPIKE_A or abs(az - az_h1) > SPIKE_A:
                    ax, ay, az = ax_h1, ay_h1, az_h1
                if abs(gx - gx_h1) > SPIKE_G or abs(gy - gy_h1) > SPIKE_G or abs(gz - gz_h1) > SPIKE_G:
                    gx, gy, gz = gx_h1, gy_h1, gz_h1

            gx_h0, gx_h1 = gx_h1, raw_gx
            gy_h0, gy_h1 = gy_h1, raw_gy
            gz_h0, gz_h1 = gz_h1, raw_gz
            ax_h0, ax_h1 = ax_h1, raw_ax
            ay_h0, ay_h1 = ay_h1, raw_ay
            az_h0, az_h1 = az_h1, raw_az
        else:
            gx, gy, gz = gx_h1, gy_h1, gz_h1
            ax, ay, az = ax_h1, ay_h1, az_h1

        # Bias-corrected measurements
        wx = gx - bgx
        wy = gy - bgy
        wz = gz - bgz
        ax_c = ax - bax
        ay_c = ay - bay
        az_c = az - baz

        # ---- Nominal State Propagation ----
        th2 = wx * wx + wy * wy + wz * wz
        if th2 > 1e-14:
            mag = _sqrt(th2)
            hw = 0.5 * mag * dt
            s = _sin(hw) / mag
            c = _cos(hw)
        else:
            s = 0.5 * dt
            c = 1.0
        dx_q = wx * s; dy_q = wy * s; dz_q = wz * s
        nw = qw * c - qx * dx_q - qy * dy_q - qz * dz_q
        nx = qw * dx_q + qx * c + qy * dz_q - qz * dy_q
        ny = qw * dy_q - qx * dz_q + qy * c + qz * dx_q
        nz = qw * dz_q + qx * dy_q - qy * dx_q + qz * c
        nn = 1.0 / _sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        qw = nw * nn; qx = nx * nn; qy = ny * nn; qz = nz * nn

        # Rotation matrix R (body -> world)
        xx = qx * qx; yy = qy * qy; zz = qz * qz
        xy = qx * qy; xz = qx * qz; yz = qy * qz
        wxq = qw * qx; wyq = qw * qy; wzq = qw * qz
        r00 = 1.0 - 2.0 * (yy + zz); r01 = 2.0 * (xy - wzq);       r02 = 2.0 * (xz + wyq)
        r10 = 2.0 * (xy + wzq);       r11 = 1.0 - 2.0 * (xx + zz); r12 = 2.0 * (yz - wxq)
        r20 = 2.0 * (xz - wyq);       r21 = 2.0 * (yz + wxq);       r22 = 1.0 - 2.0 * (xx + yy)

        awx = r00 * ax_c + r01 * ay_c + r02 * az_c
        awy = r10 * ax_c + r11 * ay_c + r12 * az_c
        awz = r20 * ax_c + r21 * ay_c + r22 * az_c - G

        hdt2 = 0.5 * dt * dt
        px += vx * dt + awx * hdt2
        py += vy * dt + awy * hdt2
        pz += vz * dt + awz * hdt2
        vx += awx * dt
        vy += awy * dt
        vz += awz * dt

        # ---- ESKF Error-State Covariance Propagation: P = F P F^T + Q ----
        # Build 15x15 transition matrix F
        F = [[1.0 if i == j else 0.0 for j in range(15)] for i in range(15)]
        for i in range(3):
            F[i][i + 3] = dt
            F[i + 6][i + 9] = -dt

        # -R * [a_c]_x * dt  (couples attitude error dtheta into velocity error dv)
        # [a]_x = [[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]]
        # Ra0 = R[:, 0], Ra1 = R[:, 1], Ra2 = R[:, 2]
        # -R [a]_x = [-(r_i1*az - r_i2*ay), -(-r_i0*az + r_i2*ax), -(r_i0*ay - r_i1*ax)]
        F[3][6] = -(r01 * az_c - r02 * ay_c) * dt
        F[3][7] = -(-r00 * az_c + r02 * ax_c) * dt
        F[3][8] = -(r00 * ay_c - r01 * ax_c) * dt

        F[4][6] = -(r11 * az_c - r12 * ay_c) * dt
        F[4][7] = -(-r10 * az_c + r12 * ax_c) * dt
        F[4][8] = -(r10 * ay_c - r11 * ax_c) * dt

        F[5][6] = -(r21 * az_c - r22 * ay_c) * dt
        F[5][7] = -(-r20 * az_c + r22 * ax_c) * dt
        F[5][8] = -(r20 * ay_c - r21 * ax_c) * dt

        # -R * dt (couples accel bias error dba into velocity error dv)
        F[3][12] = -r00 * dt; F[3][13] = -r01 * dt; F[3][14] = -r02 * dt
        F[4][12] = -r10 * dt; F[4][13] = -r11 * dt; F[4][14] = -r12 * dt
        F[5][12] = -r20 * dt; F[5][13] = -r21 * dt; F[5][14] = -r22 * dt

        # Attitude error transition: I - [w]_x * dt
        F[6][7] = wz * dt;  F[6][8] = -wy * dt
        F[7][6] = -wz * dt; F[7][8] = wx * dt
        F[8][6] = wy * dt;  F[8][7] = -wx * dt

        FP = mat_mul_15(F, P)
        P = mat_mul_15_T(FP, F)

        # Add discrete process noise Q
        qv = sa2 * dt
        qth = sg2 * dt
        qbg = sbg2 * dt
        qba = sba2 * dt
        for i in range(3):
            P[i + 3][i + 3] += qv
            P[i + 6][i + 6] += qth
            P[i + 9][i + 9] += qbg
            P[i + 12][i + 12] += qba

        # ---- ESKF Measurement Updates ----
        fn = _sqrt(ax_c * ax_c + ay_c * ay_c + az_c * az_c)
        dev = abs(fn - G)
        w_norm = _sqrt(wx * wx + wy * wy + wz * wz)

        # 1) Gravity-vector direction update (corrects roll/pitch tilt + gyro bias)
        if not is_stuck and dev < GATE_GRAV and fn > 1e-6:
            # Predicted specific force in body frame when at rest: g_b = R^T * [0, 0, G]^T = G * [r20, r21, r22]
            gx_pred = G * r20
            gy_pred = G * r21
            gz_pred = G * r22
            res_g = [ax_c - gx_pred, ay_c - gy_pred, az_c - gz_pred]

            # Jacobian H_g: d(R^T g_w)/d(dtheta) = [g_b]_x, plus I_3 wrt dba
            H_g = [[0.0] * 15 for _ in range(3)]
            H_g[0][7] = -gz_pred; H_g[0][8] = gy_pred
            H_g[1][6] = gz_pred;  H_g[1][8] = -gx_pred
            H_g[2][6] = -gy_pred; H_g[2][7] = gx_pred
            H_g[0][12] = 1.0; H_g[1][13] = 1.0; H_g[2][14] = 1.0

            # Scale variance up smoothly near gate edge
            scale_r = 1.0 + 4.0 * (dev / GATE_GRAV)
            dx, P = eskf_update_3d(P, H_g, res_g, rg_var * scale_r)

            # Inject error state into nominal state
            px += dx[0]; py += dx[1]; pz += dx[2]
            vx += dx[3]; vy += dx[4]; vz += dx[5]
            dtx, dty, dtz = dx[6], dx[7], dx[8]
            nw = qw - 0.5 * (qx * dtx + qy * dty + qz * dtz)
            nx = qx + 0.5 * (qw * dtx + qy * dtz - qz * dty)
            ny = qy + 0.5 * (qw * dty - qx * dtz + qz * dtx)
            nz = qz + 0.5 * (qw * dtz + qx * dty - qy * dtx)
            nn = 1.0 / _sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            qw = nw * nn; qx = nx * nn; qy = ny * nn; qz = nz * nn
            bgx += dx[9]; bgy += dx[10]; bgz += dx[11]
            bax += dx[12]; bay += dx[13]; baz += dx[14]

        # 2) Quasi-static ZUPT update (corrects velocity, tilt, and accel bias jointly)
        if not is_stuck and w_norm < ZUPT_W_THRESH and dev < ZUPT_A_THRESH:
            H_z = [[0.0] * 15 for _ in range(3)]
            H_z[0][3] = 1.0; H_z[1][4] = 1.0; H_z[2][5] = 1.0
            res_z = [-vx, -vy, -vz]

            dx, P = eskf_update_3d(P, H_z, res_z, rz_var)

            px += dx[0]; py += dx[1]; pz += dx[2]
            vx += dx[3]; vy += dx[4]; vz += dx[5]
            dtx, dty, dtz = dx[6], dx[7], dx[8]
            nw = qw - 0.5 * (qx * dtx + qy * dty + qz * dtz)
            nx = qx + 0.5 * (qw * dtx + qy * dtz - qz * dty)
            ny = qy + 0.5 * (qw * dty - qx * dtz + qz * dtx)
            nz = qz + 0.5 * (qw * dtz + qx * dty - qy * dtx)
            nn = 1.0 / _sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            qw = nw * nn; qx = nx * nn; qy = ny * nn; qz = nz * nn
            bgx += dx[9]; bgy += dx[10]; bgz += dx[11]
            bax += dx[12]; bay += dx[13]; baz += dx[14]

        # Optional gentle velocity leak to prevent long-horizon drift
        if leak_on:
            f = exp(-dt * leak_inv)
            vx *= f; vy *= f; vz *= f

        write(fmt % (fmt_ts(ts), px, py, pz, qx, qy, qz, qw))
        rows += 1

    # ---------------- 4) optional blind extension after the file ends -----------------
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