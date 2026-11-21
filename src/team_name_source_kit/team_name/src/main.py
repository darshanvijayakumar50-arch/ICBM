#!/usr/bin/env python3
"""
Track 03 - Sensor Jamming & Progressive Degradation
Streaming IMU-only 6-DoF trajectory estimator (pure Python, no third-party imports).

Usage:
    python main.py <input> [-o estimated_trajectory.txt] [--meta metadata.json]
                   [--order auto|accel|gyro] [--gyro-deg] [--trace-mem]

<input> may be: a CSV file, a folder containing imu0/data.csv, or a .zip containing imu0/data.csv.

Pipeline (one pass over the file, one output pose per input row):
  1. Parse rows (accel-first OR EuRoC gyro-first auto-detected, header optional, ns timestamps).
  2. Initialise gravity direction, gravity magnitude and gyro bias from the first ~1 s (clean, protected).
  3. Per sample: spike gating, stuck/duplicate detection, gap (dropout) detection from timestamps.
  4. Mahony complementary filter (gyro integration + gravity correction + gyro-bias integrator).
  5. Strapdown velocity/position with LEAKY velocity, zero-velocity updates and accel-bias learning.
  6. If the IMU reads exactly all-zeros (total failure) -> blind dead-reckoning with decaying rates.
"""
import sys
import os
import time
import math
from itertools import chain

# ----------------------------------------------------------------------------------------------
# Tunable parameters (documented in README). Untested on real EuRoC/TUM-VI data - see README.
# ----------------------------------------------------------------------------------------------
P = {
    "init_rows": 200,        # rows used for initial gravity / gyro-bias estimate (~1 s at 200 Hz)
    "kp": 2.0,               # Mahony proportional gain (gravity correction strength)
    "ki": 0.05,              # Mahony integral gain (gyro bias learning)
    "acc_gate": 0.8,         # accel-norm tolerance around g for using accel to correct attitude (m/s^2)
    "spike_acc": 2.5,        # |accel jump| above this between samples = spike (m/s^2)
    "spike_gyr": 0.25,       # |gyro jump| above this between samples = spike (rad/s)
    "leak": 1.0,             # base velocity leak rate (1/s) - anti-drift mechanism (grows with noise)
    "pos_leak": 0.1,         # pulls position toward the start point (1/s, grows with noise); bounds drift
    "leak_max_mult": 4.0,    # velocity leak grows with measured noise growth, up to this multiple
    "pos_leak_max_mult": 8.0,
    "zupt_damp": 0.7,        # velocity multiplier per sample while stationary
    "vmax": 4.0,             # velocity sanity clamp (m/s)
    "tau_v_pred": 1.5,       # blind phase: velocity decay time constant (s)
    "tau_w_pred": 2.0,       # blind phase: angular-rate decay time constant (s)
    "unlatch_rows": 25,      # consecutive non-zero valid rows needed to leave failure mode
    "dt_cap": 2.0,           # max single integration step (s)
}

G0 = 9.80665


def _open_input(path):
    import io
    if os.path.isdir(path):
        for cand in (os.path.join("imu0", "data.csv"), "data.csv"):
            full = os.path.join(path, cand)
            if os.path.isfile(full):
                return open(full, "r", encoding="utf-8-sig", newline=None), None
        raise FileNotFoundError("no imu0/data.csv found inside folder: %s" % path)
    if path.lower().endswith(".zip"):
        import zipfile
        zf = zipfile.ZipFile(path)
        names = zf.namelist()
        pick = None
        for n in names:
            nn = n.replace("\\", "/").lower()
            if nn.endswith("imu0/data.csv"):
                pick = n
                break
        if pick is None:
            for n in names:
                if n.lower().endswith("data.csv"):
                    pick = n
                    break
        if pick is None:
            for n in names:
                if n.lower().endswith(".csv"):
                    pick = n
                    break
        if pick is None:
            raise FileNotFoundError("no CSV found inside zip: %s" % path)
        return io.TextIOWrapper(zf.open(pick), encoding="utf-8-sig", newline=None), zf
    return open(path, "r", encoding="utf-8-sig", newline=None), None


def _peak_rss_kb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform == "darwin" else float(r)
    except Exception:
        pass
    try:  # Windows (untested)
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 1024.0
    except Exception:
        return None


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _parse_numeric(line):
    """Tolerant parse of one CSV line into a list of floats, or None."""
    try:
        return [float(x) for x in line.split(",")]
    except ValueError:
        return None


def run(in_path, out_path, order="auto", gyro_deg=False, params=None):
    p = dict(P)
    if params:
        p.update(params)
    t0 = time.perf_counter()
    sqrt, sin, cos, exp = math.sqrt, math.sin, math.cos, math.exp

    fh, zf = _open_input(in_path)

    # ---------------- stage 1: header + initial window (strings only, tiny) ----------------
    pre = []
    hdr = None
    it = iter(fh)
    for line in it:
        s = line.strip()
        if not s:
            continue
        if s[0] == "#" or s[0].isalpha():
            if hdr is None:
                hdr = s.lower()
            continue
        pre.append(s)
        if len(pre) >= p["init_rows"]:
            break
    if not pre:
        raise ValueError("input contains no data rows")

    nums = [r for r in (_parse_numeric(s) for s in pre) if r is not None and len(r) >= 7]
    if len(nums) < 3:
        raise ValueError("could not parse the first data rows (expected 7 comma-separated columns)")

    # column order
    gyro_first = None
    if order == "gyro":
        gyro_first = True
    elif order == "accel":
        gyro_first = False
    elif hdr:
        iw = hdr.find("w_rs")
        ia_ = hdr.find("a_rs")
        if iw < 0 and ia_ < 0:
            iw = hdr.find("gyro")
            ia_ = hdr.find("acc")
        if iw >= 0 and ia_ >= 0:
            gyro_first = iw < ia_
    if gyro_first is None:
        n13 = _median([sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2) for r in nums])
        n46 = _median([sqrt(r[4] ** 2 + r[5] ** 2 + r[6] ** 2) for r in nums])

        def sc(n):
            return min(abs(n - G0), G0 * abs(n - 1.0))
        gyro_first = sc(n46) < sc(n13)
    ia = 4 if gyro_first else 1
    ig = 1 if gyro_first else 4

    # units
    dts_raw = [nums[i][0] - nums[i - 1][0] for i in range(1, len(nums)) if nums[i][0] > nums[i - 1][0]]
    dmed = _median(dts_raw) if dts_raw else 5e6
    if dmed > 1e6:
        mi, ts_unit = 1, "ns"
    elif dmed > 1e3:
        mi, ts_unit = 1000, "us"
    elif dmed > 1.0:
        mi, ts_unit = 1000000, "ms"
    else:
        mi, ts_unit = 1000000000, "s"
    mf = float(mi)
    ndt = dmed * mi * 1e-9
    if not (1e-4 <= ndt <= 0.1):
        ndt = 0.005
    gap_thr = 1.6 * ndt

    anorm_med = _median([sqrt(r[ia] ** 2 + r[ia + 1] ** 2 + r[ia + 2] ** 2) for r in nums])
    sa = G0 if 0.5 < anorm_med < 2.0 else 1.0
    sg = (math.pi / 180.0) if gyro_deg else 1.0
    scale_a = sa != 1.0
    scale_g = sg != 1.0

    # ---------------- stage 2: initial state from clean start ----------------
    n = len(nums)
    ma = [sum(r[ia + k] for r in nums) / n * sa for k in range(3)]
    mg = [sum(r[ig + k] for r in nums) / n * sg for k in range(3)]
    g_est = sqrt(ma[0] ** 2 + ma[1] ** 2 + ma[2] ** 2)
    if not (5.0 < g_est < 15.0):
        g_est = G0
        ma = [0.0, 0.0, G0]
    gmag0 = sqrt(mg[0] ** 2 + mg[1] ** 2 + mg[2] ** 2)
    if gmag0 < 0.15:
        bg0, bg1, bg2 = mg
    else:
        bg0 = bg1 = bg2 = 0.0
    d2s = 0.0
    d2n = 0
    for i in range(1, n):
        d2s += sum(((nums[i][ia + k] - nums[i - 1][ia + k]) * sa) ** 2 for k in range(3))
        d2n += 1
    diff2_0 = max(d2s / d2n if d2n else 1e-4, 1e-6)

    # initial attitude: rotation taking measured "up" (body) to world +z, yaw = 0
    ux, uy, uz = ma[0] / g_est, ma[1] / g_est, ma[2] / g_est
    ax_, ay_, az_ = uy * 1.0 - uz * 0.0, uz * 0.0 - ux * 1.0, 0.0  # u x z = (uy, -ux, 0)
    sn = sqrt(ax_ * ax_ + ay_ * ay_)
    ang = math.atan2(sn, uz)
    if sn < 1e-9:
        qw, qx, qy, qz = (1.0, 0.0, 0.0, 0.0) if uz > 0 else (0.0, 1.0, 0.0, 0.0)
    else:
        h = 0.5 * ang
        s_ = sin(h) / sn
        qw, qx, qy, qz = cos(h), ax_ * s_, ay_ * s_, 0.0
    q_init = (qw, qx, qy, qz)

    # ---------------- state ----------------
    px = py = pz = 0.0
    vx = vy = vz = 0.0
    bax = bay = baz = 0.0
    lax, lay, laz = ma[0], ma[1], ma[2]       # last ACCEPTED accel
    lgx, lgy, lgz = mg[0], mg[1], mg[2]       # last ACCEPTED gyro
    rax, ray, raz, rgx, rgy, rgz = lax, lay, laz, lgx, lgy, lgz  # last RAW
    rej_prev = False
    m0, m1, m2 = ma                           # EWMA accel mean
    var_a = diff2_0 * 0.5
    diff2 = diff2_0
    gm0 = gm1 = gm2 = 0.0                     # EWMA corrected gyro mean
    mu0 = p["pos_leak"]
    mu = mu0
    lam0 = p["leak"]
    lam = lam0
    kp, ki = p["kp"], p["ki"]
    agate = p["acc_gate"]
    SA, SG = p["spike_acc"], p["spike_gyr"]
    zdamp = p["zupt_damp"]
    VMAX = p["vmax"]
    VMAX2 = VMAX * VMAX
    tau_v, tau_w = p["tau_v_pred"], p["tau_w_pred"]
    unl = p["unlatch_rows"]
    dt_cap = p["dt_cap"]
    failed = False
    ok_run = 0
    pw0 = pw1 = pw2 = 0.0
    prev_ns = None
    first = True
    cnt = 0
    # stats
    n_out = n_bad = n_spike = n_stuck = n_dup = n_gap = n_zupt = n_unparse = n_reset = n_nonmono = 0
    n_fail_rows = 0
    max_gap_s = 0.0
    fail_idx = -1
    fail_ts = None
    first_ts = last_ts = None

    out = open(out_path, "w", newline="\n")
    out.write("# timestamp[s] tx ty tz qx qy qz qw\n")
    buf = []
    fmt = "%d.%09d %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n"

    # ---------------- main streaming loop ----------------
    for line in chain(pre, it):
        t = line.split(",")
        try:
            try:
                ts = int(t[0]) * mi
            except ValueError:
                ts = int(round(float(t[0]) * mf))
        except (ValueError, IndexError):
            if line.lstrip()[:1] != "#" and line.strip():
                n_unparse += 1
            continue
        valid = True
        try:
            ax = float(t[ia]); ay = float(t[ia + 1]); az = float(t[ia + 2])
            gx = float(t[ig]); gy = float(t[ig + 1]); gz = float(t[ig + 2])
            if scale_a:
                ax *= sa; ay *= sa; az *= sa
            if scale_g:
                gx *= sg; gy *= sg; gz *= sg
            s6 = abs(ax) + abs(ay) + abs(az) + abs(gx) + abs(gy) + abs(gz)
            if not s6 < 1e5:
                valid = False
        except (ValueError, IndexError):
            valid = False
            s6 = 1.0

        if prev_ns is None:
            dt = 0.0
        else:
            d = ts - prev_ns
            if d > 0:
                dt = d * 1e-9
                if dt > dt_cap:
                    dt = dt_cap
            else:
                dt = 0.0
                if d < 0:
                    n_nonmono += 1
        if prev_ns is None or ts > prev_ns:
            prev_ns = ts
        if first_ts is None:
            first_ts = ts
        last_ts = ts

        if dt > 0.0:
            # ---- failure latch logic ----
            if failed:
                if valid and s6 > 0.0:
                    ok_run += 1
                    if ok_run >= unl:
                        failed = False
                        ok_run = 0
                        lax, lay, laz, lgx, lgy, lgz = ax, ay, az, gx, gy, gz
                        rax, ray, raz, rgx, rgy, rgz = ax, ay, az, gx, gy, gz
                else:
                    ok_run = 0
            elif valid and s6 == 0.0:
                failed = True
                ok_run = 0
                fail_idx = n_out
                fail_ts = ts
                pw0, pw1, pw2 = gm0, gm1, gm2

            if failed:
                # ======== T2: BLIND DEAD-RECKONING (no usable sensor data) ========
                n_fail_rows += 1
                kw = 1.0 - dt / tau_w
                if kw < 0.0:
                    kw = 0.0
                pw0 *= kw; pw1 *= kw; pw2 *= kw
                th2 = pw0 * pw0 + pw1 * pw1 + pw2 * pw2
                th = sqrt(th2) * dt
                if th < 0.02:
                    s_ = 0.5 * dt
                    c_ = 1.0 - th * th * 0.125
                else:
                    s_ = sin(0.5 * th) / sqrt(th2)
                    c_ = cos(0.5 * th)
                dx, dy, dz = pw0 * s_, pw1 * s_, pw2 * s_
                nw = qw * c_ - qx * dx - qy * dy - qz * dz
                nx = qw * dx + qx * c_ + qy * dz - qz * dy
                ny = qw * dy - qx * dz + qy * c_ + qz * dx
                nz = qw * dz + qx * dy - qy * dx + qz * c_
                inv = 1.0 / sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
                qw, qx, qy, qz = nw * inv, nx * inv, ny * inv, nz * inv
                kv = 1.0 - dt / tau_v
                if kv < 0.0:
                    kv = 0.0
                vx *= kv; vy *= kv; vz *= kv
                px += vx * dt; py += vy * dt; pz += vz * dt
            else:
                # ======== MEASUREMENT PATH (T1 + T3) ========
                gap = dt > gap_thr
                stuck = False
                if not valid:
                    # unparseable / non-finite values: treat as a dropped measurement, hold last good
                    n_bad += 1
                    fx, fy, fz = lax, lay, laz
                    wx, wy, wz = lgx, lgy, lgz
                    meas_ok = False
                else:
                    meas_ok = True
                    stuck = (ax == rax and ay == ray and az == raz and gx == rgx and gy == rgy and gz == rgz)
                    if (abs(ax - lax) > SA or abs(ay - lay) > SA or abs(az - laz) > SA or
                            abs(gx - lgx) > SG or abs(gy - lgy) > SG or abs(gz - lgz) > SG):
                        if rej_prev and (abs(ax - rax) <= SA and abs(ay - ray) <= SA and abs(az - raz) <= SA and
                                         abs(gx - rgx) <= SG and abs(gy - rgy) <= SG and abs(gz - rgz) <= SG):
                            rej_prev = False      # two consistent samples in a row: genuine change
                            fx, fy, fz, wx, wy, wz = ax, ay, az, gx, gy, gz
                        else:
                            rej_prev = True       # isolated outlier -> hold previous accepted value
                            n_spike += 1
                            fx, fy, fz = lax, lay, laz
                            wx, wy, wz = lgx, lgy, lgz
                            meas_ok = False
                    else:
                        rej_prev = False
                        fx, fy, fz, wx, wy, wz = ax, ay, az, gx, gy, gz
                    rax, ray, raz, rgx, rgy, rgz = ax, ay, az, gx, gy, gz
                if stuck:
                    n_stuck += 1
                    meas_ok = False

                # ---- T3: dropout handling - integrate across the timestamp gap with the mean of the two ends
                if gap:
                    n_gap += 1
                    if dt > max_gap_s:
                        max_gap_s = dt
                    fxm = 0.5 * (fx + lax); fym = 0.5 * (fy + lay); fzm = 0.5 * (fz + laz)
                    wxm = 0.5 * (wx + lgx); wym = 0.5 * (wy + lgy); wzm = 0.5 * (wz + lgz)
                else:
                    fxm, fym, fzm, wxm, wym, wzm = fx, fy, fz, wx, wy, wz

                # ---- Mahony attitude update ----
                wcx = wxm - bg0; wcy = wym - bg1; wcz = wzm - bg2
                an = sqrt(fxm * fxm + fym * fym + fzm * fzm)
                if meas_ok and not gap and an > 1e-3 and abs(an - g_est) < agate:
                    inv = 1.0 / an
                    nx_, ny_, nz_ = fxm * inv, fym * inv, fzm * inv
                    vx_ = 2.0 * (qx * qz - qw * qy)
                    vy_ = 2.0 * (qw * qx + qy * qz)
                    vz_ = qw * qw - qx * qx - qy * qy + qz * qz
                    ex = ny_ * vz_ - nz_ * vy_
                    ey = nz_ * vx_ - nx_ * vz_
                    ez = nx_ * vy_ - ny_ * vx_
                    bg0 -= ki * ex * dt; bg1 -= ki * ey * dt; bg2 -= ki * ez * dt
                    if bg0 > 0.3: bg0 = 0.3
                    elif bg0 < -0.3: bg0 = -0.3
                    if bg1 > 0.3: bg1 = 0.3
                    elif bg1 < -0.3: bg1 = -0.3
                    if bg2 > 0.3: bg2 = 0.3
                    elif bg2 < -0.3: bg2 = -0.3
                    wcx += kp * ex; wcy += kp * ey; wcz += kp * ez
                th2 = wcx * wcx + wcy * wcy + wcz * wcz
                th = sqrt(th2) * dt
                if th < 0.02:
                    s_ = 0.5 * dt
                    c_ = 1.0 - th * th * 0.125
                else:
                    s_ = sin(0.5 * th) / sqrt(th2)
                    c_ = cos(0.5 * th)
                dx, dy, dz = wcx * s_, wcy * s_, wcz * s_
                nw = qw * c_ - qx * dx - qy * dy - qz * dz
                nx = qw * dx + qx * c_ + qy * dz - qz * dy
                ny = qw * dy - qx * dz + qy * c_ + qz * dx
                nz = qw * dz + qx * dy - qy * dx + qz * c_
                inv = 1.0 / sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
                qw, qx, qy, qz = nw * inv, nx * inv, ny * inv, nz * inv

                # ---- strapdown: world acceleration, leaky velocity, position ----
                r00 = 1.0 - 2.0 * (qy * qy + qz * qz); r01 = 2.0 * (qx * qy - qw * qz); r02 = 2.0 * (qx * qz + qw * qy)
                r10 = 2.0 * (qx * qy + qw * qz); r11 = 1.0 - 2.0 * (qx * qx + qz * qz); r12 = 2.0 * (qy * qz - qw * qx)
                r20 = 2.0 * (qx * qz - qw * qy); r21 = 2.0 * (qy * qz + qw * qx); r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
                cx = fxm - bax; cy = fym - bay; cz = fzm - baz
                awx = r00 * cx + r01 * cy + r02 * cz
                awy = r10 * cx + r11 * cy + r12 * cz
                awz = r20 * cx + r21 * cy + r22 * cz - g_est
                ovx, ovy, ovz = vx, vy, vz
                kl = 1.0 - lam * dt
                if kl < 0.0:
                    kl = 0.0
                vx = (vx + awx * dt) * kl
                vy = (vy + awy * dt) * kl
                vz = (vz + awz * dt) * kl

                # ---- stationary detection (EWMA statistics) + ZUPT + accel-bias learning ----
                if meas_ok and not gap:
                    m0 += 0.05 * (fx - m0); m1 += 0.05 * (fy - m1); m2 += 0.05 * (fz - m2)
                    dax = fx - m0; day = fy - m1; daz = fz - m2
                    var_a += 0.05 * (dax * dax + day * day + daz * daz - var_a)
                    gm0 += 0.05 * (wx - bg0 - gm0); gm1 += 0.05 * (wy - bg1 - gm1); gm2 += 0.05 * (wz - bg2 - gm2)
                    jx = fx - lax; jy = fy - lay; jz = fz - laz
                    diff2 += 0.02 * (jx * jx + jy * jy + jz * jz - diff2)
                    mn = sqrt(m0 * m0 + m1 * m1 + m2 * m2)
                    if (gm0 * gm0 + gm1 * gm1 + gm2 * gm2 < 0.0025 and abs(mn - g_est) < 0.4
                            and var_a < 0.75 * diff2 + 0.05):
                        n_zupt += 1
                        vx *= zdamp; vy *= zdamp; vz *= zdamp
                        bax += 0.01 * ((m0 - g_est * r20) - bax)
                        bay += 0.01 * ((m1 - g_est * r21) - bay)
                        baz += 0.01 * ((m2 - g_est * r22) - baz)
                        if bax > 1.0: bax = 1.0
                        elif bax < -1.0: bax = -1.0
                        if bay > 1.0: bay = 1.0
                        elif bay < -1.0: bay = -1.0
                        if baz > 1.0: baz = 1.0
                        elif baz < -1.0: baz = -1.0
                cnt += 1
                if (cnt & 15) == 0:
                    ratio = diff2 / diff2_0
                    mult = 1.0 if ratio < 1.0 else ratio
                    if mult > p["leak_max_mult"]:
                        mult = p["leak_max_mult"]
                    lam = lam0 * mult
                    mult2 = 1.0 if ratio < 1.0 else ratio
                    if mult2 > p["pos_leak_max_mult"]:
                        mult2 = p["pos_leak_max_mult"]
                    mu = mu0 * mult2

                sp2 = vx * vx + vy * vy + vz * vz
                if sp2 > VMAX2:
                    sc_ = VMAX / sqrt(sp2)
                    vx *= sc_; vy *= sc_; vz *= sc_
                kpl = 1.0 - mu * dt
                if kpl < 0.0:
                    kpl = 0.0
                px = (px + 0.5 * (ovx + vx) * dt) * kpl
                py = (py + 0.5 * (ovy + vy) * dt) * kpl
                pz = (pz + 0.5 * (ovz + vz) * dt) * kpl

                if valid and meas_ok:
                    lax, lay, laz, lgx, lgy, lgz = fx, fy, fz, wx, wy, wz
            # ---- numerical safety net ----
            ssum = px + py + pz + vx + vy + vz + qw + qx + qy + qz
            if ssum - ssum != 0.0:
                n_reset += 1
                px = py = pz = vx = vy = vz = 0.0
                qw, qx, qy, qz = q_init
        else:
            if prev_ns is not None and dt == 0.0 and not first:
                n_dup += 1
        first = False

        s_sec, r_ns = divmod(ts, 1000000000)
        buf.append(fmt % (s_sec, r_ns, px, py, pz, qx, qy, qz, qw))
        n_out += 1
        if len(buf) >= 2000:
            out.write("".join(buf))
            buf = []

    if buf:
        out.write("".join(buf))
    out.close()
    fh.close()
    if zf is not None:
        zf.close()

    dur = ((last_ts - first_ts) * 1e-9) if first_ts is not None else 0.0
    runtime = time.perf_counter() - t0
    stats = {
        "rows_written": n_out,
        "duration_s": round(dur, 3),
        "runtime_s": round(runtime, 4),
        "runtime_ms_per_data_second": round(1000.0 * runtime / dur, 3) if dur > 0 else None,
        "peak_rss_kb": _peak_rss_kb(),
        "timestamp_unit_detected": ts_unit,
        "column_order_detected": "gyro-first (EuRoC)" if gyro_first else "accel-first",
        "accel_scaled_from_g": scale_a,
        "gravity_estimate_m_s2": round(g_est, 4),
        "unparseable_rows_skipped": n_unparse,
        "invalid_measurement_rows": n_bad,
        "timestamp_gaps_bridged": n_gap,
        "largest_gap_s": round(max_gap_s, 4),
        "duplicate_timestamp_rows": n_dup,
        "non_monotonic_rows": n_nonmono,
        "spikes_rejected": n_spike,
        "stuck_samples_flagged": n_stuck,
        "zupt_samples": n_zupt,
        "failure_detected": fail_idx >= 0,
        "failure_row_index": fail_idx,
        "failure_start_timestamp_s": (fail_ts * 1e-9) if fail_ts is not None else None,
        "blind_rows": n_fail_rows,
        "numerical_resets": n_reset,
        "python": sys.version.split()[0],
    }
    return stats


def _json(v, ind=0):
    pad = "  " * (ind + 1)
    if isinstance(v, dict):
        items = ['%s"%s": %s' % (pad, k, _json(x, ind + 1)) for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * ind + "}"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    return '"%s"' % str(v).replace("\\", "\\\\").replace('"', '\\"')


def main(argv):
    inp = None
    outp = "estimated_trajectory.txt"
    meta = None
    order = "auto"
    gdeg = False
    trace = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--out") and i + 1 < len(argv):
            outp = argv[i + 1]; i += 2
        elif a == "--meta" and i + 1 < len(argv):
            meta = argv[i + 1]; i += 2
        elif a == "--order" and i + 1 < len(argv):
            order = argv[i + 1]; i += 2
        elif a == "--gyro-deg":
            gdeg = True; i += 1
        elif a == "--trace-mem":
            trace = True; i += 1
        elif a in ("-h", "--help"):
            print(__doc__); return 0
        elif inp is None and not a.startswith("-"):
            inp = a; i += 1
        else:
            print("unknown argument: %s" % a, file=sys.stderr); return 2
    if inp is None:
        print(__doc__); return 2
    if order not in ("auto", "accel", "gyro"):
        print("--order must be auto|accel|gyro", file=sys.stderr); return 2
    if meta is None:
        meta = os.path.join(os.path.dirname(os.path.abspath(outp)), "metadata.json")
    try:
        if trace:
            import tracemalloc
            tracemalloc.start()
        stats = run(inp, outp, order=order, gyro_deg=gdeg)
        if trace:
            stats["tracemalloc_peak_kb"] = round(tracemalloc.get_traced_memory()[1] / 1024.0, 1)
    except (FileNotFoundError, ValueError, OSError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    with open(meta, "w") as f:
        f.write(_json(stats) + "\n")
    print("wrote %s (%d rows) in %.3f s | peak RSS %s KB | failure detected: %s"
          % (outp, stats["rows_written"], stats["runtime_s"], stats["peak_rss_kb"], stats["failure_detected"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
