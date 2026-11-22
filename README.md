# ICBM: IMU-Only Trajectory Estimation Under Sensor Degradation

**Hackathon track:** Track 03, *Path Finder: Trajectory Estimation Under Sensor Degradation*
**Team:** VITC_ICBM, Vellore Institute of Technology (Chennai)
**Members:** Darshan V, Pragadheesh A, Arthi Y, C K Sarveshwaran

ICBM is a pure-Python, streaming inertial dead-reckoning estimator. It reconstructs the 6-DoF trajectory (position and orientation) of an aircraft from a degraded IMU log alone, with no GPS. Around the estimator sit an `evo`-based evaluation tool, a RAM/runtime profiler, and the organizers' degradation generator.

> **Status in one paragraph.** The estimator (`src/main.py`) is implemented and runs on all three bundled datasets. It is a heuristic dead-reckoning design: **not** an EKF, **not** sensor fusion, **not** learned. It produces an 11-column CSV, **not** the TUM-format `estimated_trajectory.txt` that the organizers' specification asks for. Accuracy is moderate on the Medium and Hard datasets and poor on Easy, and the 8 MB memory budget is not demonstrated. See [Challenges and Limitations](#14-challenges-and-limitations) before relying on any number in this README.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Key Features](#4-key-features)
5. [Technology Stack](#5-technology-stack)
6. [Project Architecture](#6-project-architecture)
7. [Repository Structure](#7-repository-structure)
8. [Working Principle](#8-working-principle)
9. [Installation and Setup](#9-installation-and-setup)
10. [Execution Instructions](#10-execution-instructions)
11. [Code Explanation](#11-code-explanation)
12. [Datasets](#12-datasets)
13. [Results and Outputs](#13-results-and-outputs)
14. [Challenges and Limitations](#14-challenges-and-limitations)
15. [Future Improvements](#15-future-improvements)
16. [References and Provenance](#16-references-and-provenance)

---

## 1. Project Overview

An aircraft flies through a GPS-denied, jammed environment. Its IMU (gyroscope and accelerometer) gets progressively worse: noise grows, biases drift, samples are dropped, duplicated or frozen, and finally the sensor fails completely. ICBM estimates the flight path from whatever IMU data survives, and extrapolates through the gaps and through the final blackout.

The core design decisions are:

- **Standard library only** in the estimator. It uses `os`, `csv`, `math`, `time` and `argparse`, so it has no third-party dependency.
- **Streaming, constant memory.** The input CSV is read twice (a short calibration pass, then a full pass). Each pose is written to disk as soon as it is computed. Nothing proportional to the trajectory length is held in memory.
- **Heuristic stabilisers** (stationary detection, bias adaptation, velocity damping, a world-frame acceleration high-pass) keep pure inertial integration from drifting off immediately.
- **A momentum-decay model** for gaps and the final blackout, in place of a physical motion model.

## 2. Problem Statement

The task description below is taken from the hackathon's problem statement as relayed by the team. **The problem-statement PDF is not part of this repository**, so the items marked † could not be re-verified from the repository files.

| Item | Specification |
|---|---|
| Scenario | GPS-denied, hostile environment. Jamming degrades the IMU (rising noise, bias drift, 2 s frame dropouts) until it fails completely.† |
| Task | Estimate the 6-DoF trajectory using only the degraded IMU stream. After the IMU dies, dead-reckon the final 5 s.† |
| Resource limits | ≤ 8 MB peak RAM, ≤ 20 ms compute per 1 s of IMU data, single CPU thread, offline, unknown gravity alignment.† |
| Input | EuRoC-format IMU CSV: `#timestamp [ns]`, gyro (rad/s), accel (m/s²). |
| Required output | `estimated_trajectory.txt` in TUM format: `timestamp[s] tx ty tz qx qy qz qw`, one pose per IMU frame.† |
| Deliverables | `team_name.zip` containing the trajectory, `metadata.json`, `src/main.py` and a README, plus a 2–3 minute walkthrough video.† |
| Scoring (100 pts) | Execution and efficiency 40 % (peak memory 20 %, runtime 20 %). Trajectory closeness 60 % (waypoints 30 %, blind 5 s 20 %, dropout 10 %). Ranked with `evo` using ATE (RMSE) and RPE.† |

## 3. Objectives

1. Produce one pose per input IMU row, in a local frame starting at the origin.
2. Stay inside the resource budget: constant memory, single thread, low per-sample compute.
3. Survive degraded input (NaN or all-zero frames, timestamp gaps, duplicate timestamps) without crashing.
4. Extrapolate plausibly through gaps and the final 5 s blackout.
5. Provide tooling to measure accuracy (APE/RPE) and resource use (peak RSS, runtime).

Objectives 1, 3 and 5 are met. Objective 2 is only partly demonstrated, and objective 4 is implemented heuristically with no accuracy figure for the blind window (see [section 14](#14-challenges-and-limitations)).

## 4. Key Features

**Implemented and working**

| Feature | Where |
|---|---|
| Streaming O(1)-memory estimator, standard library only | `src/main.py` |
| Static calibration: gyro bias, initial roll/pitch, accel bias | `calibrate_from_head` |
| Midpoint quaternion attitude integration | `LowMemoryInertialEstimator.step` |
| Gravity tilt correction (roll/pitch only, gated) | `step` |
| Stationary detection with online gyro and accel bias adaptation | `step` |
| Velocity damping and a slow world-frame acceleration high-pass | `step` |
| Handling of NaN/inf rows, all-zero rows, timestamp gaps and duplicate timestamps | `is_invalid_sample`, `run_trajectory_estimation` |
| Momentum-decay extrapolation during gaps and blackout | `step` (blackout branch) |
| CSV to TUM conversion, `evo` APE/RPE, metrics JSON, 8 diagnostic plots | `src/icbm/evaluate.py` |
| Subprocess RSS poller and RAM-over-time plot | `src/icbm/perf.py` |
| IMU degradation generator (**organizer-provided**) | `src/ext/degrade.py` |

**Not implemented**

- A TUM writer inside `main.py`, `metadata.json`, and the walkthrough video.
- Waypoint or checkpoint logic.
- Spike and stuck-value rejection.
- Kalman or error-state filtering, and any learned model.
- The JSBSim simulation (`src/icbm/simulate.py` is a stub; see [section 11](#simulatepy)).

## 5. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python ≥ 3.13 | `requires-python = ">=3.13"`; `.python-version` pins `3.13` |
| Estimator runtime | Python standard library | `os`, `csv`, `math`, `time`, `argparse` only |
| Environment and packaging | [uv](https://docs.astral.sh/uv/), `uv_build` | `uv.lock` has 29 package entries |
| Evaluation | `evo` ≥ 1.38.1, `numpy` ≥ 2.5.5, `matplotlib` ≥ 3.11.2 | used by `evaluate.py` |
| Profiling | `psutil` ≥ 7.2.2, `matplotlib` | used by `perf.py` |
| Degradation tool | `numpy`, `pandas` ≥ 3.1.1 | used by `ext/degrade.py` |
| Simulation (stub) | `jsbsim` ≥ 1.3.3 | used by `simulate.py`; not part of the estimator |

Minimum versions are those declared in `pyproject.toml`. The exact resolved versions are in `uv.lock`.

## 6. Project Architecture

```
                     ┌────────────────────────── estimator (src/main.py) ──────────────────────────┐
 IMU CSV  ──────────►│ Pass 1: read first 300 valid rows ► calibrate_from_head ► bg, ba, q0        │
 (EuRoC, 200 Hz)     │ Pass 2: stream row by row                                                   │
                     │    parse ► validity / gap check ─┬─ valid, dt ≤ 50 ms ► step(): strapdown   │──► estimated CSV
                     │                                  └─ invalid or dt > 50 ms ► step(): decay   │    (11 columns)
                     │    write one pose row immediately                                           │
                     └─────────────────────────────────────────────────────────────────────────────┘
                                                                                                         │
 ground_truth.csv ──────────────────────────┐                                                            │
                                            ▼                                                            ▼
                               src/icbm/evaluate.py  ◄───────────────────────────────────────  estimated CSV
                               CSV ► TUM ► evo APE/RPE ► metrics.json, diagnostics.npz, 8 PNG plots

 src/icbm/perf.py     runs  python src/main.py <in> <out>  as a subprocess and polls its RSS
 src/ext/degrade.py   turns a clean IMU CSV into a degraded one (organizer-provided)
 src/icbm/simulate.py JSBSim CLI stub, not connected to anything
```

The estimator is deliberately independent of the tooling. `evaluate.py`, `perf.py` and `degrade.py` read and write files, or launch `src/main.py` as a subprocess. They never import it.

## 7. Repository Structure

As delivered in the project ZIP (`__pycache__/` folders omitted):

```
.
├── README.md                  This file
├── pyproject.toml             Project metadata, dependencies, console-script entry points
├── uv.lock                    Locked dependency set (uv)
├── .python-version            3.13
├── .gitignore                 Notably ignores data/, evaluation/, *.csv, *.png, plots/
├── src/
│   ├── main.py                The estimator (306 lines, standard library only)
│   ├── README.md              Earlier short run instructions (superseded by this file)
│   ├── icbm/
│   │   ├── __init__.py        Empty
│   │   ├── evaluate.py        CSV→TUM conversion, evo APE/RPE, metrics, plots (796 lines)
│   │   ├── perf.py            RSS/runtime profiler (198 lines)
│   │   └── simulate.py        JSBSim CLI stub (149 lines)
│   └── ext/
│       └── degrade.py         Organizer-provided IMU degradation generator (277 lines)
├── data/
│   ├── Easy/    imu.csv, degraded.csv,      ground_truth.csv
│   ├── Medium/  imu.csv, degraded.csv,      ground_truth.csv
│   └── Hard/    imu.csv, degraded_imu.csv,  ground_truth.csv, meta.json,
│                estimated.csv (header only, no data rows)
└── evaluation/                Stored outputs of one earlier evaluation run (stale, see §13.3)
    ├── metrics.json, diagnostics.npz
    ├── estimated.tum, ground_truth.tum
    ├── ape_results.zip, ape_plot.pdf
    └── plots/                 8 PNG diagnostic plots
```

`data/` and `evaluation/` are listed in `.gitignore`, so a clone of the Git repository will not contain them. They exist only in the project ZIP. The Hard dataset's degraded file is named `degraded_imu.csv`, while Easy and Medium use `degraded.csv`.

## 8. Working Principle

### 8.1 Frames, conventions and input

- Input rows follow the EuRoC order: `timestamp [ns], ωx, ωy, ωz [rad/s], ax, ay, az [m/s²]`. The first line is treated as a header and skipped.
- Quaternions are stored as `[w, x, y, z]`. Gravity is `(0, 0, −9.80665)` in the world frame.
- The estimator starts at position `(0, 0, 0)` with zero velocity. **The output is therefore in a local frame**, not the ground-truth world frame, and any scoring has to align trajectories first. `evaluate.py` does this with SE(3) alignment, without scale correction.
- There is no resampling or interpolation. Rows are processed as received.

### 8.2 Pass 1: static calibration (`calibrate_from_head`)

1. Read the first 300 *valid* rows (parsable, finite, not all zero).
2. Slide a 50-sample window over them and choose the window that minimises `accel_variance + 15 × gyro_variance`. This finds the stillest stretch. If fewer than 10 samples are available, calibration is skipped.
3. In that window:
   - gyro bias `bg` = mean gyro;
   - initial roll and pitch come from the mean accelerometer direction (yaw is set to 0);
   - accel bias `ba` = mean accel − the gravity vector expected in the body frame.

### 8.3 Pass 2: streaming estimation

For each row, `dt` is the timestamp difference to the previous row in seconds. The row is a **blackout** if it is invalid (NaN, inf, unparsable, or all six channels zero) **or** `dt > 0.05 s`. If `dt > 0`, `step()` is called. Then one pose line is written. A row with `dt ≤ 0` (a duplicate timestamp) skips the update but **still produces an output row**, so duplicate timestamps can appear in the output.

### 8.4 Normal propagation (`step`, not blackout)

`dt` is split into sub-steps of at most 5 ms. For every sub-step:

| # | Stage | Detail |
|---|---|---|
| 1 | Bias correction | `ω = ω_meas − bg`, `a = a_meas − ba` |
| 2 | Attitude | Midpoint integration: the mean of the previous and current ω, through the rotation-vector exponential map, then renormalise |
| 3 | Gravity tilt correction | Corrects roll and pitch only, with gain 0.002, **only when** `abs(‖a‖ − g) < 0.15` and `‖ω‖ < 0.25` |
| 4 | World acceleration | Rotate `a` into the world frame and add gravity. Subtract a slow exponential moving average (τ ≈ 10 s) of that residual, which acts as a world-frame high-pass |
| 5 | Stationary detection (ZUPT-style) | Stationary when the 20-sample accel standard deviation is < 0.10, `‖ω‖ < 0.03`, and `abs(‖a‖ − g) < 0.20`. Then velocity and acceleration are set to zero and `bg`, `ba` adapt with an EMA of 0.005 |
| 6 | Velocity damping | When not stationary, `v ← v · exp(−0.55 · dt)` every sub-step. This is a heuristic leak, not a physical model |
| 7 | Integration | `p += v·dt + ½·a·dt²`, then `v += a·dt` |

### 8.5 Blackout and gaps (`step`, blackout branch)

There is no learned or physical motion model. Per sub-step, the last world acceleration decays as `e^(−4t)`, the angular rate as `e^(−3.5t)`, and the velocity as `e^(−1.5t)`, plus the decaying acceleration. Position integrates the decaying velocity, and attitude keeps rotating with the decaying rate. This is **momentum-decay extrapolation**. The same code path serves mid-flight gaps (> 50 ms), invalid frames and the final 5 s failure.

Short drops (≤ 50 ms) are not treated as blackout; they are bridged by normal integration over the longer `dt`. The sample that arrives after a long gap is itself not used for that interval.

### 8.6 Output

Each input row produces one CSV row immediately. After the last row the script prints the pose count and the mean per-row latency.

## 9. Installation and Setup

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (the project's documented environment manager, per `src/README.md`)

### Estimator only (no installation)

`src/main.py` uses only the standard library. With any Python 3 interpreter, no environment is needed:

```bash
python src/main.py --help
```

It was run successfully on Python 3.13.18. The project as a whole declares Python ≥ 3.13, and the estimator does not depend on that version by itself.

### Full environment (evaluation, profiling, degradation, simulation stub)

```bash
uv sync
```

This installs the dependencies listed in `pyproject.toml`: `evo`, `jsbsim`, `matplotlib`, `numpy`, `pandas`, `psutil`. The command is the one documented in `src/README.md`. It could not be executed while this README was being verified (no network access), so a full `uv sync`, including console-script installation, is **unverified**.

If you prefer not to use the console scripts, every tool below can be run as a plain script (`python src/...`) once the dependencies are installed.

## 10. Execution Instructions

Run every command from the **repository root** (the folder that contains `src/` and `data/`). `perf.py` launches `src/main.py` by that relative path.

### 10.1 Run the estimator

```bash
python src/main.py <input_imu.csv> <output_estimate.csv>
```

Example:

```bash
python src/main.py data/Medium/degraded.csv out/medium_estimate.csv
# Saved 22880 poses to 'out/medium_estimate.csv' (avg latency: 0.0248 ms)
```

The output folder is created automatically if it does not exist. If the input is missing, a `FileNotFoundError` is raised. For the Hard dataset use `data/Hard/degraded_imu.csv`.

Alternative forms that produce byte-identical output, verified on Medium:

```bash
python -m src.main data/Medium/degraded.csv out/medium_estimate.csv     # from the repo root
cd src && python -m main ../data/Medium/degraded.csv ../out/medium_estimate.csv
```

> **Do not use the `estimate` console script yet.** `pyproject.toml` maps `estimate = "main:main"`, but `src/main.py` defines no `main()` function. That entry point would fail. The older `src/README.md` lists `estimate [input] [output]`, which is wrong for the same reason. See [section 14](#14-challenges-and-limitations).

### 10.2 Evaluate a trajectory

`evaluate.py` converts both CSVs to TUM format, runs `evo` APE (SE(3) alignment, no scale correction) and RPE, writes `metrics.json` and `diagnostics.npz`, and saves 8 plots. **Both timestamp scales must be passed explicitly**, because the CSV timestamps are in nanoseconds and the tool's defaults assume seconds:

```bash
python src/icbm/evaluate.py \
    --ground-truth data/Medium/ground_truth.csv \
    --estimated    out/medium_estimate.csv \
    --output       out/eval_medium \
    --gt-time-scale 1e-9 \
    --est-time-scale 1e-9 \
    --max-time-diff 0.0025
```

| Option | Default | Meaning |
|---|---|---|
| `--ground-truth` | `ground_truth.csv` | Ground-truth CSV (EuRoC format) |
| `--estimated` | `estimated_trajectory.csv` | Estimator output CSV |
| `--output` | `evaluation` | Output folder |
| `--gt-time-scale`, `--est-time-scale` | `1.0` | Multiplier applied to timestamps. Use `1e-9` for these files |
| `--max-time-diff` | `0.01` | Maximum timestamp mismatch for association, in seconds |
| `--dropout-seconds` | `5.0` | Length of the final blind window |
| `--rpe-delta` | `1` | RPE interval **in frames**. The `--help` text says "seconds", but the code uses frames |
| `--plots` | `none` | `all` also opens the figures interactively. Plots are saved in both cases |

Status of this command: the CSV→TUM conversion was run successfully on the Medium files. The `evo` stages could not be run while verifying this README because `evo` could not be installed offline, so the full command is **unverified end to end**. Without the two time-scale flags the association step and the dropout window are meaningless (see [section 13.4](#134-stored-evaluation-artifacts-do-not-quote)).

### 10.3 Profile memory and runtime

```bash
python src/icbm/perf.py data/Medium/degraded.csv out/medium_estimate.csv --no-graph
```

It launches `src/main.py` as a subprocess, polls the RSS of the process and its children every `--interval` ms (default 100), and logs the exit code, elapsed time and peak RAM. Without `--no-graph` it opens a RAM-over-time plot with `plt.show()`, which needs a display. Verified to run.

### 10.4 Generate a degraded IMU file (organizer tool)

```bash
python src/ext/degrade.py data/Hard/imu.csv out/hard_degraded.csv --meta out/hard_meta.json
```

The seed is fixed (`SEED = 42`), so the output is deterministic. Optional flags: `--max-level`, `--gamma`, `--ts-unit`, `--accel-unit`, `--gyro-deg`. See also the [note on column order](#14-challenges-and-limitations).

### 10.5 JSBSim stub

```bash
cd src && python -m icbm.simulate list     # lists JSBSim aircraft and scripts (needs jsbsim)
cd src && python -m icbm.simulate run x    # prints "STUB: TODO"
```

Not executed during verification (`jsbsim` not installed). The behaviour is read from the code.

## 11. Code Explanation

### `src/main.py`: the estimator

| Symbol | Responsibility |
|---|---|
| `quat_multiply`, `quat_normalize`, `rotvec_to_quat`, `quat_rotate`, `quat_rotate_inv` | Hand-written quaternion math (Hamilton product, normalisation, rotation-vector exponential map, rotate and inverse-rotate) |
| `LowMemoryInertialEstimator` | State: position `p`, velocity `v`, attitude `q`, biases `bg`/`ba`, the world-acceleration bias, last acceleration and rate for the blackout model, and two 20-entry rolling buffers for stationary detection |
| `.calibrate_from_head(samples)` | Static calibration (section 8.2) |
| `.step(dt, omega_m, acc_m, is_blackout)` | One time step, split into ≤ 5 ms sub-steps (sections 8.4 and 8.5) |
| `is_invalid_sample(omega, acc)` | True for NaN/inf or for an all-zero frame, which is how the organizers' failure frames appear |
| `run_trajectory_estimation(in, out)` | The two-pass driver: read, validate, step, write, time |
| `__main__` block | `argparse` CLI with two positional arguments, `input_imu` and `output_est` |

The class docstring claims "< 0.02 ms per sample". That is not matched by measurement (see [section 13.2](#132-efficiency)).

### `src/icbm/evaluate.py`: evaluation

`find_columns`/`ALIASES` map column names case-insensitively, including the EuRoC names. `convert_to_tum` normalises quaternions, sorts by time and writes `t tx ty tz qx qy qz qw` (reordering the quaternion from `w`-first). `run_evo` associates timestamps, runs APE with SE(3) alignment, derives a "dropout" interval from the last `--dropout-seconds` of the associated trajectory, runs translational and rotational RPE, and writes `metrics.json` and `diagnostics.npz`. `plot_results` saves eight PNGs: `trajectory_3d`, `trajectory_xy`, `position_error`, `position_components`, `rpe_translation`, `rpe_rotation`, `error_distribution`, `dropout_error_growth`.

### `src/icbm/perf.py`: profiler

`benchmark` starts `sys.executable src/main.py <in> <out>`, samples summed RSS (process plus children) via `psutil`, and returns exit code, elapsed time, the RSS series and the peak. `plot` draws the series with matplotlib. The measured figure is whole-process RSS, which includes the Python interpreter itself.

### `src/icbm/simulate.py`

A command-line skeleton for JSBSim. `list` prints the JSBSim data directory, aircraft and flight scripts. `run` is a **stub** that only prints `STUB: TODO`. The parser's program name is still the placeholder `mytool`. It is not used by anything else in the project.

### `src/ext/degrade.py`

Organizer-provided (stated in the file). It ramps a severity from 0.02 to 1.0 with `severity = floor + (max − floor) · (t/T)^γ`, with γ = 1.7. It then adds white noise, bias random walks and scale-factor random walks, and injects spikes, stuck samples, single drops, burst gaps (5–50 samples) and duplicated rows. The first 2 s are protected, and the last 5 s are zeroed as a total failure.

## 12. Datasets

Each dataset folder holds the clean IMU log, a degraded copy, and the ground truth. All logs are EuRoC-style CSVs at 200 Hz in the clean files.

| Dataset | Clean IMU rows | Degraded rows | Ground-truth rows | Degraded file |
|---|---:|---:|---:|---|
| Easy | 22,800 | 22,162 | 22,401 | `data/Easy/degraded.csv` |
| Medium | 23,490 | 22,880 | 23,091 | `data/Medium/degraded.csv` |
| Hard | 23,370 | 22,623 | 22,970 | `data/Hard/degraded_imu.csv` |

Counts are data rows excluding the header. Each estimator output has exactly as many rows as its degraded input.

**Hard dataset degradation statistics** (`data/Hard/meta.json`, seed 42): duration 116.8 s, severity 0.02 → 1.0, 25 spikes, 273 stuck samples, 460 single drops, 16 burst gaps (775 dropped rows in total), 28 duplicated rows. `meta.json` reports 1,001 failure frames, while the degraded file contains 1,004 all-zero rows; the 3-row difference was not investigated.

**Reproducibility.** Running `src/ext/degrade.py` with default settings on each `imu.csv` reproduced the corresponding shipped degraded file with identical values. For example Hard gives 22,623 rows and a `meta.json` matching the stored one. The three datasets therefore appear to differ by **recording segment**, not by degradation settings. The labels "Easy/Medium/Hard" have no documented meaning in the repository.

## 13. Results and Outputs

### 13.1 Output format of `main.py`

An 11-column CSV with a header, one row per input row:

```
#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1]
```

Timestamps are integer nanoseconds. Positions, quaternion (**w first**) and velocities are written with 9 decimals. **This is not the TUM format** (8 columns, seconds, `qx qy qz qw`). Conversion exists only inside `evaluate.py`.

### 13.2 Efficiency

Measured in the verification environment (one Linux machine, Python 3.13.18). Absolute values depend on the machine and the interpreter build.

| Dataset | Rows | Wall time | Mean per-row latency printed by `main.py` |
|---|---:|---:|---:|
| Easy | 22,162 | 0.73 s | 0.029 ms |
| Medium | 22,880 | 0.76 s | 0.030 ms |
| Hard | 22,623 | 0.70 s | 0.028 ms |

- The latency includes CSV parsing and writing. At 200 Hz it implies roughly 5–6 ms of compute per 1 s of IMU data, compared with the 20 ms limit.†
- **Peak RSS** of the whole process was 15.1 MB, against 13.3 MB for a bare, idle Python interpreter on the same machine. The estimator itself therefore adds about 1.8 MB. Because a bare CPython interpreter already exceeds 8 MB here, **the 8 MB budget cannot be demonstrated by RSS in CPython on this machine**. How the organizers measure memory is unknown.
- `perf.py` reported a peak of 14.94 MB for Medium (100 ms polling).

### 13.3 Accuracy (independent check of the current code)

The figures below come from a separate NumPy check of the *current* `main.py` output: nearest-timestamp association within 2.5 ms, then SE(3) Umeyama alignment without scale. It is **not** the organizers' scorer and **not** the `evo` pipeline in `evaluate.py`. Treat the numbers as indicative.

| Dataset | Associated poses | Aligned APE RMSE | Max error | Behaviour |
|---|---:|---:|---:|---|
| Easy | 21,733 | 51.0 m | 130 m | Diverges (max‖p‖ ≈ 161 m); the cause is not confirmed (see §14) |
| Medium | 22,453 | 4.23 m | 10.5 m | Bounded (max‖p‖ ≈ 14 m) |
| Hard | 22,195 | 3.55 m | 8.2 m | Bounded (max‖p‖ ≈ 11 m) |

**No separate accuracy figure was computed for the blind 5 s window.** On Medium, the estimate moves only about 0.3 m along its path in the final 5 s while the velocity decays toward zero. Ground truth covers only about 4.25 s of that window.

### 13.4 Stored evaluation artifacts: do not quote

The `evaluation/` folder in the ZIP holds the output of one earlier run. It is internally inconsistent and does not match the current code. Regenerate it before citing anything from it.

- `metrics.json` reports a full-trajectory APE RMSE of 10.76 m, but its dropout statistics are degenerate (min = max = mean = 22.94 m, a single pose). Its `duration_seconds` is about 1.15 × 10¹¹, which shows the timestamps were never scaled from nanoseconds.
- `ape_results.zip` (a separate `evo` run) reports a different RMSE of 2.30 m, over a different pose count.
- `estimated.tum` holds 23,881 poses, whereas the current estimator output for Medium has 22,880. It was produced by some other version or input.

### 13.5 Plots

`evaluate.py` produces eight diagnostic plots (listed in section 11). The copies stored in `evaluation/plots/` belong to the stale run above, so they are not reproduced here.

## 14. Challenges and Limitations

| # | Limitation | Evidence |
|---|---|---|
| 1 | **Output is not in the required format.** `main.py` writes an 11-column nanosecond CSV with a w-first quaternion, not an 8-column TUM file in seconds. | `run_trajectory_estimation`; no TUM writer in `main.py` |
| 2 | **`estimate` console script is broken.** `pyproject.toml` points to `main:main`, which does not exist. Whether `uv_build` packages the top-level `main` and `ext` modules at all is also unverified. | `main.py` defines no `main()`; checked by import |
| 3 | **Easy dataset diverges** (aligned APE ≈ 51 m). See the observation below. | §13.3 |
| 4 | **8 MB RAM budget not demonstrated.** Process RSS is above 8 MB even for an idle interpreter. | §13.2 |
| 5 | **No outlier handling.** Spikes and stuck values go straight into the integration. Only the stationary gating limits their effect. | no such code in `main.py` |
| 6 | **Blackout model is a heuristic.** Exponential decay has no physical basis. No accuracy figure exists for the final 5 s. | `step` |
| 7 | **Heuristic tuning.** Thresholds and gains are hard-coded constants with no documented derivation or tuning procedure. | `main.py` |
| 8 | **`evaluate.py` pitfalls.** Default time scales assume seconds. The `--rpe-delta` help text says seconds but the code uses frames. | `evaluate.py` |
| 9 | **Stored results are stale or inconsistent.** | §13.4 |
| 10 | **Local frame.** Poses start at the origin, so any score depends on whether the scorer aligns trajectories. | §8.1 |
| 11 | **Missing deliverables** in the repository: `metadata.json`, the walkthrough video, and the problem-statement PDF. `data/Hard/estimated.csv` is a header-only placeholder. | file listing |
| 12 | **`simulate.py` is a stub** with a leftover `mytool` placeholder. | §11 |
| 13 | **Docstring claim not matched.** "< 0.02 ms per sample" versus about 0.025–0.030 ms measured. | §13.2 |
| 14 | **Project metadata is a placeholder.** `pyproject.toml` still reads "Add your description here", and the fourth team member is not listed in its authors. | `pyproject.toml` |

**Observation on `degrade.py` column order (not an official finding).** The script's docstring and code assume the column order `timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z`, whereas the EuRoC-style files in `data/` are gyro first, accelerometer second. The script's unit auto-detection looks at "columns 1–3", which in these files are the gyro channels. It picked `g` for Medium and Hard and `m/s^2` for Easy, so the injected white-noise level differs between datasets. In the shipped files, the injected error on the gyro channels over roughly the 70–90 % portion of each record has a standard deviation of about 0.23 rad/s for Easy against about 0.03 rad/s for Medium and Hard (accelerometer noise is comparable across the three). This is a plausible explanation for Easy's divergence, but it has **not** been tested by any experiment in this repository. It also means that running `degrade.py` on gyro-first files may not behave as its docstring describes.

## 15. Future Improvements

None of the following exists in the code today.

- Write `estimated_trajectory.txt` (TUM, seconds, `qx qy qz qw`) directly from `main.py`, add `metadata.json`, and fix the `estimate` entry point.
- Add spike and stuck-value gating, and stronger robustness to heavy gyro noise (the likely cause of the Easy divergence; to be investigated).
- Replace the heuristic stabilisers with an error-state Kalman filter with bias states.
- Use a better blackout model, for example constant-turn-rate extrapolation, and measure error on the final 5 s separately.
- Tune parameters per degradation level.
- Reduce the memory footprint further (for example with the `array` module), and settle how memory is measured before optimising it.
- Add unit tests and a one-command reproducible evaluation script with the correct time-scale flags.
- Complete or remove the JSBSim stub.

## 16. References and Provenance

- **Organizer-provided code:** `src/ext/degrade.py` (marked as such in the file). It is not authored by the project team.
- **Datasets:** EuRoC-format IMU and ground-truth CSVs in `data/`. The repository does not state their original source.
- **Tooling:** [`evo`](https://github.com/MichaelGrupp/evo) for APE/RPE, [`psutil`](https://github.com/giampaolo/psutil) for RSS sampling, [`uv`](https://docs.astral.sh/uv/) for environment management.
- **Hackathon problem statement:** not included in the repository, so the specification in section 2 could not be re-verified from the files.
- **Authors listed in `pyproject.toml`:** Pragadheesh A, Darshan V, Arthi Y, Sarveshwaran