import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.metrics import RPE, PoseRelation, Unit
from evo.main_ape import ape as evo_ape
from evo.tools import file_interface

# Supported CSV column names, including EuRoC ground-truth names.
ALIASES = {
    "timestamp": ["timestamp", "time", "#timestamp"],
    "tx": ["tx", "x", "p_rs_r_x", "p_rs_r_x [m]", "position_x"],
    "ty": ["ty", "y", "p_rs_r_y", "p_rs_r_y [m]", "position_y"],
    "tz": ["tz", "z", "p_rs_r_z", "p_rs_r_z [m]", "position_z"],
    "qx": ["qx", "q_x", "q_rs_x", "q_rs_x []"],
    "qy": ["qy", "q_y", "q_rs_y", "q_rs_y []"],
    "qz": ["qz", "q_z", "q_rs_z", "q_rs_z []"],
    "qw": ["qw", "q_w", "q_rs_w", "q_rs_w []"],
}


def find_columns(fieldnames):
    normalized = {
        name.strip().lower(): name for name in fieldnames
    }

    columns = {}

    for target, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                columns[target] = normalized[alias]
                break

        if target not in columns:
            raise ValueError(
                f"Missing column '{target}'. "
                f"Available columns: {fieldnames}"
            )

    return columns

def convert_to_tum(input_path, output_tum, timestamp_scale=1.0, allow_duplicates=False):
    """
    Convert a CSV trajectory to TUM trajectory format, or normalize/scale
    an existing TUM trajectory file.
    """
    with open(input_path, newline="") as file:
        first_line = file.readline()
        if not first_line.strip():
            raise ValueError(f"Empty file: {input_path}")

        is_tum = "," not in first_line
        file.seek(0)
        poses = []

        if is_tum:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                parts = stripped.split()
                if len(parts) < 8:
                    raise ValueError(
                        f"Invalid TUM line at {input_path}:{line_number}"
                    )

                try:
                    timestamp = float(parts[0]) * timestamp_scale
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid float at {input_path}:{line_number}"
                    ) from exc

                values = [timestamp, tx, ty, tz, qx, qy, qz, qw]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(
                        f"Non-finite value at {input_path}:{line_number}"
                    )

                norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                if norm < 1e-12:
                    raise ValueError(
                        f"Zero quaternion at {input_path}:{line_number}"
                    )

                poses.append(
                    (timestamp, tx, ty, tz, qx / norm, qy / norm, qz / norm, qw / norm)
                )
        else:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"No CSV header found: {input_path}")

            columns = find_columns(reader.fieldnames)

            for line_number, row in enumerate(reader, start=2):
                try:
                    timestamp = float(row[columns["timestamp"]]) * timestamp_scale
                    tx = float(row[columns["tx"]])
                    ty = float(row[columns["ty"]])
                    tz = float(row[columns["tz"]])
                    qx = float(row[columns["qx"]])
                    qy = float(row[columns["qy"]])
                    qz = float(row[columns["qz"]])
                    qw = float(row[columns["qw"]])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid value at {input_path}:{line_number}"
                    ) from exc

                values = [timestamp, tx, ty, tz, qx, qy, qz, qw]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(
                        f"Non-finite value at {input_path}:{line_number}"
                    )

                norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                if norm < 1e-12:
                    raise ValueError(
                        f"Zero quaternion at {input_path}:{line_number}"
                    )

                poses.append(
                    (timestamp, tx, ty, tz, qx / norm, qy / norm, qz / norm, qw / norm)
                )

    if len(poses) < 2:
        raise ValueError(f"Not enough poses in {input_path} (found {len(poses)})")

    poses.sort(key=lambda pose: pose[0])

    if allow_duplicates:
        deduped_poses = []
        for pose in poses:
            if deduped_poses and math.isclose(
                pose[0], deduped_poses[-1][0], rel_tol=0.0, abs_tol=1e-12
            ):
                deduped_poses[-1] = pose
            else:
                deduped_poses.append(pose)

        if len(deduped_poses) < len(poses):
            print(
                f"[INFO] Removed {len(poses) - len(deduped_poses)} "
                f"duplicate timestamps from {input_path}"
            )
        poses = deduped_poses

    if len(poses) < 2:
        sample_ts = [p[0] for p in poses[:5]]
        raise ValueError(
            f"Not enough unique poses in {input_path} (remaining: {len(poses)}, "
            f"sample timestamps: {sample_ts})"
        )

    with open(output_tum, "w") as file:
        file.writelines(" ".join(f"{v:.12f}" for v in pose) + "\n" for pose in poses)

    mode = "Normalized TUM" if is_tum else "Converted CSV"
    print(f"{mode}: {input_path} -> {output_tum}")
    print(f"Number of poses: {len(poses)}")

    return len(poses)


def print_statistics(name, values, unit):
    values = np.asarray(values)

    if values.size == 0:
        print(f"\n{name}: No samples available.")
        return

    print(f"\n{name}")
    print("-" * 50)
    print(f"Samples:  {len(values)}")
    print(f"Min:      {np.min(values):.6f} {unit}")
    print(f"Max:      {np.max(values):.6f} {unit}")
    print(f"Mean:     {np.mean(values):.6f} {unit}")
    print(f"Median:   {np.median(values):.6f} {unit}")
    print(f"Std:      {np.std(values):.6f} {unit}")
    print(f"RMSE:     {np.sqrt(np.mean(values**2)):.6f} {unit}")


def plot_results(
    output_dir,
    timestamps,
    gt_positions,
    est_positions,
    ape_errors,
    rpe_timestamps,
    rpe_translation,
    rpe_rotation,
    dropout_seconds=5.0,
    show=False,
):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    time = timestamps - timestamps[0]
    rpe_time = rpe_timestamps - timestamps[0]

    dropout_start = timestamps[-1] - dropout_seconds
    dropout_start_relative = dropout_start - timestamps[0]

    dropout_mask = timestamps >= dropout_start

    position_components = est_positions - gt_positions

    def mark_dropout(ax):
        ax.axvspan(
            dropout_start_relative,
            time[-1],
            color="red",
            alpha=0.12,
            label=f"Blind {dropout_seconds:g}s dropout",
        )

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(
            plot_dir / filename,
            dpi=180,
            bbox_inches="tight",
        )
        figures.append(fig)

    # --------------------------------------------------
    # 1. 3D trajectory
    # --------------------------------------------------

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(*gt_positions.T, label="Ground truth")
    ax.plot(*est_positions.T, label="Estimated")
    ax.scatter(
        *est_positions[dropout_mask].T,
        label="Dropout prediction",
        s=12,
    )

    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    save(fig, "trajectory_3d.png")

    # --------------------------------------------------
    # 2. XY trajectory
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(gt_positions[:, 0], gt_positions[:, 1], label="Ground truth")
    ax.plot(est_positions[:, 0], est_positions[:, 1], label="Estimated")
    ax.scatter(
        est_positions[dropout_mask, 0],
        est_positions[dropout_mask, 1],
        label="Dropout prediction",
        s=12,
    )

    ax.set_title("XY Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    save(fig, "trajectory_xy.png")

    # --------------------------------------------------
    # 3. Absolute position error
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time, ape_errors, label="Absolute position error")
    mark_dropout(ax)

    ax.set_title(
        f"Absolute Position Error | RMSE: "
        f"{np.sqrt(np.mean(ape_errors**2)):.4f} m"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True)
    ax.legend()

    save(fig, "position_error.png")

    # --------------------------------------------------
    # 4. X/Y/Z error components
    # --------------------------------------------------

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for i, axis_name in enumerate(("X", "Y", "Z")):
        axes[i].plot(time, position_components[:, i])
        mark_dropout(axes[i])
        axes[i].set_ylabel(f"{axis_name} error (m)")
        axes[i].grid(True)

    axes[0].set_title("Position Error Components")
    axes[-1].set_xlabel("Time (s)")

    save(fig, "position_components.png")

    # --------------------------------------------------
    # 5. Translational RPE
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(rpe_time) > 0:
        ax.plot(rpe_time, rpe_translation, label="Translational RPE")
    mark_dropout(ax)

    ax.set_title("Translational Relative Pose Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True)
    ax.legend()

    save(fig, "rpe_translation.png")

    # --------------------------------------------------
    # 6. Rotational RPE
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(rpe_time) > 0:
        ax.plot(rpe_time, rpe_rotation, label="Rotational RPE")
    mark_dropout(ax)

    ax.set_title("Rotational Relative Pose Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (degrees)")
    ax.grid(True)
    ax.legend()

    save(fig, "rpe_rotation.png")

    # --------------------------------------------------
    # 7. Error distribution
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(ape_errors, bins=40, edgecolor="black")
    axes[0].set_title("Position Error Distribution")
    axes[0].set_xlabel("Error (m)")
    axes[0].set_ylabel("Frequency")

    sorted_errors = np.sort(ape_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    axes[1].plot(sorted_errors, cdf)
    axes[1].set_title("Position Error CDF")
    axes[1].set_xlabel("Error (m)")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].grid(True)

    save(fig, "error_distribution.png")

    # --------------------------------------------------
    # 8. Dropout error growth
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    dropout_elapsed = time[dropout_mask] - dropout_start_relative

    ax.plot(
        dropout_elapsed,
        ape_errors[dropout_mask],
        marker=".",
        label="Position error",
    )

    ax.set_title("Position Error Growth During Blind Prediction")
    ax.set_xlabel("Time since dropout began (s)")
    ax.set_ylabel("Absolute position error (m)")
    ax.grid(True)
    ax.legend()

    save(fig, "dropout_error_growth.png")

    print(f"\nSaved {len(figures)} diagnostic plots to {plot_dir}")

    if show:
        plt.show()

    for fig in figures:
        plt.close(fig)


def run_evo(
    gt_file,
    estimated_file,
    output_dir,
    max_time_diff,
    dropout_seconds=5.0,
    rpe_delta=1,
    max_rpe_gap=0.5,
    show_plots=False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("TRAJECTORY EVALUATION")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Load trajectories using evo
    # --------------------------------------------------

    traj_ref = file_interface.read_tum_trajectory_file(str(gt_file))
    traj_est = file_interface.read_tum_trajectory_file(str(estimated_file))

    print(f"Ground truth poses: {traj_ref.num_poses}")
    print(f"Estimated poses:    {traj_est.num_poses}")

    # Auto-normalize timestamp units if one is in ns/us/ms and the other is in seconds
    def _normalize_to_seconds(traj, label):
        t0 = abs(traj.timestamps[0])
        if t0 > 1e16:
            print(f"[INFO] Scaling {label} timestamps from nanoseconds to seconds (1e-9)")
            traj.timestamps *= 1e-9
        elif t0 > 1e13:
            print(f"[INFO] Scaling {label} timestamps from microseconds to seconds (1e-6)")
            traj.timestamps *= 1e-6
        elif t0 > 1e10:
            print(f"[INFO] Scaling {label} timestamps from milliseconds to seconds (1e-3)")
            traj.timestamps *= 1e-3

    _normalize_to_seconds(traj_ref, "ground truth")
    _normalize_to_seconds(traj_est, "estimated")

    # Safety deduplication after scaling
    def _deduplicate_trajectory(traj, label):
        ts = traj.timestamps
        if len(ts) < 2:
            return traj
        order = np.argsort(ts, kind="mergesort")
        ts_sorted = ts[order]
        keep_mask = np.append(np.diff(ts_sorted) > 1e-9, True)
        kept_indices = order[keep_mask]
        if len(kept_indices) < len(ts):
            print(
                f"[INFO] Removed {len(ts) - len(kept_indices)} "
                f"duplicate timestamps from {label}"
            )
            traj.reduce_to_ids(kept_indices)
        return traj

    traj_ref = _deduplicate_trajectory(traj_ref, "ground truth")
    traj_est = _deduplicate_trajectory(traj_est, "estimated")

    print(
        f"GT time range:  [{traj_ref.timestamps[0]:.6f}, {traj_ref.timestamps[-1]:.6f}] "
        f"(duration: {traj_ref.timestamps[-1] - traj_ref.timestamps[0]:.3f} s)"
    )
    print(
        f"Est time range: [{traj_est.timestamps[0]:.6f}, {traj_est.timestamps[-1]:.6f}] "
        f"(duration: {traj_est.timestamps[-1] - traj_est.timestamps[0]:.3f} s)"
    )

    # --------------------------------------------------
    # 2. Timestamp association using evo (skips missing poses/dropouts)
    # --------------------------------------------------

    try:
        traj_ref, traj_est = sync.associate_trajectories(
            traj_ref,
            traj_est,
            max_diff=max_time_diff,
        )
    except sync.SyncException as exc:
        offset = traj_ref.timestamps[0] - traj_est.timestamps[0]
        raise sync.SyncException(
            f"{exc}\n"
            f"Start timestamp difference (GT - Est): {offset:.6f} s. "
            f"Check whether one trajectory uses relative time (starting at 0.0) "
            f"while the other uses epoch time, or increase --max-time-diff."
        ) from exc

    if traj_ref.num_poses < 2:
        raise ValueError("Insufficient associated trajectory poses.")

    timestamps = traj_ref.timestamps.copy()

    print(f"Associated poses:   {traj_ref.num_poses}")
    print(f"Association limit:  {max_time_diff:.6f} s")

    # Log any mid-trajectory dropout gaps in associated poses
    dt = np.diff(timestamps)
    gap_indices = np.where(dt > max_rpe_gap)[0]
    for idx in gap_indices:
        print(
            f"[INFO] Dropout gap detected: {timestamps[idx]:.3f} s -> "
            f"{timestamps[idx + 1]:.3f} s (duration: {dt[idx]:.3f} s)"
        )

    # --------------------------------------------------
    # 3. APE using evo
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("ABSOLUTE POSE ERROR (APE)")
    print("=" * 65)

    ape_result = evo_ape(
        traj_ref,
        traj_est,
        PoseRelation.translation_part,
        align=True,
        correct_scale=False,
    )

    ape_errors = ape_result.np_arrays["error_array"]

    gt_positions = traj_ref.positions_xyz.copy()
    est_positions = traj_est.positions_xyz.copy()

    print("\nFull trajectory:")
    print(ape_result.pretty_str())

    # --------------------------------------------------
    # 4. Identify dropout interval
    # --------------------------------------------------

    dropout_start = timestamps[-1] - dropout_seconds

    dropout_mask = timestamps >= dropout_start
    pre_dropout_mask = timestamps < dropout_start

    if not np.any(dropout_mask):
        raise ValueError(
            "No associated poses found in the dropout interval."
        )

    if not np.any(pre_dropout_mask):
        raise ValueError(
            "No associated poses found before dropout."
        )

    print("\nDropout interval:")
    print(f"Start: {dropout_start:.6f}")
    print(f"End:   {timestamps[-1]:.6f}")
    print(f"Duration: {timestamps[-1] - dropout_start:.3f} s")

    print_statistics("Pre-dropout APE", ape_errors[pre_dropout_mask], "m")
    print_statistics("Blind dropout APE", ape_errors[dropout_mask], "m")
    print_statistics("Full trajectory APE", ape_errors, "m")

    # --------------------------------------------------
    # 5. RPE using evo (filtering out pairs spanning dropout gaps)
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("RELATIVE POSE ERROR (RPE)")
    print("=" * 65)

    if not isinstance(rpe_delta, int) or rpe_delta < 1:
        raise ValueError(
            "rpe_delta must be a positive integer when using frames."
        )

    rpe_translation_metric = RPE(
        pose_relation=PoseRelation.translation_part,
        delta=rpe_delta,
        delta_unit=Unit.frames,
    )
    rpe_translation_metric.process_data((traj_ref, traj_est))

    rpe_rotation_metric = RPE(
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=rpe_delta,
        delta_unit=Unit.frames,
    )
    rpe_rotation_metric.process_data((traj_ref, traj_est))

    delta_ids = np.asarray(rpe_translation_metric.delta_ids)
    rpe_start_indices = delta_ids - rpe_delta
    rpe_dt = timestamps[delta_ids] - timestamps[rpe_start_indices]

    # Exclude relative pairs that jump across a sensor/estimation dropout gap
    valid_rpe_mask = rpe_dt <= max_rpe_gap
    skipped_rpe = int(np.sum(~valid_rpe_mask))
    if skipped_rpe > 0:
        print(
            f"[INFO] Excluded {skipped_rpe} RPE pairs spanning gaps > {max_rpe_gap:.3f} s"
        )

    rpe_translation = rpe_translation_metric.error[valid_rpe_mask].copy()
    rpe_rotation = rpe_rotation_metric.error[valid_rpe_mask].copy()
    rpe_timestamps = timestamps[delta_ids][valid_rpe_mask]
    valid_start_indices = rpe_start_indices[valid_rpe_mask]

    print(f"RPE interval: {rpe_delta} frames")

    print_statistics("Translational RPE", rpe_translation, "m")
    print_statistics("Rotational RPE", rpe_rotation, "deg")

    # --------------------------------------------------
    # 6. Dropout-specific RPE
    # --------------------------------------------------

    rpe_dropout_mask = (
        (timestamps[valid_start_indices] >= dropout_start)
        & (rpe_timestamps >= dropout_start)
    )

    if np.any(rpe_dropout_mask):
        print_statistics(
            "Dropout translational RPE",
            rpe_translation[rpe_dropout_mask],
            "m",
        )
        print_statistics(
            "Dropout rotational RPE",
            rpe_rotation[rpe_dropout_mask],
            "deg",
        )
    else:
        print(
            "\nInsufficient RPE pairs entirely inside the dropout interval."
        )

    # --------------------------------------------------
    # 7. Save metric results
    # --------------------------------------------------

    def _array_stats(arr):
        if len(arr) == 0:
            return {}
        return {
            "rmse": float(np.sqrt(np.mean(arr**2))),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "samples": int(len(arr)),
        }

    metrics_output = {
        "trajectory": {
            "associated_poses": int(traj_ref.num_poses),
            "duration_seconds": float(timestamps[-1] - timestamps[0]),
            "dropout_seconds": float(dropout_seconds),
            "rpe_delta_frames": int(rpe_delta),
            "max_rpe_gap_seconds": float(max_rpe_gap),
        },
        "ape": {
            "full": ape_result.stats,
            "pre_dropout": _array_stats(ape_errors[pre_dropout_mask]),
            "dropout": _array_stats(ape_errors[dropout_mask]),
        },
        "rpe_translation": _array_stats(rpe_translation),
        "rpe_rotation": _array_stats(rpe_rotation),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=4)

    np.savez_compressed(
        output_dir / "diagnostics.npz",
        timestamps=timestamps,
        ape=ape_errors,
        rpe_timestamps=rpe_timestamps,
        rpe_translation=rpe_translation,
        rpe_rotation=rpe_rotation,
    )

    # --------------------------------------------------
    # 8. Generate plots
    # --------------------------------------------------

    plot_results(
        output_dir=output_dir,
        timestamps=timestamps,
        gt_positions=gt_positions,
        est_positions=est_positions,
        ape_errors=ape_errors,
        rpe_timestamps=rpe_timestamps,
        rpe_translation=rpe_translation,
        rpe_rotation=rpe_rotation,
        dropout_seconds=dropout_seconds,
        show=show_plots,
    )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Metrics:      {output_dir / 'metrics.json'}")
    print(f"Diagnostics:  {output_dir / 'diagnostics.npz'}")
    print(f"Plots:        {output_dir / 'plots'}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate estimated trajectory using evo APE/RPE"
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ground_truth.csv"),
    )

    parser.add_argument(
        "--estimated",
        type=Path,
        default=Path("estimated_trajectory.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation"),
    )

    parser.add_argument(
        "--gt-time-scale",
        type=float,
        default=1.0,
        help="Multiply ground-truth timestamps by this factor",
    )

    parser.add_argument(
        "--est-time-scale",
        type=float,
        default=1.0,
        help="Multiply estimated timestamps by this factor",
    )

    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.01,
        help="Maximum timestamp association difference in seconds",
    )

    parser.add_argument(
        "--dropout-seconds",
        type=float,
        default=5.0,
        help="Duration of the final blind prediction interval",
    )

    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=1,
        help="RPE interval in frames",
    )

    parser.add_argument(
        "--max-rpe-gap",
        type=float,
        default=0.5,
        help="Maximum allowed time gap (seconds) between RPE pose pairs before treating as a dropout",
    )

    parser.add_argument(
        "--plots",
        choices=["none", "all"],
        default="none",
        help="Save plots always; optionally display all plots",
    )

    args = parser.parse_args()

    if not args.ground_truth.exists():
        parser.error(f"Ground truth not found: {args.ground_truth}")

    if not args.estimated.exists():
        parser.error(f"Estimated trajectory not found: {args.estimated}")

    if args.dropout_seconds <= 0:
        parser.error("--dropout-seconds must be positive")

    if args.rpe_delta <= 0:
        parser.error("--rpe-delta must be positive")

    if args.max_time_diff < 0:
        parser.error("--max-time-diff cannot be negative")

    if args.max_rpe_gap <= 0:
        parser.error("--max-rpe-gap must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    gt_tum = args.output / "ground_truth.tum"
    est_tum = args.output / "estimated.tum"

    print("Converting/normalizing trajectories to TUM format...")

    convert_to_tum(
        args.ground_truth,
        gt_tum,
        args.gt_time_scale,
        allow_duplicates=False,
    )

    convert_to_tum(
        args.estimated,
        est_tum,
        args.est_time_scale,
        allow_duplicates=True,
    )
    run_evo(
        gt_file=gt_tum,
        estimated_file=est_tum,
        output_dir=args.output,
        max_time_diff=args.max_time_diff,
        dropout_seconds=args.dropout_seconds,
        rpe_delta=args.rpe_delta,
        max_rpe_gap=args.max_rpe_gap,
        show_plots=args.plots == "all",
    )


if __name__ == "__main__":
    main()