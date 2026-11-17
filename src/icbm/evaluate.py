
import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


# Supported CSV column names, including EuRoC ground-truth names.
ALIASES = {
    "timestamp": ["timestamp", "time", "#timestamp"],
    "tx": ["tx", "x", "p_rs_r_x", "position_x"],
    "ty": ["ty", "y", "p_rs_r_y", "position_y"],
    "tz": ["tz", "z", "p_rs_r_z", "position_z"],
    "qx": ["qx", "q_x", "q_rs_x"],
    "qy": ["qy", "q_y", "q_rs_y"],
    "qz": ["qz", "q_z", "q_rs_z"],
    "qw": ["qw", "q_w", "q_rs_w"],
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


def convert_to_tum(input_csv, output_tum, timestamp_scale=1.0):
    """
    Convert a CSV trajectory to TUM trajectory format.

    TUM format:
    timestamp tx ty tz qx qy qz qw
    """

    poses = []

    with open(input_csv, newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"No CSV header found: {input_csv}")

        columns = find_columns(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = (
                    float(row[columns["timestamp"]])
                    * timestamp_scale
                )

                tx = float(row[columns["tx"]])
                ty = float(row[columns["ty"]])
                tz = float(row[columns["tz"]])

                qx = float(row[columns["qx"]])
                qy = float(row[columns["qy"]])
                qz = float(row[columns["qz"]])
                qw = float(row[columns["qw"]])

            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid value at {input_csv}:{line_number}"
                ) from exc

            values = [
                timestamp, tx, ty, tz,
                qx, qy, qz, qw
            ]

            if not all(math.isfinite(v) for v in values):
                raise ValueError(
                    f"Non-finite value at {input_csv}:{line_number}"
                )

            # Normalize quaternion.
            norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

            if norm < 1e-12:
                raise ValueError(
                    f"Zero quaternion at {input_csv}:{line_number}"
                )

            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm

            poses.append(
                (timestamp, tx, ty, tz, qx, qy, qz, qw)
            )

    if len(poses) < 2:
        raise ValueError(f"Not enough poses in {input_csv}")

    poses.sort(key=lambda pose: pose[0])


    with open(output_tum, "w") as file:
        for pose in poses:
            file.write(
                " ".join(f"{v:.12f}" for v in pose) + "\n"
            )

    print(f"Converted {input_csv} -> {output_tum}")
    print(f"Number of poses: {len(poses)}")

    return len(poses)


def run_evo(gt_file, estimated_file, output_dir, max_time_diff):
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "ape_results.zip"
    plot_file = output_dir / "ape_plot.pdf"

    command = [
        "evo_ape",
        "tum",
        str(gt_file),
        str(estimated_file),
        "--align",
        "--pose_relation",
        "trans_part",
        "--t_max_diff",
        str(max_time_diff),
        "--save_results",
        str(results_file),
        "--save_plot",
        str(plot_file),
        "--plot",
        "--plot_mode",
        "xy",
        "--verbose",
    ]

    print("\nRunning evo APE evaluation:")
    print(" ".join(command))
    print()

    subprocess.run(command, check=True)

    print("\nEvaluation completed.")
    print(f"Results: {results_file}")
    print(f"Plot:    {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate estimated trajectory using evo SE(3) alignment"
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

    args = parser.parse_args()

    if not args.ground_truth.exists():
        parser.error(f"Ground truth not found: {args.ground_truth}")

    if not args.estimated.exists():
        parser.error(f"Estimated trajectory not found: {args.estimated}")

    args.output.mkdir(parents=True, exist_ok=True)

    gt_tum = args.output / "ground_truth.tum"
    est_tum = args.output / "estimated.tum"

    convert_to_tum(
        args.ground_truth,
        gt_tum,
        args.gt_time_scale,
    )

    convert_to_tum(
        args.estimated,
        est_tum,
        args.est_time_scale,
    )

    run_evo(
        gt_tum,
        est_tum,
        args.output,
        args.max_time_diff,
    )


if __name__ == "__main__":
    main()
