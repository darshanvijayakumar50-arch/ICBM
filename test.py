
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation, Slerp

# Import your existing ESKF pipeline
from intit import run_master_eskf_pipeline


GT_COLUMNS = [
    "time", "x", "y", "z",
    "vx", "vy", "vz",
    "qx", "qy", "qz", "qw",
]

IMU_COLUMNS = [
    "time", "ax", "ay", "az",
    "wx", "wy", "wz",
]


# ---------------------------------------------------------
# CSV LOADING
# ---------------------------------------------------------

def load_csv_files(gt_path, imu_path):

    gt = pd.read_csv(gt_path)
    imu = pd.read_csv(imu_path)

    for df, columns, name in [
        (gt, GT_COLUMNS, "Ground truth"),
        (imu, IMU_COLUMNS, "IMU"),
    ]:
        missing = set(columns) - set(df.columns)

        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

        if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains NaN or infinite values")

    gt = gt[GT_COLUMNS].astype(float).sort_values("time")
    imu = imu[IMU_COLUMNS].astype(float).sort_values("time")

    if gt["time"].duplicated().any():
        raise ValueError("Duplicate ground-truth timestamps")

    if imu["time"].duplicated().any():
        raise ValueError("Duplicate IMU timestamps")

    if len(gt) < 2 or len(imu) < 2:
        raise ValueError("Insufficient CSV samples")

    # Normalize ground-truth quaternions
    q = gt[["qx", "qy", "qz", "qw"]].to_numpy()
    q_norm = np.linalg.norm(q, axis=1)

    if np.any(q_norm < 1e-12):
        raise ValueError("Invalid zero quaternion")

    gt[["qx", "qy", "qz", "qw"]] = q / q_norm[:, None]

    return gt.reset_index(drop=True), imu.reset_index(drop=True)


# ---------------------------------------------------------
# GROUND TRUTH SYNCHRONIZATION
# ---------------------------------------------------------

def synchronize_ground_truth(gt, imu):

    t = imu["time"].to_numpy()
    gt_t = gt["time"].to_numpy()

    if t[0] < gt_t[0] or t[-1] > gt_t[-1]:
        raise ValueError(
            "IMU time range exceeds ground-truth time range"
        )

    result = {}

    # Interpolate position and velocity
    for col in ["x", "y", "z", "vx", "vy", "vz"]:

        result[col] = np.interp(
            t,
            gt_t,
            gt[col].to_numpy()
        )

    # Interpolate quaternion using SLERP
    rotations = Rotation.from_quat(
        gt[["qx", "qy", "qz", "qw"]].to_numpy()
    )

    slerp = Slerp(gt_t, rotations)

    result["quaternion"] = slerp(t).as_quat()

    return pd.DataFrame({
        "time": t,
        "x": result["x"],
        "y": result["y"],
        "z": result["z"],
        "vx": result["vx"],
        "vy": result["vy"],
        "vz": result["vz"],
        "qx": result["quaternion"][:, 0],
        "qy": result["quaternion"][:, 1],
        "qz": result["quaternion"][:, 2],
        "qw": result["quaternion"][:, 3],
    })


# ---------------------------------------------------------
# ESKF EXECUTION
# ---------------------------------------------------------

def run_estimator(imu):

    # Exactly [time, ax, ay, az, wx, wy, wz]
    imu_dataset = imu[IMU_COLUMNS].to_numpy(dtype=float)

    estimated = np.asarray(
        run_master_eskf_pipeline(imu_dataset),
        dtype=float,
    )

    if estimated.ndim != 2:
        raise ValueError("ESKF output must be a 2D array")

    if estimated.shape[0] != len(imu):
        raise ValueError(
            "ESKF output sample count does not match IMU sample count"
        )

    if estimated.shape[1] not in (8, 11):
        raise ValueError(
            "Expected ESKF output columns: "
            "[time,x,y,z,qx,qy,qz,qw] or "
            "[time,x,y,z,vx,vy,vz,qx,qy,qz,qw]"
        )

    if not np.isfinite(estimated).all():
        raise ValueError("ESKF output contains non-finite values")

    # Output format:
    # 8 columns: time, position, quaternion
    # 11 columns: time, position, velocity, quaternion

    if estimated.shape[1] == 8:

        output = pd.DataFrame(
            estimated,
            columns=[
                "time", "x", "y", "z",
                "qx", "qy", "qz", "qw",
            ],
        )

        output["vx"] = np.nan
        output["vy"] = np.nan
        output["vz"] = np.nan

    else:

        output = pd.DataFrame(
            estimated,
            columns=GT_COLUMNS,
        )

    q = output[["qx", "qy", "qz", "qw"]].to_numpy()
    norms = np.linalg.norm(q, axis=1)

    if np.any(norms < 1e-12):
        print("oh no")
        raise ValueError("ESKF returned invalid quaternion")

    output[["qx", "qy", "qz", "qw"]] = q / norms[:, None]

    return output


# ---------------------------------------------------------
# ERROR CALCULATION
# ---------------------------------------------------------

def calculate_errors(gt, est):

    position_error = (
        est[["x", "y", "z"]].to_numpy()
        - gt[["x", "y", "z"]].to_numpy()
    )

    errors = {
        "position": position_error,
    }

    # Velocity errors, only when ESKF provides velocity
    if est[["vx", "vy", "vz"]].notna().all().all():

        errors["velocity"] = (
            est[["vx", "vy", "vz"]].to_numpy()
            - gt[["vx", "vy", "vz"]].to_numpy()
        )

    estimated_rotations = Rotation.from_quat(
        est[["qx", "qy", "qz", "qw"]].to_numpy()
    )

    true_rotations = Rotation.from_quat(
        gt[["qx", "qy", "qz", "qw"]].to_numpy()
    )

    relative_rotation = (
        true_rotations.inv() * estimated_rotations
    )

    errors["orientation_deg"] = np.rad2deg(
        relative_rotation.magnitude()
    )

    return errors


def print_metrics(errors):

    print("\n========== ESKF EVALUATION ==========")

    for name in ["position", "velocity"]:

        if name not in errors:
            continue

        error = errors[name]

        component_rmse = np.sqrt(np.mean(error**2, axis=0))
        total_rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))

        print(f"\n{name.upper()}")

        print("Component RMSE:", component_rmse)
        print("3D RMSE:", total_rmse)

    orientation_rmse = np.sqrt(
        np.mean(errors["orientation_deg"]**2)
    )

    print("\nORIENTATION")
    print(f"Angular RMSE: {orientation_rmse:.6f} degrees")

    print("====================================\n")


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def plot_trajectory(gt, est):

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gt["x"], gt["y"], gt["z"],
        label="Ground Truth",
        linewidth=2,
    )

    ax.plot(
        est["x"], est["y"], est["z"],
        "--",
        label="ESKF Estimate",
    )

    ax.scatter(
        gt["x"].iloc[0],
        gt["y"].iloc[0],
        gt["z"].iloc[0],
        marker="o",
        label="Start",
    )

    ax.scatter(
        gt["x"].iloc[-1],
        gt["y"].iloc[-1],
        gt["z"].iloc[-1],
        marker="x",
        label="End",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Flight Trajectory")

    ax.legend()
    plt.tight_layout()


def plot_position(gt, est):

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for i, axis in enumerate(axes):

        col = ["x", "y", "z"][i]

        axis.plot(gt["time"], gt[col], label="Ground Truth")
        axis.plot(est["time"], est[col], "--", label="ESKF")

        axis.set_ylabel(f"{col.upper()} (m)")
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position Comparison")
    plt.tight_layout()


def plot_velocity(gt, est):

    if est[["vx", "vy", "vz"]].isna().any().any():
        print("Velocity plots skipped: ESKF does not return velocity.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for i, axis in enumerate(axes):

        col = ["vx", "vy", "vz"][i]

        axis.plot(gt["time"], gt[col], label="Ground Truth")
        axis.plot(est["time"], est[col], "--", label="ESKF")

        axis.set_ylabel(f"{col.upper()} (m/s)")
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Velocity Comparison")
    plt.tight_layout()


def plot_orientation(gt, est):

    gt_euler = Rotation.from_quat(
        gt[["qx", "qy", "qz", "qw"]].to_numpy()
    ).as_euler("xyz", degrees=True)

    est_euler = Rotation.from_quat(
        est[["qx", "qy", "qz", "qw"]].to_numpy()
    ).as_euler("xyz", degrees=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for i, axis in enumerate(axes):

        axis.plot(gt["time"], gt_euler[:, i], label="Ground Truth")
        axis.plot(est["time"], est_euler[:, i], "--", label="ESKF")

        axis.set_ylabel(["Roll (deg)", "Pitch (deg)", "Yaw (deg)"][i])
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Orientation Comparison")
    plt.tight_layout()


def plot_errors(gt, errors):

    t = gt["time"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    for i, axis in enumerate(axes):

        axis.plot(t, errors["position"][:, i])
        axis.set_ylabel(["X error (m)", "Y error (m)", "Z error (m)"][i])
        axis.grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Position Estimation Error")
    plt.tight_layout()

    if "velocity" in errors:

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

        for i, axis in enumerate(axes):

            axis.plot(t, errors["velocity"][:, i])
            axis.set_ylabel(
                ["Vx error (m/s)", "Vy error (m/s)", "Vz error (m/s)"][i]
            )
            axis.grid(True)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Velocity Estimation Error")
        plt.tight_layout()

    plt.figure(figsize=(13, 4))

    plt.plot(t, errors["orientation_deg"])

    plt.xlabel("Time (s)")
    plt.ylabel("Angular Error (degrees)")
    plt.title("Orientation Estimation Error")
    plt.grid(True)


def plot_imu(imu):

    t = imu["time"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for col in ["ax", "ay", "az"]:
        axes[0].plot(t, imu[col], label=col)

    for col in ["wx", "wy", "wz"]:
        axes[1].plot(t, imu[col], label=col)

    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[1].set_ylabel("Angular velocity (rad/s)")

    axes[0].legend()
    axes[1].legend()

    axes[0].grid(True)
    axes[1].grid(True)

    axes[1].set_xlabel("Time (s)")

    fig.suptitle("Raw IMU Measurements")
    plt.tight_layout()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth.csv",
    )

    parser.add_argument(
        "--imu",
        default="data/imu_data.csv",
    )

    parser.add_argument(
        "--output",
        default="estimated_trajectory.csv",
    )

    args = parser.parse_args()

    print("Loading CSV files...")

    gt, imu = load_csv_files(
        args.ground_truth,
        args.imu,
    )

    print(f"Ground truth samples: {len(gt)}")
    print(f"IMU samples: {len(imu)}")

    dt = np.diff(imu["time"].to_numpy())

    print(f"Average IMU frequency: {1.0 / np.mean(dt):.2f} Hz")

    print("\nSynchronizing ground truth...")

    gt_sync = synchronize_ground_truth(gt, imu)

    print("Running ESKF...")

    estimated = run_estimator(imu)

    # Ensure timestamps correspond
    if not np.allclose(
        estimated["time"],
        gt_sync["time"],
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(
            "ESKF output timestamps do not match IMU timestamps"
        )

    print("Calculating errors...")

    errors = calculate_errors(gt_sync, estimated)

    print_metrics(errors)

    # Save results
    estimated.to_csv(args.output, index=False)

    print(f"Estimated trajectory saved to: {args.output}")
    print("Output shape:", estimated.shape)

    print("\nLast estimated row:")
    print(estimated.iloc[-1])

    # Visualizations
    plot_trajectory(gt_sync, estimated)
    plot_position(gt_sync, estimated)
    plot_velocity(gt_sync, estimated)
    plot_orientation(gt_sync, estimated)
    plot_errors(gt_sync, errors)
    plot_imu(imu)

    plt.show()


if __name__ == "__main__":
    main()
