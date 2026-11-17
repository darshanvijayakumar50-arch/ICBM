
import csv
import math
from pathlib import Path

import jsbsim


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AIRCRAFT = "c172p"

DT = 1.0 / 120.0
DURATION = 120.0

INITIAL_LAT = 13.0827
INITIAL_LON = 80.2707

ALTITUDE_FT = 5000.0
AIRSPEED_KTS = 100.0
HEADING_DEG = 90.0

OUTPUT_DIR = Path("data")

FT_TO_M = 0.3048
KNOT_TO_MPS = 0.514444
EARTH_RADIUS_M = 6378137.0


# ---------------------------------------------------------
# Quaternion conversion
# Euler angles: roll, pitch, yaw
# Output: qx, qy, qz, qw
# ---------------------------------------------------------

def euler_to_quaternion(roll, pitch, yaw):

    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)

    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)

    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


# ---------------------------------------------------------
# Ground truth coordinate conversion
#
# Local NED:
# x = North
# y = East
# z = Down
# ---------------------------------------------------------

def geodetic_to_ned(lat, lon, altitude_ft):

    lat0 = math.radians(INITIAL_LAT)
    lat_rad = math.radians(lat)
    lon0 = math.radians(INITIAL_LON)
    lon_rad = math.radians(lon)

    north = EARTH_RADIUS_M * (lat_rad - lat0)

    east = (
        EARTH_RADIUS_M
        * math.cos(lat0)
        * (lon_rad - lon0)
    )

    down = (ALTITUDE_FT - altitude_ft) * FT_TO_M

    return north, east, down


# ---------------------------------------------------------
# Initialize JSBSim
# ---------------------------------------------------------
def create_simulator():

    fdm = jsbsim.FGFDMExec(None)

    fdm.set_debug_level(0)
    fdm.set_dt(1.0 / 120.0)

    # 1. Load aircraft
    if not fdm.load_model("c172x"):
        raise RuntimeError("Could not load C172x")

    # 2. Configure initial conditions

    fdm["ic/lat-gc-deg"] = 13.0827
    fdm["ic/long-gc-deg"] = 80.2707

    fdm["ic/h-sl-ft"] = 5000.0

    # True airspeed, not calibrated airspeed
    fdm["ic/vt-kts"] = 100.0

    fdm["ic/psi-true-deg"] = 90.0

    # Initial guesses / flight constraints
    fdm["ic/phi-deg"] = 0.0
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/beta-deg"] = 0.0

    # Zero wind
    fdm["ic/vw-north-fps"] = 0.0
    fdm["ic/vw-east-fps"] = 0.0
    fdm["ic/vw-down-fps"] = 0.0

    # 3. Apply initial conditions
    if not fdm.run_ic():
        raise RuntimeError("Initial condition initialization failed")

    # 4. Initialize piston engine
    fdm["propulsion/set-running"] = -1

    # 5. Perform full trim
    try:
        fdm["simulation/do_simple_trim"] = 1

    except Exception as exc:
        raise RuntimeError(
            f"C172x full trim failed: {exc}"
        ) from exc

    # 6. Verify trim completion
    if fdm["simulation/trim-completed"] != 1:
        raise RuntimeError("Trim did not complete")

    print("\nAircraft successfully initialized and trimmed.")

    return fdm

# ---------------------------------------------------------
# Read IMU measurements
# ---------------------------------------------------------

def read_imu(fdm):

    ax = fdm["accelerations/a-pilot-x-ft_sec2"] * FT_TO_M
    ay = fdm["accelerations/a-pilot-y-ft_sec2"] * FT_TO_M
    az = fdm["accelerations/a-pilot-z-ft_sec2"] * FT_TO_M

    # Body angular velocity, radians/second
    wx = fdm["velocities/p-rad_sec"]
    wy = fdm["velocities/q-rad_sec"]
    wz = fdm["velocities/r-rad_sec"]

    return ax, ay, az, wx, wy, wz


# ---------------------------------------------------------
# Read ground truth
# ---------------------------------------------------------

def read_ground_truth(fdm):

    lat = fdm["position/lat-gc-deg"]
    lon = fdm["position/long-gc-deg"]
    altitude = fdm["position/h-sl-ft"]

    # Local NED position
    x, y, z = geodetic_to_ned(lat, lon, altitude)

    # NED velocity, converted from ft/s to m/s
    vx = fdm["velocities/v-north-fps"] * FT_TO_M
    vy = fdm["velocities/v-east-fps"] * FT_TO_M
    vz = fdm["velocities/v-down-fps"] * FT_TO_M

    # Aircraft attitude
    roll = fdm["attitude/phi-rad"]
    pitch = fdm["attitude/theta-rad"]
    yaw = fdm["attitude/psi-rad"]

    qx, qy, qz, qw = euler_to_quaternion(
        roll, pitch, yaw
    )

    return (
        x, y, z,
        vx, vy, vz,
        qx, qy, qz, qw
    )


# ---------------------------------------------------------
# Generate CSV files
# ---------------------------------------------------------

def generate():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imu_path = OUTPUT_DIR / "imu_data.csv"
    truth_path = OUTPUT_DIR / "ground_truth.csv"

    fdm = create_simulator()

    imu_headers = [
        "time",
        "ax", "ay", "az",
        "wx", "wy", "wz",
    ]

    truth_headers = [
        "time",
        "x", "y", "z",
        "vx", "vy", "vz",
        "qx", "qy", "qz", "qw",
    ]

    steps = round(DURATION / DT)

    try:
        with (
            imu_path.open("w", newline="") as imu_file,
            truth_path.open("w", newline="") as truth_file,
        ):

            imu_writer = csv.writer(imu_file)
            truth_writer = csv.writer(truth_file)

            imu_writer.writerow(imu_headers)
            truth_writer.writerow(truth_headers)

            for i in range(steps + 1):

                time = i * DT

                # Record current simulation state
                imu = read_imu(fdm)
                truth = read_ground_truth(fdm)

                imu_writer.writerow([
                    f"{time:.6f}",
                    *[f"{v:.9f}" for v in imu],
                ])

                truth_writer.writerow([
                    f"{time:.6f}",
                    *[f"{v:.9f}" for v in truth],
                ])

                # Advance simulation
                if i < steps and not fdm.run():
                    raise RuntimeError(
                        f"JSBSim stopped at t={time:.3f}s"
                    )

    finally:
        del fdm

    print("\nGeneration complete.")
    print(f"Duration: {DURATION} seconds")
    print(f"Sampling rate: {1 / DT:.0f} Hz")
    print(f"Samples per file: {steps + 1}")
    print(f"IMU: {imu_path}")
    print(f"Ground truth: {truth_path}")


if __name__ == "__main__":
    generate()
