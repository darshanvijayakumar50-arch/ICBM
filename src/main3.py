# Pragadheesh
import argparse
from dataclasses import dataclass
from enum import Enum

N = 2000

class StateMachine(Enum):
    INITIALIZING = 0
    RUNNING = 1
    BLACKOUT = 2

@dataclass
class State:
    window = []
    state = StateMachine.INITIALIZING

    i = 0

    dt = 0.0
    time = 0.0

    qx = 0.0
    qy = 0.0
    qz = 0.0
    qw = 0.0

    tx = 0.0
    ty = 0.0
    tz = 0.0

    vx = 0.0
    vy = 0.0
    vz = 0.0
    

def handle_state(state: State, time, wx, wy, wz, ax, ay, az ):
    state.window.append((time, wx, wy, wz, ax, ay, az))
    state.i += 1
        
    if state.state == StateMachine.INITIALIZING and state.i == N:
        state.state = StateMachine.RUNNING
        dt = 0
        t0 = state.window[0][0]
        for t, wx, wy, wz, ax, ay, az in state.window:
            pass

        state.dt = (state.window[-1][0] - state.window[0][0]) / N
        print(state.dt)

    if state.i > N:
        state.window.pop(0)

    return (0, 0, 0, 0, 0, 0, 0, 0)

def run_trajectory_estimation(imu_csv_path, output_csv_path):
    #Naive algortithm

    with open(imu_csv_path) as inf, open(output_csv_path, "w") as outf:
        #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        header = inf.readline()

        outf.write("# timestamp tx ty tz qx qy qz qw\n")
        state = State()

        while True:
            line = inf.readline()
            if not line:
                break

            time, wx, wy, wz, ax, ay, az = [float(x.strip()) for x in line.split(",")]
            
            output = handle_state(state, time, wx, wy, wz, ax, ay, az )

            # for t, tx, ty, tz, qx, qy, qz, qw in output:
            #     pass




        print("done", state.i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imu", help="Input degraded IMU CSV")
    parser.add_argument("output_est", help="Output estimated trajectory CSV")
    args = parser.parse_args()
    run_trajectory_estimation(args.input_imu, args.output_est)
