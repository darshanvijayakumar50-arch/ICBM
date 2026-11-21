# Pragadheesh
import argparse
import math
from dataclasses import dataclass
from enum import Enum

N = 4000

class StateMachine(Enum):
    INITIALIZING = 0
    RUNNING = 1
    BLACKOUT = 2


@dataclass(frozen=True)
class ImuRecord:
    time: float
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float


@dataclass
class State:
    window: list[ImuRecord]
    state = StateMachine.INITIALIZING

    i = 0

    dt = 0.0
    time = 0.0
    imu0: ImuRecord

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

    def __init__(self) -> None:
        self.window = []
        self.imu0 = ImuRecord(*(0.0,)*7)

def cummulate(state: State, rec: ImuRecord ):
    dt = rec.time - state.imu0.time

    if dt == 0:
        print(f"duplicate at {rec.time}")

    if dt < 0:
        print(f"Negative DT?! at {rec.time}")

    if dt > state.dt:
        n = round(dt/state.dt) - 1
        if n > 4:
            print(f"Burst Dropout detected {n} at {rec.time}")
        else:
            print(f"Small droupout detected {n} at {rec.time}")

def handle_state(state: State, imu: ImuRecord ) -> list[ImuRecord]:
    state.window.append(imu)
    state.i += 1
        
    if state.state == StateMachine.INITIALIZING and state.i == N:
        state.state = StateMachine.RUNNING

        state.dt = (state.window[-1].time - state.window[0].time) / N
        print(state.dt)
        state.time = state.window[0].time - state.dt
        state.imu0 = ImuRecord(state.time, *(0,)*6)

        for record in state.window:
            cummulate(state, record)
            state.time = record.time
            state.imu0 = record


    if state.i > N:
        state.window.pop(0)

    return []

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
            
            output = handle_state(state, ImuRecord(time, wx, wy, wz, ax, ay, az) )

            for rec in output:
                pass




        print("done", state.i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imu", help="Input degraded IMU CSV")
    parser.add_argument("output_est", help="Output estimated trajectory CSV")
    args = parser.parse_args()
    run_trajectory_estimation(args.input_imu, args.output_est)
