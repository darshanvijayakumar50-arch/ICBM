# Pragadheesh
import argparse
import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

N = 4000
G = 9.80665


@dataclass(frozen=True)
class ImuRecord:
    i: int
    time: float
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float

    def __eq__(self, other) -> bool:
        return (
                self.wx == other.wx
                and self.wy == other.wy
                and self.wz == other.wz
                and self.ax == other.ax
                and self.ay == other.ay
                and self.az == other.az
            )

    def mag_acceleration(self):
        return math.sqrt(
            self.ax*self.ax +
            self.ay*self.ay +
            self.az*self.az
        )

    def mag_gyroscope(self):
        return math.sqrt(
            self.wx*self.wx +
            self.wy*self.wy +
            self.wz*self.wz
        )


@dataclass
class EstimatorState:
    window: list[ImuRecord]


    class State(IntEnum):
        FIRST_PASS_INIT = 0
        FIRST_PASS_RUNNING = 1
        SECOND_PASS_INIT = 4
        SECOND_PASS_RUNNING = 5
        SECOND_PASS_BLACKOUT = 6
    state = State.FIRST_PASS_INIT

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

    orientation_estimated = False

    #Discrete error analysis
    stuck_frames: set[int]
    duplicate_frames: set[int]
    single_dropouts: set[int]
    burst_dropouts: set[tuple[int, int]] # contains records in the form (index, frames_skipped)
    spike_frames: set[int]
    blackout_start: int = -1
    blackout_length: int = 0

    def __init__(self) -> None:
        self.window = []
        self.imu0 = ImuRecord(0, *(0.0,)*7)
        self.stuck_frames = set()
        self.duplicate_frames = set()
        self.single_dropouts = set()
        self.burst_dropouts = set()
        self.spike_frames = set()

    def tum(self):
        return (
            f"{self.time} {self.tx} {self.ty} {self.tz} {self.qx} {self.qy} {self.qz} {self.qw}"
        )


def analyse_discrete_errors(rec: ImuRecord, prev: ImuRecord, state:EstimatorState):

    dt = rec.time - prev.time

    blackout = (
        rec.ax == 0 and
        rec.ay == 0 and
        rec.az == 0 and
        rec.wx == 0 and
        rec.wy == 0 and
        rec.wz == 0
    )

    if dt == 0:
        state.duplicate_frames.add(rec.i)
    elif prev == rec and not blackout:
        state.stuck_frames.add(rec.i)

    n = round(dt / state.dt) - 1
    if n == 1:
        state.single_dropouts.add(rec.i)
    elif n > 1:
        state.burst_dropouts.add((rec.i, n))
    return blackout

def handle_state(state: EstimatorState, imu: ImuRecord ) -> list[str]:
    if state.state >= EstimatorState.State.SECOND_PASS_INIT:
        # Second pass not yet implemented
        return []
    else:

        state.window.append(imu)

        if state.state == EstimatorState.State.FIRST_PASS_RUNNING:
            blackout = analyse_discrete_errors(imu, state.imu0, state)
            if blackout:
                if state.blackout_start == -1:
                    state.blackout_start = imu.i
            else:
                state.blackout_start = -1

            state.time = imu.time
            state.imu0 = imu
            
        if state.state == EstimatorState.State.FIRST_PASS_INIT and imu.i == N:
            state.state = EstimatorState.State.FIRST_PASS_RUNNING

            state.dt = (state.window[N].time - state.window[0].time) / (N+1)
            print(f"Estimated dt: {state.dt*1e-6} ms")

            state.time = state.window[0].time - state.dt
            state.imu0 = ImuRecord(-1, state.time, *(0,)*6)

            for record in state.window:
                analyse_discrete_errors(record, state.imu0, state)
                state.time = record.time
                state.imu0 = record
        if len(state.window) > N:
            state.window.pop(0)

        return []

def run_pass(state, imu_csv_path, outf=None):
    with open(imu_csv_path) as inf:
        header = inf.readline()
        i = 0
        j = 0
        while True:
            line = inf.readline()
            if not line:
                break

            time, wx, wy, wz, ax, ay, az = [float(x.strip()) for x in line.split(",")]
            output = handle_state(state, ImuRecord(i, time, wx, wy, wz, ax, ay, az) )

            if outf:
                outf.writelines(rec + "\n" for rec in output)
                j += len(output)
            i += 1
    return j

def run_trajectory_estimation(imu_csv_path, output_csv_path):
    start = time.perf_counter()
    state = EstimatorState()
    run_pass(state, imu_csv_path)
    state.blackout_length = state.imu0.i - state.blackout_start

    print(f"Single Dropouts: {len(state.single_dropouts)}")
    print(f"Burst Dropouts: {len(state.burst_dropouts)}")
    print(f"Duplicates: {len(state.duplicate_frames)}")
    print(f"Stuck: {len(state.stuck_frames)}")
    #Not implemented Yet!!
    print(f"Spike: {len(state.spike_frames)}")

    elapsed = time.perf_counter() - start

    print(f"Elapsed (pass 1): {elapsed:.6f}s ({elapsed * 1000:.2f} ms)")
    
    print(f"{state.blackout_length} blackout frames detected")
    print(f"Blackout for {state.blackout_length*state.dt*1e-9:.3f} s")
    print(f"Final state is :{state.state}, starting second pass")
    
    state.state = EstimatorState.State.SECOND_PASS_INIT

    with open(output_csv_path, "w") as outf:
        outf.write("# timestamp tx ty tz qx qy qz qw\n")
        written_records = run_pass(state, imu_csv_path, outf)
        print("done", state.state, written_records)

    elapsed = time.perf_counter() - start
    print(f"Elapsed (total): {elapsed:.6f}s ({elapsed * 1000:.2f} ms)")
    duration = state.imu0.i*state.dt*1e-9
    print(f"run for {duration}s")
    print(f"Elapsed (per second): {elapsed*1e3/duration:.6f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imu", help="Input degraded IMU CSV")
    parser.add_argument("output_est", help="Output estimated trajectory CSV")
    args = parser.parse_args()
    run_trajectory_estimation(args.input_imu, args.output_est)
