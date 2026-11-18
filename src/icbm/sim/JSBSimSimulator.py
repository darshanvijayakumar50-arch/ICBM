
from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import jsbsim


class JSBSimSimulator:
    """Convenient wrapper around JSBSim's FGFDMExec."""

    def __init__(
        self,
        aircraft: str,
        *,
        root_dir: str | Path | None = None,
        dt: float = 1.0 / 240.0,
    ):
        self.fdm: jsbsim.FGFDMExec = jsbsim.FGFDMExec(
            str(root_dir) if root_dir else None,
        )

        self.aircraft = aircraft
        self.dt = dt
        self.time = 0.0
        self.initialized = False

        self._configure()

    def _configure(self) -> None:
        """Configure the simulation."""

        self.fdm.set_dt(self.dt)

        if not self.fdm.load_model(self.aircraft):
            raise RuntimeError(
                f"Failed to load aircraft: {self.aircraft}"
            )

    def set(self, property_name: str, value: float) -> None:
        """Set a JSBSim property."""
        self.fdm[property_name] = value

    def get(self, property_name: str) -> float:
        """Get a JSBSim property."""
        return float(self.fdm[property_name])

    def configure_initial_conditions(
        self,
        *,
        latitude: float,
        longitude: float,
        altitude_ft: float,
        airspeed_kts: float,
        heading_deg: float = 0.0,
        climb_fpm: float = 0.0,
        gamma_deg: float = 0.0,
    ) -> None:
        """Set initial aircraft conditions."""

        conditions = self.fdm

        conditions["ic/lat-gc-deg"] = latitude
        conditions["ic/long-gc-deg"] = longitude
        conditions["ic/h-sl-ft"] = altitude_ft
        conditions["ic/vt-kts"] = airspeed_kts
        conditions["ic/psi-true-deg"] = heading_deg
        conditions["ic/roc-fpm"] = climb_fpm
        conditions["ic/gamma-deg"] = gamma_deg

    def initialize(self) -> None:
        """Initialize the aircraft state."""

        if not self.fdm.run_ic():
            raise RuntimeError("Failed to initialize JSBSim")

        self.initialized = True
        self.time = 0.0

    def step(self, steps: int = 1) -> None:
        """Advance simulation by a specified number of frames."""

        if not self.initialized:
            raise RuntimeError("Simulator has not been initialized")

        for _ in range(steps):
            if not self.fdm.run():
                raise RuntimeError("JSBSim simulation terminated")

            self.time += self.dt

    def run_for(self, duration: float) -> None:
        """Advance simulation by a duration in seconds."""

        steps = round(duration / self.dt)
        self.step(steps)

        
    # ---------------------------------------------------------
    # Position
    # ---------------------------------------------------------

    @property
    def latitude(self) -> float:
        return self.get("position/lat-geod-deg")

    @property
    def longitude(self) -> float:
        return self.get("position/long-gc-deg")

    @property
    def altitude(self) -> float:
        return self.get("position/h-sl-ft")

    @property
    def altitude_agl(self) -> float:
        return self.get("position/h-agl-ft")


    # ---------------------------------------------------------
    # Attitude
    # ---------------------------------------------------------

    @property
    def roll(self) -> float:
        return self.get("attitude/phi-deg")

    @property
    def pitch(self) -> float:
        return self.get("attitude/theta-deg")

    @property
    def heading(self) -> float:
        return self.get("attitude/psi-deg")


    # ---------------------------------------------------------
    # Velocity
    # ---------------------------------------------------------

    @property
    def airspeed(self) -> float:
        return self.get("velocities/vtrue-kts")

    @property
    def ground_speed(self) -> float:
        return self.get("velocities/vg-kts")

    @property
    def u(self) -> float:
        return self.get("velocities/u-fps")

    @property
    def v(self) -> float:
        return self.get("velocities/v-fps")

    @property
    def w(self) -> float:
        return self.get("velocities/w-fps")

    @property
    def climb_rate(self) -> float:
        return self.get("velocities/h-dot-fps")


    # ---------------------------------------------------------
    # Angular velocity
    # ---------------------------------------------------------

    @property
    def p(self) -> float:
        return self.get("velocities/p-rad_sec")

    @property
    def q(self) -> float:
        return self.get("velocities/q-rad_sec")

    @property
    def r(self) -> float:
        return self.get("velocities/r-rad_sec")


    # ---------------------------------------------------------
    # Acceleration
    # ---------------------------------------------------------

    @property
    def ax(self) -> float:
        return self.get("accelerations/a-pilot-x-ft_sec2")

    @property
    def ay(self) -> float:
        return self.get("accelerations/a-pilot-y-ft_sec2")

    @property
    def az(self) -> float:
        return self.get("accelerations/a-pilot-z-ft_sec2")

    @property
    def simulation_time(self) -> float:
        return self.time

    def property(self, name: str) -> float:
        """Read any JSBSim property."""
        return self.get(name)


    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------

    def close(self) -> None:
        """Release simulator reference."""
        self.initialized = False
        self.fdm = None # type: ignore

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
