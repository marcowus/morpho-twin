from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Controller, Estimate


@dataclass
class PIDController(Controller):
    kp: float
    ki: float
    kd: float
    dt: float
    u_min: float
    u_max: float

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._e_int = 0.0
        self._e_prev = 0.0

    def compute_u(self, ref: np.ndarray, est: Estimate) -> np.ndarray:
        r = float(np.asarray(ref).reshape(()))
        y = float(est.x_hat.reshape(()))
        e = r - y
        self._e_int += e * self.dt
        de = (e - self._e_prev) / self.dt
        self._e_prev = e

        u = self.kp * e + self.ki * self._e_int + self.kd * de
        u = float(np.clip(u, self.u_min, self.u_max))
        return np.array([u], dtype=float)
