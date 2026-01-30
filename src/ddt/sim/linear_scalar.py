from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Plant, StepResult


@dataclass
class LinearScalarPlant(Plant):
    """Scalar plant:
    x_{k+1} = a x_k + b u_k + w_k
    y_k = x_k + v_k

    This is only for demo/testing of the scaffold.
    """

    dt: float
    a_true: float
    b_true: float
    process_noise_std: float
    meas_noise_std: float
    x0: float

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._x = np.array([self.x0], dtype=float)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> StepResult:
        self._x = np.array([self.x0], dtype=float)
        y = self._x + self._rng.normal(0.0, self.meas_noise_std, size=self._x.shape)
        return StepResult(x=self._x.copy(), y=y)

    def step(self, u: np.ndarray) -> StepResult:
        u = np.asarray(u, dtype=float).reshape(1)
        w = self._rng.normal(0.0, self.process_noise_std, size=self._x.shape)
        self._x = self.a_true * self._x + self.b_true * u + w
        v = self._rng.normal(0.0, self.meas_noise_std, size=self._x.shape)
        y = self._x + v
        return StepResult(x=self._x.copy(), y=y)
