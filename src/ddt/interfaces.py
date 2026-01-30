from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class StepResult:
    x: np.ndarray
    y: np.ndarray


@runtime_checkable
class Plant(Protocol):
    dt: float

    def reset(self) -> StepResult: ...
    def step(self, u: np.ndarray) -> StepResult: ...


@dataclass(frozen=True)
class Estimate:
    x_hat: np.ndarray
    theta_hat: np.ndarray
    theta_cov: np.ndarray  # covariance estimate (may be approximate)


@runtime_checkable
class Estimator(Protocol):
    def reset(self) -> None: ...
    def update(self, y: np.ndarray, u_applied: np.ndarray) -> Estimate: ...


@runtime_checkable
class Controller(Protocol):
    def reset(self) -> None: ...
    def compute_u(self, ref: np.ndarray, est: Estimate) -> np.ndarray: ...


@runtime_checkable
class SafetyFilter(Protocol):
    def reset(self) -> None: ...
    def filter(self, u_nom: np.ndarray, est: Estimate) -> np.ndarray: ...
