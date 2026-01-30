from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    iae: float
    max_constraint_violation: float


def compute_metrics(y: np.ndarray, ref: np.ndarray, x_min: float, x_max: float) -> Metrics:
    y = np.asarray(y, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    iae = float(np.sum(np.abs(ref - y)))
    violation_low = np.maximum(0.0, x_min - y)
    violation_high = np.maximum(0.0, y - x_max)
    max_violation = float(np.max(np.maximum(violation_low, violation_high))) if y.size else 0.0
    return Metrics(iae=iae, max_constraint_violation=max_violation)
