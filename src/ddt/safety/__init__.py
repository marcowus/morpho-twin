"""Safety module: CBF-QP and barrier functions."""

from __future__ import annotations

from .barriers import (
    BarrierFunction,
    BoxBarrier,
    CompositeBarrier,
    EllipsoidBarrier,
    create_box_barriers,
)
from .robust_margin import (
    adaptive_alpha,
    compute_robust_margin,
)
from .state_box import StateBoxSafetyFilter

__all__ = [
    "StateBoxSafetyFilter",
    "BarrierFunction",
    "BoxBarrier",
    "EllipsoidBarrier",
    "CompositeBarrier",
    "create_box_barriers",
    "compute_robust_margin",
    "adaptive_alpha",
]

# CBF-QP is conditionally available (requires osqp)
try:
    from .cbf_qp import CBFQPSafetyFilter  # noqa: F401

    __all__.append("CBFQPSafetyFilter")
except ImportError:
    pass
