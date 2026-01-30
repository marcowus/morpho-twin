"""Solver backends for NMPC."""

from __future__ import annotations

__all__ = []

try:
    from .casadi_backend import CasADiSolver  # noqa: F401

    __all__.append("CasADiSolver")
except ImportError:
    pass

try:
    from .acados_backend import AcadosSolver  # noqa: F401

    __all__.append("AcadosSolver")
except ImportError:
    pass
