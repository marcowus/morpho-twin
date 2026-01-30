"""Estimation module: state and parameter estimation."""

from __future__ import annotations

from .windowed_ls import WindowedLeastSquaresEstimator

__all__ = ["WindowedLeastSquaresEstimator"]

# MHE is conditionally available (requires casadi)
try:
    from .mhe import CasADiMHE, MHEBase  # noqa: F401

    __all__.extend(["MHEBase", "CasADiMHE"])

    # acados MHE is also conditional
    try:
        from .mhe import AcadosMHE  # noqa: F401

        __all__.append("AcadosMHE")
    except ImportError:
        pass
except ImportError:
    pass
