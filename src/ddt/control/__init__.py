"""Control module: NMPC with RTI and dual-control."""

from __future__ import annotations

from .nmpc_base import NMPCBase, freeze_theta

__all__ = ["NMPCBase", "freeze_theta"]

# CasADi-based NMPC (fallback)
try:
    from .nmpc_casadi import CasADiNMPC  # noqa: F401

    __all__.append("CasADiNMPC")
except ImportError:
    pass

# RTI-NMPC (acados, primary)
try:
    from .nmpc_rti import RTINMPC  # noqa: F401

    __all__.append("RTINMPC")
except ImportError:
    pass
