"""Symbolic dynamics for OCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass


@dataclass
class DiscreteDynamics:
    """Discrete-time dynamics for optimal control.

    x_{k+1} = f(x_k, u_k, theta)
    """

    f: ca.Function  # Dynamics function
    A: ca.Function  # Jacobian df/dx
    B: ca.Function  # Jacobian df/du
    C: ca.Function  # Jacobian df/dtheta
    nx: int
    nu: int
    ntheta: int


def build_discrete_dynamics(
    dt: float,
    nx: int = 1,
    nu: int = 1,
    ntheta: int = 2,
    model_type: str = "linear_scalar",
) -> DiscreteDynamics:
    """Build discrete dynamics for OCP.

    Args:
        dt: Sampling time
        nx: State dimension
        nu: Input dimension
        ntheta: Parameter dimension
        model_type: Type of model ('linear_scalar', 'generic')

    Returns:
        DiscreteDynamics object with function and Jacobians
    """
    if ca is None:
        raise ImportError("CasADi required for dynamics building.")

    if model_type == "linear_scalar":
        return _build_linear_scalar_dynamics(dt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_linear_scalar_dynamics(dt: float) -> DiscreteDynamics:
    """Build linear scalar dynamics: x+ = a*x + b*u."""
    nx, nu, ntheta = 1, 1, 2

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    theta = ca.SX.sym("theta", ntheta)

    a = theta[0]
    b = theta[1]

    # Discrete dynamics (already discrete form)
    x_next = a * x + b * u

    # Create functions
    f = ca.Function("f", [x, u, theta], [x_next], ["x", "u", "theta"], ["x_next"])

    # Jacobians
    A_expr = ca.jacobian(x_next, x)
    B_expr = ca.jacobian(x_next, u)
    C_expr = ca.jacobian(x_next, theta)

    A = ca.Function("A", [x, u, theta], [A_expr], ["x", "u", "theta"], ["A"])
    B = ca.Function("B", [x, u, theta], [B_expr], ["x", "u", "theta"], ["B"])
    C = ca.Function("C", [x, u, theta], [C_expr], ["x", "u", "theta"], ["C"])

    return DiscreteDynamics(f=f, A=A, B=B, C=C, nx=nx, nu=nu, ntheta=ntheta)


def evaluate_dynamics(
    dyn: DiscreteDynamics,
    x: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Evaluate dynamics at a point."""
    return np.array(dyn.f(x, u, theta)).flatten()


def linearize_dynamics(
    dyn: DiscreteDynamics,
    x: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearize dynamics at operating point.

    Returns A, B, C matrices such that:
        x_next ≈ f(x0,u0,θ) + A*(x-x0) + B*(u-u0) + C*(θ-θ0)
    """
    A = np.array(dyn.A(x, u, theta)).reshape(dyn.nx, dyn.nx)
    B = np.array(dyn.B(x, u, theta)).reshape(dyn.nx, dyn.nu)
    C = np.array(dyn.C(x, u, theta)).reshape(dyn.nx, dyn.ntheta)
    return A, B, C
