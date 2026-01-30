"""CasADi symbolic model builder for MHE/NMPC."""

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
class SymbolicModel:
    """CasADi symbolic model for estimation and control.

    Dynamics: x_{k+1} = f(x_k, u_k, theta)
    Output:   y_k = h(x_k)
    """

    # Dimensions
    nx: int
    nu: int
    ny: int
    ntheta: int

    # CasADi symbols
    x: ca.SX  # State (nx,)
    u: ca.SX  # Input (nu,)
    theta: ca.SX  # Parameters (ntheta,)
    y: ca.SX  # Output (ny,)

    # Dynamics and output functions
    f_discrete: ca.Function  # x_next = f(x, u, theta)
    h_output: ca.Function  # y = h(x)

    # Jacobians (for linearization)
    A_func: ca.Function  # df/dx
    B_func: ca.Function  # df/du
    C_func: ca.Function  # df/dtheta


def build_linear_scalar_model(dt: float) -> SymbolicModel:
    """Build symbolic model for scalar linear dynamics.

    x_{k+1} = a * x_k + b * u_k
    y_k = x_k

    theta = [a, b]
    """
    if ca is None:
        raise ImportError("CasADi required for symbolic model building.")

    # Dimensions
    nx, nu, ny, ntheta = 1, 1, 1, 2

    # Symbolic variables
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    theta = ca.SX.sym("theta", ntheta)

    a = theta[0]
    b = theta[1]

    # Discrete dynamics (already discrete, no need for RK4)
    x_next = a * x + b * u

    # Output (identity)
    y = x

    # Create CasADi functions
    f_discrete = ca.Function("f_discrete", [x, u, theta], [x_next], ["x", "u", "theta"], ["x_next"])
    h_output = ca.Function("h_output", [x], [y], ["x"], ["y"])

    # Jacobians
    A = ca.jacobian(x_next, x)
    B = ca.jacobian(x_next, u)
    C = ca.jacobian(x_next, theta)

    A_func = ca.Function("A", [x, u, theta], [A], ["x", "u", "theta"], ["A"])
    B_func = ca.Function("B", [x, u, theta], [B], ["x", "u", "theta"], ["B"])
    C_func = ca.Function("C", [x, u, theta], [C], ["x", "u", "theta"], ["C"])

    return SymbolicModel(
        nx=nx,
        nu=nu,
        ny=ny,
        ntheta=ntheta,
        x=x,
        u=u,
        theta=theta,
        y=y,
        f_discrete=f_discrete,
        h_output=h_output,
        A_func=A_func,
        B_func=B_func,
        C_func=C_func,
    )


def build_continuous_to_discrete(
    f_cont: ca.Function,
    dt: float,
    method: str = "rk4",
) -> ca.Function:
    """Convert continuous dynamics to discrete using integration.

    Args:
        f_cont: Continuous dynamics x_dot = f(x, u, theta)
        dt: Sampling time
        method: Integration method ('euler' or 'rk4')

    Returns:
        Discrete dynamics function x_next = f_discrete(x, u, theta)
    """
    if ca is None:
        raise ImportError("CasADi required.")

    # Get symbolic inputs from function
    x = ca.SX.sym("x", f_cont.size1_in(0))
    u = ca.SX.sym("u", f_cont.size1_in(1))
    theta = ca.SX.sym("theta", f_cont.size1_in(2))

    if method == "euler":
        x_next = x + dt * f_cont(x, u, theta)
    elif method == "rk4":
        k1 = f_cont(x, u, theta)
        k2 = f_cont(x + dt / 2 * k1, u, theta)
        k3 = f_cont(x + dt / 2 * k2, u, theta)
        k4 = f_cont(x + dt * k3, u, theta)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return ca.Function(
        "f_discrete",
        [x, u, theta],
        [x_next],
        ["x", "u", "theta"],
        ["x_next"],
    )


def linearize_model(
    model: SymbolicModel,
    x0: np.ndarray,
    u0: np.ndarray,
    theta0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearize dynamics around operating point.

    Returns:
        A: df/dx at (x0, u0, theta0)
        B: df/du at (x0, u0, theta0)
        C: df/dtheta at (x0, u0, theta0)
    """
    A = np.array(model.A_func(x0, u0, theta0)).reshape(model.nx, model.nx)
    B = np.array(model.B_func(x0, u0, theta0)).reshape(model.nx, model.nu)
    C = np.array(model.C_func(x0, u0, theta0)).reshape(model.nx, model.ntheta)
    return A, B, C
