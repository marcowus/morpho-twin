"""Constraint handling for OCP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoxConstraints:
    """Box constraints on states and inputs."""

    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray


def build_constraint_vectors(
    x_min: list[float],
    x_max: list[float],
    u_min: list[float],
    u_max: list[float],
) -> BoxConstraints:
    """Build constraint object from lists.

    Args:
        x_min: State lower bounds
        x_max: State upper bounds
        u_min: Input lower bounds
        u_max: Input upper bounds

    Returns:
        BoxConstraints object
    """
    return BoxConstraints(
        x_min=np.array(x_min, dtype=np.float64),
        x_max=np.array(x_max, dtype=np.float64),
        u_min=np.array(u_min, dtype=np.float64),
        u_max=np.array(u_max, dtype=np.float64),
    )


def check_state_feasibility(
    x: np.ndarray,
    constraints: BoxConstraints,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Check if state satisfies constraints.

    Args:
        x: State vector
        constraints: Box constraints
        tolerance: Feasibility tolerance

    Returns:
        (is_feasible, max_violation)
    """
    x = np.atleast_1d(x)
    violation_low = np.maximum(0, constraints.x_min - x)
    violation_high = np.maximum(0, x - constraints.x_max)
    max_violation = float(np.max(np.maximum(violation_low, violation_high)))
    is_feasible = max_violation <= tolerance
    return is_feasible, max_violation


def check_input_feasibility(
    u: np.ndarray,
    constraints: BoxConstraints,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Check if input satisfies constraints.

    Args:
        u: Input vector
        constraints: Box constraints
        tolerance: Feasibility tolerance

    Returns:
        (is_feasible, max_violation)
    """
    u = np.atleast_1d(u)
    violation_low = np.maximum(0, constraints.u_min - u)
    violation_high = np.maximum(0, u - constraints.u_max)
    max_violation = float(np.max(np.maximum(violation_low, violation_high)))
    is_feasible = max_violation <= tolerance
    return is_feasible, max_violation


def project_to_constraints(
    x: np.ndarray,
    u: np.ndarray,
    constraints: BoxConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    """Project state and input to feasible region.

    Args:
        x: State vector
        u: Input vector
        constraints: Box constraints

    Returns:
        (x_proj, u_proj) projected to feasible region
    """
    x_proj = np.clip(x, constraints.x_min, constraints.x_max)
    u_proj = np.clip(u, constraints.u_min, constraints.u_max)
    return x_proj, u_proj


def soften_constraints(
    constraints: BoxConstraints,
    softening_factor: float = 0.1,
) -> BoxConstraints:
    """Create softened constraints for robustness.

    Tightens constraints by softening_factor to create margin.

    Args:
        constraints: Original constraints
        softening_factor: Fraction to tighten (0-1)

    Returns:
        Softened BoxConstraints
    """
    x_range = constraints.x_max - constraints.x_min
    u_range = constraints.u_max - constraints.u_min

    return BoxConstraints(
        x_min=constraints.x_min + softening_factor * x_range,
        x_max=constraints.x_max - softening_factor * x_range,
        u_min=constraints.u_min + softening_factor * u_range,
        u_max=constraints.u_max - softening_factor * u_range,
    )
