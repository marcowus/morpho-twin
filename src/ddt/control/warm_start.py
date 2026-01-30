"""Warm-start management for NMPC solvers.

This module provides trajectory shifting and validation for warm-starting
NMPC solvers between time steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class WarmStartStatus:
    """Status of warm-start validation.

    Attributes:
        shift_applied: Whether the trajectory shift was applied
        is_valid: Whether the warm-start is valid (trajectory continuity ok)
        continuity_error: Error between predicted and actual state
        message: Human-readable status message
    """

    shift_applied: bool
    is_valid: bool
    continuity_error: float
    message: str


@dataclass
class WarmStartManager:
    """Manager for trajectory warm-starting.

    Handles trajectory shifting and validation for NMPC solvers.
    The key operations are:
    1. Shift: Move trajectory forward by one step
    2. Validate: Check if predicted x1 matches actual x0

    Attributes:
        horizon: MPC prediction horizon
        nx: State dimension
        nu: Input dimension
        max_state_continuity_error: Maximum allowed error for valid warm-start
    """

    horizon: int
    nx: int = 1
    nu: int = 1
    max_state_continuity_error: float = 0.5

    def shift_and_validate(
        self,
        x_actual: np.ndarray,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, WarmStartStatus]:
        """Shift trajectory and validate for warm-starting.

        The shift operation moves the trajectory forward:
        - x_shifted = [x_1, x_2, ..., x_N, x_N]  (duplicate last)
        - u_shifted = [u_1, u_2, ..., u_{N-1}, u_{N-1}]  (duplicate last)

        Validation checks if x_traj[1] (what we predicted for now)
        matches x_actual (what we measured now).

        Args:
            x_actual: Actual measured state at current time (nx,)
            x_traj: Previous state trajectory (N+1, nx)
            u_traj: Previous input trajectory (N, nu)

        Returns:
            Tuple of (x_shifted, u_shifted, status)
        """
        N = self.horizon
        x_actual = np.atleast_1d(x_actual)

        # Validate dimensions
        if x_traj.shape != (N + 1, self.nx):
            logger.warning(
                "x_traj shape mismatch: {} vs expected ({}, {})",
                x_traj.shape,
                N + 1,
                self.nx,
            )
            return self._cold_start(x_actual), self._zero_inputs(), WarmStartStatus(
                shift_applied=False,
                is_valid=False,
                continuity_error=float("inf"),
                message="Invalid trajectory dimensions",
            )

        if u_traj.shape != (N, self.nu):
            logger.warning(
                "u_traj shape mismatch: {} vs expected ({}, {})",
                u_traj.shape,
                N,
                self.nu,
            )
            return self._cold_start(x_actual), self._zero_inputs(), WarmStartStatus(
                shift_applied=False,
                is_valid=False,
                continuity_error=float("inf"),
                message="Invalid trajectory dimensions",
            )

        # Compute continuity error
        # x_traj[1] is what we predicted x would be at current time
        x_predicted = x_traj[1]
        continuity_error = float(np.linalg.norm(x_predicted - x_actual))

        # Check validity
        is_valid = continuity_error < self.max_state_continuity_error

        if not is_valid:
            logger.warning(
                "Warm-start continuity error too large | error={:.4f} | threshold={:.4f}",
                continuity_error,
                self.max_state_continuity_error,
            )

        # Perform shift regardless (still useful even if not perfectly valid)
        x_shifted = np.zeros((N + 1, self.nx))
        x_shifted[:-1] = x_traj[1:]  # [x_1, x_2, ..., x_N]
        x_shifted[-1] = x_traj[-1]  # Duplicate last state

        u_shifted = np.zeros((N, self.nu))
        u_shifted[:-1] = u_traj[1:]  # [u_1, u_2, ..., u_{N-1}]
        u_shifted[-1] = u_traj[-1]  # Duplicate last input

        status = WarmStartStatus(
            shift_applied=True,
            is_valid=is_valid,
            continuity_error=continuity_error,
            message="Warm-start valid" if is_valid else f"Continuity error: {continuity_error:.4f}",
        )

        return x_shifted, u_shifted, status

    def _cold_start(self, x0: np.ndarray) -> np.ndarray:
        """Create cold-start trajectory from initial state.

        Args:
            x0: Initial state (nx,)

        Returns:
            Trajectory initialized to x0 at all stages
        """
        x_traj = np.zeros((self.horizon + 1, self.nx))
        x_traj[:] = x0
        return x_traj

    def _zero_inputs(self) -> np.ndarray:
        """Create zero input trajectory.

        Returns:
            Zero input trajectory (N, nu)
        """
        return np.zeros((self.horizon, self.nu))

    def extrapolate_terminal(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
        dynamics_fn: Callable[..., np.ndarray] | None = None,
        theta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrapolate terminal state and input for shifted trajectory.

        Instead of duplicating, extrapolate using dynamics if available.

        Args:
            x_traj: Current state trajectory (N+1, nx)
            u_traj: Current input trajectory (N, nu)
            dynamics_fn: Optional dynamics function f(x, u, theta) -> x_next
            theta: Parameters for dynamics function

        Returns:
            Tuple of (x_shifted, u_shifted) with extrapolated terminal
        """
        N = self.horizon

        # Shift base trajectory
        x_shifted = np.zeros((N + 1, self.nx))
        x_shifted[:-1] = x_traj[1:]
        u_shifted = np.zeros((N, self.nu))
        u_shifted[:-1] = u_traj[1:]

        # Extrapolate terminal
        if dynamics_fn is not None and theta is not None:
            # Use dynamics to predict terminal state
            x_shifted[-1] = dynamics_fn(x_traj[-1], u_traj[-1], theta)
            # Keep terminal input same as second-to-last
            u_shifted[-1] = u_traj[-1]
        else:
            # Fallback: duplicate
            x_shifted[-1] = x_traj[-1]
            u_shifted[-1] = u_traj[-1]

        return x_shifted, u_shifted
