"""Unified supervisor interface."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ..estimation.innovation_monitor import UncertaintyValidation
from ..events import FailureSeverity, SolverFailureEvent
from .mode_manager import ModeConfig, ModeManager, OperationMode
from .pe_monitor import PEMonitor, PEStatus

if TYPE_CHECKING:
    from ..interfaces import Estimate


@dataclass(frozen=True)
class SupervisorState:
    """Current supervisor state snapshot."""

    mode: OperationMode
    pe_status: PEStatus
    safety_margin_factor: float
    uncertainty_norm: float
    solver_failure_count: int = 0
    recent_failures: tuple[SolverFailureEvent, ...] = ()
    nis_margin_multiplier: float = 1.0
    uncertainty_validation: UncertaintyValidation | None = None


@dataclass
class Supervisor:
    """Unified supervisor for Morpho Twin system.

    Integrates:
    - PE monitoring for parameter identifiability
    - Mode management for safety margin adjustment
    - Recommended probing weight for dual-control

    The supervisor observes estimation and control state and
    adjusts the safety filter margins accordingly.
    """

    # Configuration
    pe_window: int = 100
    pe_lambda_threshold: float = 0.1
    ntheta: int = 2
    mode_config: ModeConfig = field(default_factory=ModeConfig)
    solver_failure_escalation_threshold: int = 5  # Escalate after N consecutive failures

    # Components
    _pe_monitor: PEMonitor = field(init=False)
    _mode_manager: ModeManager = field(init=False)
    _last_state: SupervisorState | None = field(default=None, init=False)
    _solver_failure_count: int = field(default=0, init=False)
    _recent_failures: list[SolverFailureEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._pe_monitor = PEMonitor(
            window=self.pe_window,
            lambda_threshold=self.pe_lambda_threshold,
            ntheta=self.ntheta,
        )
        self._mode_manager = ModeManager(config=self.mode_config)

    def reset(self) -> None:
        """Reset supervisor state."""
        self._pe_monitor.reset()
        self._mode_manager.reset()
        self._last_state = None
        self._solver_failure_count = 0
        self._recent_failures.clear()

    def update(
        self,
        estimate: Estimate,
        regressor: np.ndarray | None = None,
        solver_events: list[SolverFailureEvent] | None = None,
        uncertainty_validation: UncertaintyValidation | None = None,
    ) -> SupervisorState:
        """Update supervisor with new estimation data.

        Args:
            estimate: Current state and parameter estimate
            regressor: Optional regressor for PE monitoring
                If None, uses default [x_hat, 1] for scalar systems
            solver_events: Optional list of solver failure events from this step
            uncertainty_validation: Optional NIS-based uncertainty validation result

        Returns:
            Current supervisor state
        """
        # Process solver failure events
        if solver_events:
            for event in solver_events:
                self.report_solver_failure(event)

        # Compute uncertainty norm
        theta_cov = np.atleast_2d(estimate.theta_cov)
        uncertainty_norm = float(np.trace(theta_cov))

        # Update PE monitor
        if regressor is None:
            # Default regressor for linear scalar system (suboptimal - uses [x, 1])
            # For proper PE monitoring, pass regressor=[x_hat, u_safe] explicitly
            warnings.warn(
                "No regressor provided to supervisor. Using [x, 1] which does not "
                "properly monitor control excitation. Pass regressor=[x_hat, u_safe] "
                "for correct PE monitoring.",
                stacklevel=2,
            )
            x = np.atleast_1d(estimate.x_hat)
            regressor = np.concatenate([x, np.ones(1)])

        pe_status = self._pe_monitor.update(regressor)

        # Update mode manager
        mode = self._mode_manager.update(
            uncertainty_norm=uncertainty_norm,
            is_pe_satisfied=pe_status.is_pe_satisfied,
        )

        # NIS-based margin adjustment
        nis_multiplier = 1.0
        if uncertainty_validation is not None and not uncertainty_validation.is_valid:
            nis_multiplier = uncertainty_validation.recommended_margin_multiplier
            if uncertainty_validation.warning_message:
                warnings.warn(uncertainty_validation.warning_message, RuntimeWarning, stacklevel=2)

        # Combine mode margin with NIS multiplier
        combined_margin = self._mode_manager.safety_margin_factor * nis_multiplier

        # Build state snapshot
        state = SupervisorState(
            mode=mode,
            pe_status=pe_status,
            safety_margin_factor=combined_margin,
            uncertainty_norm=uncertainty_norm,
            solver_failure_count=self._solver_failure_count,
            recent_failures=tuple(self._recent_failures[-5:]),  # Keep last 5
            nis_margin_multiplier=nis_multiplier,
            uncertainty_validation=uncertainty_validation,
        )

        self._last_state = state
        return state

    def report_solver_failure(self, event: SolverFailureEvent) -> None:
        """Report a solver failure event.

        Handles escalation based on severity and accumulated failures.

        Args:
            event: The failure event to report
        """
        self._solver_failure_count += 1
        self._recent_failures.append(event)

        # Keep only recent failures (last 10)
        if len(self._recent_failures) > 10:
            self._recent_failures.pop(0)

        # CBF failures are CRITICAL - immediate escalation to SAFE_STOP
        if event.severity == FailureSeverity.CRITICAL:
            logger.error(
                "CRITICAL failure from {} - triggering safe stop | message={}",
                event.component.name,
                event.message,
            )
            self.trigger_safe_stop()
            return

        # Accumulating failures trigger mode escalation
        if self._solver_failure_count >= self.solver_failure_escalation_threshold:
            current_mode = self._mode_manager.mode

            if current_mode == OperationMode.NORMAL:
                logger.warning(
                    "Solver failures exceeded threshold ({}) - escalating to CONSERVATIVE",
                    self._solver_failure_count,
                )
                self._mode_manager.trigger_conservative()
            elif current_mode == OperationMode.CONSERVATIVE:
                logger.error(
                    "Solver failures exceeded threshold ({}) in CONSERVATIVE mode "
                    "- escalating to SAFE_STOP",
                    self._solver_failure_count,
                )
                self.trigger_safe_stop()

    def clear_failure_count(self) -> None:
        """Clear the accumulated failure count.

        Call this after successful recovery or manual intervention.
        """
        self._solver_failure_count = 0

    def get_state(self) -> SupervisorState | None:
        """Get last supervisor state."""
        return self._last_state

    @property
    def mode(self) -> OperationMode:
        """Current operation mode."""
        return self._mode_manager.mode

    @property
    def safety_margin_factor(self) -> float:
        """Current safety margin factor."""
        return self._mode_manager.safety_margin_factor

    def get_recommended_probe_weight(self) -> float:
        """Get recommended probing weight for dual-control."""
        if self._last_state is None:
            return 0.0
        return self._last_state.pe_status.recommended_probe_weight

    def is_safe_to_operate(self) -> bool:
        """Check if system is safe to operate."""
        return self._mode_manager.mode != OperationMode.SAFE_STOP

    def trigger_safe_stop(self) -> None:
        """Explicitly trigger safe stop mode."""
        self._mode_manager.trigger_safe_stop()

    def get_zero_control(self, nu: int = 1) -> np.ndarray:
        """Get zero control for safe stop mode.

        In SAFE_STOP mode, the system should apply zero or
        minimum control to bring the system to rest.

        Args:
            nu: Input dimension

        Returns:
            Zero control vector
        """
        return np.zeros(nu)


def create_supervisor(
    ntheta: int = 2,
    pe_window: int = 100,
    pe_threshold: float = 0.1,
    solver_failure_threshold: int = 5,
    **mode_kwargs: Any,
) -> Supervisor:
    """Factory function for creating a supervisor.

    Args:
        ntheta: Number of parameters
        pe_window: Rolling window for PE monitoring
        pe_threshold: Minimum eigenvalue threshold for PE
        solver_failure_threshold: Consecutive failures before mode escalation
        **mode_kwargs: Additional arguments for ModeConfig

    Returns:
        Configured Supervisor instance
    """
    mode_config = ModeConfig(**mode_kwargs)

    return Supervisor(
        pe_window=pe_window,
        pe_lambda_threshold=pe_threshold,
        ntheta=ntheta,
        mode_config=mode_config,
        solver_failure_escalation_threshold=solver_failure_threshold,
    )
