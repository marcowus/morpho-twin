"""Structured events for failure handling and monitoring.

This module defines event types for reporting solver failures, warnings,
and other runtime events that require supervisor attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class FailureSeverity(Enum):
    """Severity levels for solver failures."""

    WARNING = auto()  # Recoverable, degraded operation
    ERROR = auto()  # Significant degradation, may need intervention
    CRITICAL = auto()  # Immediate supervisor action required


class ComponentType(Enum):
    """Types of system components that can generate events."""

    CBF_QP = auto()  # Control Barrier Function QP safety filter
    NMPC_CASADI = auto()  # CasADi-based NMPC
    NMPC_RTI = auto()  # RTI-based NMPC (acados)
    MHE_CASADI = auto()  # CasADi-based MHE
    MHE_ACADOS = auto()  # acados-based MHE
    SUPERVISOR = auto()  # Supervisor itself


@dataclass(frozen=True)
class SolverFailureEvent:
    """Structured event for solver failures.

    Attributes:
        component: Which component generated the failure
        severity: How severe the failure is
        message: Human-readable description
        solver_status: Raw status from the solver (if available)
        fallback_action: What fallback action was taken
        iteration_count: Number of solver iterations (if applicable)
        residual: Solver residual at termination (if applicable)
    """

    component: ComponentType
    severity: FailureSeverity
    message: str
    solver_status: str | None = None
    fallback_action: str = "none"
    iteration_count: int | None = None
    residual: float | None = None

    def __str__(self) -> str:
        """Format event as string for logging."""
        parts = [
            f"[{self.severity.name}]",
            f"{self.component.name}:",
            self.message,
        ]
        if self.solver_status:
            parts.append(f"(status={self.solver_status})")
        if self.fallback_action != "none":
            parts.append(f"-> {self.fallback_action}")
        return " ".join(parts)


@dataclass(frozen=True)
class TimingEvent:
    """Event for RTI timing budget violations.

    Attributes:
        component: Which component generated the event
        phase: Which phase exceeded budget (prepare/feedback)
        elapsed_ms: Actual elapsed time in milliseconds
        budget_ms: Budget time in milliseconds
        message: Human-readable description
    """

    component: ComponentType
    phase: str
    elapsed_ms: float
    budget_ms: float
    message: str

    @property
    def severity(self) -> FailureSeverity:
        """Timing violations are warnings unless extreme."""
        if self.elapsed_ms > 2 * self.budget_ms:
            return FailureSeverity.ERROR
        return FailureSeverity.WARNING


@dataclass(frozen=True)
class UncertaintyEvent:
    """Event for uncertainty-related issues.

    Attributes:
        component: Which component generated the event
        severity: How severe the issue is
        message: Human-readable description
        nis_value: Normalized Innovation Squared value (if applicable)
        recommended_margin_multiplier: Suggested safety margin adjustment
    """

    component: ComponentType
    severity: FailureSeverity
    message: str
    nis_value: float | None = None
    recommended_margin_multiplier: float = 1.0
