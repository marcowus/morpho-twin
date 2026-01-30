"""Operation mode manager with state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class OperationMode(Enum):
    """Operating modes for the Morpho Twin system."""

    NORMAL = auto()  # Standard operation
    CONSERVATIVE = auto()  # Increased safety margins
    SAFE_STOP = auto()  # Emergency stop


@dataclass
class ModeConfig:
    """Configuration for mode transitions."""

    # Uncertainty thresholds
    uncertainty_normal_to_conservative: float = 1.0
    uncertainty_conservative_to_safe: float = 5.0
    uncertainty_safe_to_conservative: float = 3.0
    uncertainty_conservative_to_normal: float = 0.5

    # PE violation thresholds
    pe_violation_count_conservative: int = 10
    pe_violation_count_safe: int = 50

    # Recovery requirements
    pe_satisfaction_count_recovery: int = 20

    # Safety margin factors
    margin_factor_normal: float = 1.0
    margin_factor_conservative: float = 2.0
    margin_factor_safe: float = 5.0


@dataclass
class ModeManager:
    """State machine for operation mode management.

    Transitions:
        NORMAL ←→ CONSERVATIVE ←→ SAFE_STOP

    Transitions are based on:
    - Uncertainty threshold (trace of θ_cov)
    - PE violation count
    - Explicit triggers
    """

    config: ModeConfig = field(default_factory=ModeConfig)

    # State
    _mode: OperationMode = field(default=OperationMode.NORMAL, init=False)
    _pe_violation_count: int = field(default=0, init=False)
    _pe_satisfaction_count: int = field(default=0, init=False)
    _mode_history: list[OperationMode] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Reset mode manager to initial state."""
        self._mode = OperationMode.NORMAL
        self._pe_violation_count = 0
        self._pe_satisfaction_count = 0
        self._mode_history.clear()

    @property
    def mode(self) -> OperationMode:
        """Current operation mode."""
        return self._mode

    @property
    def safety_margin_factor(self) -> float:
        """Get safety margin factor for current mode."""
        if self._mode == OperationMode.NORMAL:
            return self.config.margin_factor_normal
        elif self._mode == OperationMode.CONSERVATIVE:
            return self.config.margin_factor_conservative
        else:  # SAFE_STOP
            return self.config.margin_factor_safe

    def update(
        self,
        uncertainty_norm: float,
        is_pe_satisfied: bool,
    ) -> OperationMode:
        """Update mode based on current conditions.

        Args:
            uncertainty_norm: Trace of parameter covariance
            is_pe_satisfied: Whether PE condition is satisfied

        Returns:
            New operation mode
        """
        old_mode = self._mode

        # Update PE counters
        if is_pe_satisfied:
            self._pe_satisfaction_count += 1
            self._pe_violation_count = max(0, self._pe_violation_count - 1)
        else:
            self._pe_violation_count += 1
            self._pe_satisfaction_count = 0

        # State machine transitions
        if self._mode == OperationMode.NORMAL:
            self._mode = self._transition_from_normal(uncertainty_norm)
        elif self._mode == OperationMode.CONSERVATIVE:
            self._mode = self._transition_from_conservative(uncertainty_norm)
        elif self._mode == OperationMode.SAFE_STOP:
            self._mode = self._transition_from_safe_stop(uncertainty_norm)

        # Track history
        if self._mode != old_mode:
            self._mode_history.append(self._mode)

        return self._mode

    def _transition_from_normal(self, uncertainty: float) -> OperationMode:
        """Determine transition from NORMAL mode."""
        cfg = self.config

        # Transition to CONSERVATIVE
        if uncertainty > cfg.uncertainty_normal_to_conservative:
            return OperationMode.CONSERVATIVE

        if self._pe_violation_count >= cfg.pe_violation_count_conservative:
            return OperationMode.CONSERVATIVE

        return OperationMode.NORMAL

    def _transition_from_conservative(self, uncertainty: float) -> OperationMode:
        """Determine transition from CONSERVATIVE mode."""
        cfg = self.config

        # Transition to SAFE_STOP
        if uncertainty > cfg.uncertainty_conservative_to_safe:
            return OperationMode.SAFE_STOP

        if self._pe_violation_count >= cfg.pe_violation_count_safe:
            return OperationMode.SAFE_STOP

        # Transition back to NORMAL
        if (
            uncertainty < cfg.uncertainty_conservative_to_normal
            and self._pe_satisfaction_count >= cfg.pe_satisfaction_count_recovery
        ):
            return OperationMode.NORMAL

        return OperationMode.CONSERVATIVE

    def _transition_from_safe_stop(self, uncertainty: float) -> OperationMode:
        """Determine transition from SAFE_STOP mode."""
        cfg = self.config

        # Transition to CONSERVATIVE (recovery)
        if (
            uncertainty < cfg.uncertainty_safe_to_conservative
            and self._pe_satisfaction_count >= cfg.pe_satisfaction_count_recovery
        ):
            return OperationMode.CONSERVATIVE

        return OperationMode.SAFE_STOP

    def trigger_safe_stop(self) -> None:
        """Explicitly trigger SAFE_STOP mode."""
        self._mode = OperationMode.SAFE_STOP
        self._mode_history.append(self._mode)

    def trigger_conservative(self) -> None:
        """Explicitly trigger CONSERVATIVE mode."""
        if self._mode == OperationMode.NORMAL:
            self._mode = OperationMode.CONSERVATIVE
            self._mode_history.append(self._mode)

    def get_mode_duration(self) -> int:
        """Get number of steps in current mode."""
        if not self._mode_history:
            return 0

        count = 0
        for mode in reversed(self._mode_history):
            if mode == self._mode:
                count += 1
            else:
                break
        return max(count, 1)
