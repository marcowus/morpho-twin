"""Innovation monitoring for uncertainty validation.

This module implements Normalized Innovation Squared (NIS) testing
to validate that the estimator's uncertainty (covariance) is consistent
with actual prediction errors.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class InnovationStats:
    """Statistics from innovation monitoring.

    Attributes:
        nis_mean: Mean Normalized Innovation Squared
        nis_std: Standard deviation of NIS
        normalized_nis: NIS divided by output dimension (should be ~1.0)
        sample_count: Number of samples collected
        is_consistent: Whether covariance is consistent with innovations
    """

    nis_mean: float
    nis_std: float
    normalized_nis: float
    sample_count: int
    is_consistent: bool


@dataclass(frozen=True)
class UncertaintyValidation:
    """Result of uncertainty validation.

    Attributes:
        is_valid: Whether the uncertainty estimate is valid
        normalized_nis: The normalized NIS value
        recommended_margin_multiplier: Suggested safety margin adjustment
        warning_message: Human-readable warning if not valid
    """

    is_valid: bool
    normalized_nis: float
    recommended_margin_multiplier: float
    warning_message: str = ""


@dataclass
class InnovationMonitor:
    """Monitor for innovation-based uncertainty validation.

    Uses Normalized Innovation Squared (NIS) testing to validate
    that the estimator's covariance is consistent with actual
    prediction errors.

    For a well-tuned estimator:
    - NIS should follow a chi-squared distribution
    - Mean NIS / ny should be approximately 1.0
    - NIS > 2*ny indicates optimistic covariance (underestimates uncertainty)
    - NIS < 0.5*ny indicates pessimistic covariance (overestimates uncertainty)

    Attributes:
        ny: Output dimension
        window_size: Number of samples for statistics
        nis_high_threshold: NIS/ny threshold for "too optimistic"
        nis_low_threshold: NIS/ny threshold for "too pessimistic"
        R_diag: Measurement noise covariance diagonal
    """

    ny: int = 1
    window_size: int = 50
    nis_high_threshold: float = 2.0  # NIS/ny > 2 = too optimistic
    nis_low_threshold: float = 0.3  # NIS/ny < 0.3 = too pessimistic
    R_diag: np.ndarray | None = None

    _nis_values: deque[float] = field(init=False)
    _R_inv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._nis_values = deque(maxlen=self.window_size)

        if self.R_diag is None:
            self._R_inv = np.eye(self.ny)
        else:
            R_diag = np.atleast_1d(self.R_diag)
            self._R_inv = np.diag(1.0 / (R_diag + 1e-10))

    def reset(self) -> None:
        """Reset the monitor."""
        self._nis_values.clear()

    def update(
        self,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        S: np.ndarray | None = None,
    ) -> InnovationStats:
        """Update with new innovation.

        Args:
            y_actual: Actual measurement (ny,)
            y_predicted: Predicted measurement (ny,)
            S: Innovation covariance (ny, ny). If None, uses R.

        Returns:
            Current innovation statistics
        """
        y_actual = np.atleast_1d(y_actual)
        y_predicted = np.atleast_1d(y_predicted)

        # Innovation
        innovation = y_actual - y_predicted

        # Innovation covariance
        if S is not None:
            S = np.atleast_2d(S)
            try:
                S_inv = np.linalg.inv(S + 1e-10 * np.eye(self.ny))
            except np.linalg.LinAlgError:
                S_inv = self._R_inv
        else:
            S_inv = self._R_inv

        # Normalized Innovation Squared: v' * S^{-1} * v
        nis = float(innovation.T @ S_inv @ innovation)
        self._nis_values.append(nis)

        return self._compute_stats()

    def _compute_stats(self) -> InnovationStats:
        """Compute current statistics."""
        if len(self._nis_values) == 0:
            return InnovationStats(
                nis_mean=0.0,
                nis_std=0.0,
                normalized_nis=0.0,
                sample_count=0,
                is_consistent=True,
            )

        nis_arr = np.array(self._nis_values)
        nis_mean = float(np.mean(nis_arr))
        nis_std = float(np.std(nis_arr))
        normalized_nis = nis_mean / self.ny

        # Check consistency
        is_consistent = (
            self.nis_low_threshold <= normalized_nis <= self.nis_high_threshold
        )

        return InnovationStats(
            nis_mean=nis_mean,
            nis_std=nis_std,
            normalized_nis=normalized_nis,
            sample_count=len(self._nis_values),
            is_consistent=is_consistent,
        )

    def validate_uncertainty(self) -> UncertaintyValidation:
        """Validate uncertainty estimate and recommend adjustments.

        Returns:
            UncertaintyValidation with status and recommendations
        """
        stats = self._compute_stats()

        # Not enough samples yet
        if stats.sample_count < 10:
            return UncertaintyValidation(
                is_valid=True,
                normalized_nis=stats.normalized_nis,
                recommended_margin_multiplier=1.0,
                warning_message="",
            )

        # Check if covariance is too optimistic (underestimates uncertainty)
        if stats.normalized_nis > self.nis_high_threshold:
            # NIS too high - actual errors larger than covariance suggests
            # Recommend increasing safety margins
            margin_multiplier = min(stats.normalized_nis / self.nis_high_threshold, 3.0)

            warning_msg = (
                f"NIS test failed: normalized_nis={stats.normalized_nis:.2f} > "
                f"{self.nis_high_threshold} - covariance too optimistic. "
                f"Recommend margin multiplier: {margin_multiplier:.2f}"
            )

            logger.warning(warning_msg)
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

            return UncertaintyValidation(
                is_valid=False,
                normalized_nis=stats.normalized_nis,
                recommended_margin_multiplier=margin_multiplier,
                warning_message=warning_msg,
            )

        # Check if covariance is too pessimistic (overestimates uncertainty)
        if stats.normalized_nis < self.nis_low_threshold:
            # NIS too low - actual errors smaller than covariance suggests
            # This is less dangerous but worth noting
            logger.info(
                "NIS low: normalized_nis={:.2f} < {} - covariance may be pessimistic",
                stats.normalized_nis,
                self.nis_low_threshold,
            )

        # Consistent uncertainty
        return UncertaintyValidation(
            is_valid=True,
            normalized_nis=stats.normalized_nis,
            recommended_margin_multiplier=1.0,
            warning_message="",
        )

    def get_stats(self) -> InnovationStats:
        """Get current innovation statistics."""
        return self._compute_stats()
