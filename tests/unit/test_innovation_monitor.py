"""Tests for innovation monitoring and NIS-based uncertainty validation.

These tests verify that:
1. NIS is correctly computed from innovations
2. Consistent covariance is detected
3. Optimistic covariance (too small) triggers margin adjustment
4. Pessimistic covariance (too large) is detected but doesn't trigger action
5. Supervisor correctly integrates NIS-based margin adjustment
"""

from __future__ import annotations

import numpy as np
import pytest

from ddt.estimation.innovation_monitor import (
    InnovationMonitor,
    InnovationStats,
    UncertaintyValidation,
)


@pytest.mark.unit
class TestInnovationMonitor:
    """Tests for InnovationMonitor."""

    def test_creation_with_defaults(self) -> None:
        """Test monitor creation with default parameters."""
        monitor = InnovationMonitor(ny=1)

        assert monitor.ny == 1
        assert monitor.window_size == 50
        assert monitor.nis_high_threshold == 2.0
        assert monitor.nis_low_threshold == 0.3

    def test_custom_parameters(self) -> None:
        """Test monitor creation with custom parameters."""
        R_diag = np.array([0.01])
        monitor = InnovationMonitor(
            ny=1,
            window_size=100,
            nis_high_threshold=3.0,
            nis_low_threshold=0.2,
            R_diag=R_diag,
        )

        assert monitor.window_size == 100
        assert monitor.nis_high_threshold == 3.0
        np.testing.assert_array_equal(monitor.R_diag, R_diag)

    def test_nis_computation_scalar(self) -> None:
        """Test NIS computation for scalar measurements."""
        R_diag = np.array([1.0])  # Unit variance
        monitor = InnovationMonitor(ny=1, R_diag=R_diag)

        # Innovation = 1.0, with R=1.0, NIS = 1^2 / 1 = 1.0
        y_actual = np.array([1.0])
        y_predicted = np.array([0.0])

        stats = monitor.update(y_actual, y_predicted)

        assert stats.sample_count == 1
        assert abs(stats.nis_mean - 1.0) < 0.01

    def test_nis_computation_vector(self) -> None:
        """Test NIS computation for vector measurements."""
        R_diag = np.array([1.0, 4.0])  # Different variances
        monitor = InnovationMonitor(ny=2, R_diag=R_diag)

        # Innovation = [1.0, 2.0]
        # NIS = 1^2/1 + 2^2/4 = 1 + 1 = 2.0
        y_actual = np.array([1.0, 2.0])
        y_predicted = np.array([0.0, 0.0])

        stats = monitor.update(y_actual, y_predicted)

        assert abs(stats.nis_mean - 2.0) < 0.01
        assert abs(stats.normalized_nis - 1.0) < 0.01  # 2/2 = 1

    def test_consistent_covariance(self) -> None:
        """Test detection of consistent covariance."""
        R_diag = np.array([1.0])
        monitor = InnovationMonitor(ny=1, R_diag=R_diag)

        # Generate innovations consistent with R=1
        np.random.seed(42)
        for _ in range(50):
            innovation = np.random.randn(1) * 1.0  # std=1
            y_actual = innovation
            y_predicted = np.zeros(1)
            monitor.update(y_actual, y_predicted)

        validation = monitor.validate_uncertainty()

        # Should be valid (normalized NIS close to 1)
        assert validation.is_valid
        assert 0.5 < validation.normalized_nis < 2.0
        assert validation.recommended_margin_multiplier == 1.0

    def test_optimistic_covariance_detection(self) -> None:
        """Test detection of optimistic (too small) covariance."""
        R_diag = np.array([1.0])  # Assumed variance
        monitor = InnovationMonitor(ny=1, R_diag=R_diag, nis_high_threshold=2.0)

        # Generate innovations with larger actual variance (R=4 but assumed R=1)
        # This makes actual errors 2x larger than expected
        np.random.seed(42)
        for _ in range(50):
            innovation = np.random.randn(1) * 2.0  # std=2, but R=1 assumes std=1
            y_actual = innovation
            y_predicted = np.zeros(1)
            monitor.update(y_actual, y_predicted)

        validation = monitor.validate_uncertainty()

        # Should detect optimistic covariance
        # NIS will be ~4 (because variance is 4x but R=1)
        assert not validation.is_valid
        assert validation.normalized_nis > 2.0
        assert validation.recommended_margin_multiplier > 1.0

    def test_pessimistic_covariance_detection(self) -> None:
        """Test detection of pessimistic (too large) covariance."""
        R_diag = np.array([4.0])  # Assumed variance
        monitor = InnovationMonitor(ny=1, R_diag=R_diag, nis_low_threshold=0.3)

        # Generate innovations with smaller actual variance (R=1 but assumed R=4)
        np.random.seed(42)
        for _ in range(50):
            innovation = np.random.randn(1) * 1.0  # std=1, but R=4 assumes std=2
            y_actual = innovation
            y_predicted = np.zeros(1)
            monitor.update(y_actual, y_predicted)

        stats = monitor.get_stats()

        # NIS will be ~0.25 (because variance is 1 but R=4)
        # This is valid (not dangerous) but pessimistic
        assert stats.normalized_nis < 0.5

    def test_margin_multiplier_capped(self) -> None:
        """Test that margin multiplier is capped at 3.0."""
        R_diag = np.array([1.0])
        monitor = InnovationMonitor(ny=1, R_diag=R_diag, nis_high_threshold=2.0)

        # Generate very large innovations
        for _ in range(20):
            y_actual = np.array([10.0])  # Very large error
            y_predicted = np.zeros(1)
            monitor.update(y_actual, y_predicted)

        validation = monitor.validate_uncertainty()

        # Multiplier should be capped at 3.0
        assert validation.recommended_margin_multiplier <= 3.0

    def test_insufficient_samples(self) -> None:
        """Test behavior with insufficient samples."""
        monitor = InnovationMonitor(ny=1)

        # Only add a few samples
        for _ in range(5):
            monitor.update(np.array([1.0]), np.array([0.0]))

        validation = monitor.validate_uncertainty()

        # Should return valid since not enough samples
        assert validation.is_valid
        assert validation.recommended_margin_multiplier == 1.0

    def test_reset(self) -> None:
        """Test reset clears statistics."""
        monitor = InnovationMonitor(ny=1)

        # Add some samples
        for _ in range(20):
            monitor.update(np.array([1.0]), np.array([0.0]))

        assert monitor.get_stats().sample_count == 20

        # Reset
        monitor.reset()

        assert monitor.get_stats().sample_count == 0

    def test_with_innovation_covariance(self) -> None:
        """Test NIS computation with explicit innovation covariance S."""
        monitor = InnovationMonitor(ny=1)

        y_actual = np.array([2.0])
        y_predicted = np.array([0.0])
        S = np.array([[4.0]])  # Innovation variance = 4

        stats = monitor.update(y_actual, y_predicted, S=S)

        # NIS = 2^2 / 4 = 1.0
        assert abs(stats.nis_mean - 1.0) < 0.01


@pytest.mark.unit
class TestInnovationStats:
    """Tests for InnovationStats dataclass."""

    def test_creation(self) -> None:
        """Test stats creation."""
        stats = InnovationStats(
            nis_mean=1.5,
            nis_std=0.5,
            normalized_nis=1.5,
            sample_count=100,
            is_consistent=True,
        )

        assert stats.nis_mean == 1.5
        assert stats.normalized_nis == 1.5
        assert stats.is_consistent


@pytest.mark.unit
class TestUncertaintyValidation:
    """Tests for UncertaintyValidation dataclass."""

    def test_valid_creation(self) -> None:
        """Test valid uncertainty creation."""
        validation = UncertaintyValidation(
            is_valid=True,
            normalized_nis=1.0,
            recommended_margin_multiplier=1.0,
            warning_message="",
        )

        assert validation.is_valid
        assert validation.recommended_margin_multiplier == 1.0

    def test_invalid_creation(self) -> None:
        """Test invalid uncertainty creation."""
        validation = UncertaintyValidation(
            is_valid=False,
            normalized_nis=3.0,
            recommended_margin_multiplier=1.5,
            warning_message="NIS too high",
        )

        assert not validation.is_valid
        assert validation.recommended_margin_multiplier == 1.5
        assert "NIS" in validation.warning_message


@pytest.mark.unit
class TestSupervisorNISIntegration:
    """Tests for supervisor NIS integration."""

    def test_supervisor_applies_nis_multiplier(self) -> None:
        """Test that supervisor applies NIS-based margin multiplier."""
        from ddt.interfaces import Estimate
        from ddt.supervision.supervisor import Supervisor

        supervisor = Supervisor(
            pe_window=10,
            pe_lambda_threshold=0.01,
            ntheta=2,
        )

        estimate = Estimate(
            x_hat=np.array([0.5]),
            theta_hat=np.array([1.02, 0.10]),
            theta_cov=np.array([[0.01, 0.0], [0.0, 0.01]]),
        )
        regressor = np.array([0.5, 0.1])

        # Update without NIS validation
        state1 = supervisor.update(estimate, regressor)
        base_margin = state1.safety_margin_factor

        # Update with NIS validation indicating optimistic covariance
        nis_validation = UncertaintyValidation(
            is_valid=False,
            normalized_nis=3.0,
            recommended_margin_multiplier=1.5,
            warning_message="NIS test failed",
        )

        with pytest.warns(RuntimeWarning, match="NIS test failed"):
            state2 = supervisor.update(estimate, regressor, uncertainty_validation=nis_validation)

        # Margin should be increased by NIS multiplier
        assert state2.safety_margin_factor == base_margin * 1.5
        assert state2.nis_margin_multiplier == 1.5

    def test_supervisor_stores_uncertainty_validation(self) -> None:
        """Test that supervisor stores uncertainty validation in state."""
        from ddt.interfaces import Estimate
        from ddt.supervision.supervisor import Supervisor

        supervisor = Supervisor(pe_window=10, pe_lambda_threshold=0.01, ntheta=2)

        estimate = Estimate(
            x_hat=np.array([0.5]),
            theta_hat=np.array([1.02, 0.10]),
            theta_cov=np.array([[0.01, 0.0], [0.0, 0.01]]),
        )
        regressor = np.array([0.5, 0.1])

        nis_validation = UncertaintyValidation(
            is_valid=True,
            normalized_nis=1.0,
            recommended_margin_multiplier=1.0,
        )

        state = supervisor.update(estimate, regressor, uncertainty_validation=nis_validation)

        assert state.uncertainty_validation is not None
        assert state.uncertainty_validation.is_valid
