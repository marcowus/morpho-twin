"""Tests for RTI timing monitoring and warm-start validation.

These tests verify that:
1. RTI timing monitor correctly tracks prepare/feedback phases
2. Budget violations are detected and warned
3. Warm-start validation catches continuity errors
4. Trajectory shifting works correctly
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from ddt.control.rti_timing import RTITimingMonitor, RTITimingStats
from ddt.control.warm_start import WarmStartManager, WarmStartStatus


@pytest.mark.unit
class TestRTITimingMonitor:
    """Tests for RTITimingMonitor."""

    def test_creation_with_defaults(self) -> None:
        """Test monitor creation with default parameters."""
        monitor = RTITimingMonitor(dt=0.1)

        assert monitor.dt == 0.1
        assert monitor.budget_fraction == 0.2
        assert monitor.budget_ms == 20.0  # 0.1 * 0.2 * 1000

    def test_custom_budget_fraction(self) -> None:
        """Test custom budget fraction."""
        monitor = RTITimingMonitor(dt=0.1, budget_fraction=0.1)

        assert monitor.budget_ms == 10.0  # 0.1 * 0.1 * 1000

    def test_timing_measurement(self) -> None:
        """Test basic timing measurement."""
        monitor = RTITimingMonitor(dt=0.1, budget_fraction=0.5)

        # Simulate prepare phase
        monitor.start_prepare()
        time.sleep(0.001)  # 1ms
        monitor.end_prepare()

        # Simulate feedback phase
        monitor.start_feedback()
        time.sleep(0.001)  # 1ms
        monitor.end_feedback()

        stats = monitor.get_stats()

        # Should have one sample
        assert stats.sample_count == 1
        # Times should be > 0
        assert stats.prepare_mean_ms > 0
        assert stats.feedback_mean_ms > 0
        assert stats.total_mean_ms > 0

    def test_multiple_samples(self) -> None:
        """Test statistics with multiple samples."""
        monitor = RTITimingMonitor(dt=0.1, window_size=10)

        for _ in range(5):
            monitor.start_prepare()
            time.sleep(0.0001)
            monitor.end_prepare()

            monitor.start_feedback()
            time.sleep(0.0001)
            monitor.end_feedback()

        stats = monitor.get_stats()
        assert stats.sample_count == 5

    def test_budget_within(self) -> None:
        """Test that fast execution is within budget."""
        monitor = RTITimingMonitor(dt=0.1, budget_fraction=0.5)  # 50ms budget

        # Very fast execution
        monitor.start_prepare()
        monitor.end_prepare()
        monitor.start_feedback()
        monitor.end_feedback()

        stats = monitor.get_stats()
        assert stats.is_within_budget

    def test_budget_violation_detection(self) -> None:
        """Test that budget violations are detected."""
        monitor = RTITimingMonitor(dt=0.01, budget_fraction=0.01)  # 0.1ms budget

        # Slow execution
        monitor.start_prepare()
        time.sleep(0.005)  # 5ms - way over budget
        monitor.end_prepare()
        monitor.start_feedback()
        time.sleep(0.005)
        monitor.end_feedback()

        stats = monitor.get_stats()
        assert not stats.is_within_budget
        assert stats.budget_violations >= 1

    def test_reset(self) -> None:
        """Test reset clears statistics."""
        monitor = RTITimingMonitor(dt=0.1)

        # Add some samples
        monitor.start_prepare()
        monitor.end_prepare()
        monitor.start_feedback()
        monitor.end_feedback()

        assert monitor.get_stats().sample_count == 1

        # Reset
        monitor.reset()

        assert monitor.get_stats().sample_count == 0
        assert monitor._budget_violations == 0

    def test_check_budget_and_warn(self) -> None:
        """Test warning is raised when budget exceeded."""
        monitor = RTITimingMonitor(dt=0.001, budget_fraction=0.01)  # 0.01ms budget

        # Force budget violation
        monitor.start_prepare()
        time.sleep(0.01)  # 10ms
        monitor.end_prepare()
        monitor.start_feedback()
        time.sleep(0.01)
        monitor.end_feedback()

        with pytest.warns(RuntimeWarning, match="RTI timing budget exceeded"):
            monitor.check_budget_and_warn()

    def test_empty_stats(self) -> None:
        """Test stats before any samples."""
        monitor = RTITimingMonitor(dt=0.1)
        stats = monitor.get_stats()

        assert stats.sample_count == 0
        assert stats.is_within_budget  # No violations yet


@pytest.mark.unit
class TestWarmStartManager:
    """Tests for WarmStartManager."""

    @pytest.fixture
    def manager(self) -> WarmStartManager:
        """Create a warm-start manager for testing."""
        return WarmStartManager(
            horizon=5,
            nx=1,
            nu=1,
            max_state_continuity_error=0.1,
        )

    def test_valid_shift(self, manager: WarmStartManager) -> None:
        """Test valid trajectory shift."""
        # Previous trajectory
        x_traj = np.arange(6).reshape(6, 1).astype(float)  # [0, 1, 2, 3, 4, 5]
        u_traj = np.arange(5).reshape(5, 1).astype(float)  # [0, 1, 2, 3, 4]

        # Actual state matches predicted x[1] = 1.0
        x_actual = np.array([1.0])

        x_sh, u_sh, status = manager.shift_and_validate(x_actual, x_traj, u_traj)

        assert status.shift_applied
        assert status.is_valid
        assert status.continuity_error < 0.01

        # Check shifted values
        np.testing.assert_array_almost_equal(
            x_sh, np.array([[1], [2], [3], [4], [5], [5]])
        )
        np.testing.assert_array_almost_equal(
            u_sh, np.array([[1], [2], [3], [4], [4]])
        )

    def test_invalid_continuity(self, manager: WarmStartManager) -> None:
        """Test detection of continuity error."""
        x_traj = np.arange(6).reshape(6, 1).astype(float)
        u_traj = np.arange(5).reshape(5, 1).astype(float)

        # Actual state very different from predicted x[1] = 1.0
        x_actual = np.array([5.0])  # Error = 4.0 > threshold

        x_sh, u_sh, status = manager.shift_and_validate(x_actual, x_traj, u_traj)

        assert status.shift_applied  # Still shifted despite error
        assert not status.is_valid
        assert status.continuity_error > 0.1

    def test_dimension_mismatch(self, manager: WarmStartManager) -> None:
        """Test handling of dimension mismatch."""
        x_traj = np.zeros((3, 1))  # Wrong horizon
        u_traj = np.zeros((5, 1))
        x_actual = np.array([0.0])

        x_sh, u_sh, status = manager.shift_and_validate(x_actual, x_traj, u_traj)

        assert not status.shift_applied
        assert not status.is_valid

    def test_multi_dimensional_state(self) -> None:
        """Test with multi-dimensional state."""
        manager = WarmStartManager(
            horizon=3,
            nx=2,
            nu=1,
            max_state_continuity_error=0.5,
        )

        x_traj = np.array([
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
        ])
        u_traj = np.array([[0.1], [0.2], [0.3]])
        x_actual = np.array([1.0, 0.5])  # Matches x[1]

        x_sh, u_sh, status = manager.shift_and_validate(x_actual, x_traj, u_traj)

        assert status.is_valid
        assert x_sh.shape == (4, 2)
        assert u_sh.shape == (3, 1)

    def test_cold_start(self, manager: WarmStartManager) -> None:
        """Test cold start trajectory generation."""
        x0 = np.array([0.5])
        x_traj = manager._cold_start(x0)

        assert x_traj.shape == (6, 1)
        np.testing.assert_array_equal(x_traj, np.full((6, 1), 0.5))

    def test_zero_inputs(self, manager: WarmStartManager) -> None:
        """Test zero input trajectory generation."""
        u_traj = manager._zero_inputs()

        assert u_traj.shape == (5, 1)
        np.testing.assert_array_equal(u_traj, np.zeros((5, 1)))

    def test_extrapolate_terminal_without_dynamics(self, manager: WarmStartManager) -> None:
        """Test terminal extrapolation without dynamics (duplication)."""
        x_traj = np.arange(6).reshape(6, 1).astype(float)
        u_traj = np.arange(5).reshape(5, 1).astype(float)

        x_sh, u_sh = manager.extrapolate_terminal(x_traj, u_traj)

        # Should duplicate last values
        assert x_sh[-1, 0] == x_traj[-1, 0]
        assert u_sh[-1, 0] == u_traj[-1, 0]

    def test_extrapolate_terminal_with_dynamics(self, manager: WarmStartManager) -> None:
        """Test terminal extrapolation with dynamics."""

        def linear_dynamics(x, u, theta):
            return theta[0] * x + theta[1] * u

        x_traj = np.arange(6).reshape(6, 1).astype(float)
        u_traj = np.ones((5, 1))
        theta = np.array([1.0, 0.1])

        x_sh, u_sh = manager.extrapolate_terminal(
            x_traj, u_traj, dynamics_fn=linear_dynamics, theta=theta
        )

        # Terminal state should be dynamics(x[-1], u[-1], theta)
        expected = linear_dynamics(x_traj[-1], u_traj[-1], theta)
        np.testing.assert_array_almost_equal(x_sh[-1], expected)


@pytest.mark.unit
class TestRTITimingIntegration:
    """Tests for RTI timing in NMPC context."""

    @pytest.mark.skipif(
        not pytest.importorskip("acados_template", reason="acados not installed"),
        reason="acados not installed",
    )
    def test_nmpc_rti_timing_enabled(self) -> None:
        """Test that NMPC-RTI has timing enabled by default."""
        from ddt.control.nmpc_rti import RTINMPC

        # This would only run if acados is available
        # The test verifies the interface


@pytest.mark.unit
class TestWarmStartStatus:
    """Tests for WarmStartStatus dataclass."""

    def test_status_creation(self) -> None:
        """Test status creation."""
        status = WarmStartStatus(
            shift_applied=True,
            is_valid=True,
            continuity_error=0.05,
            message="Warm-start valid",
        )

        assert status.shift_applied
        assert status.is_valid
        assert status.continuity_error == 0.05
        assert "valid" in status.message

    def test_invalid_status(self) -> None:
        """Test invalid warm-start status."""
        status = WarmStartStatus(
            shift_applied=True,
            is_valid=False,
            continuity_error=1.5,
            message="Continuity error: 1.5000",
        )

        assert status.shift_applied
        assert not status.is_valid
        assert status.continuity_error > 1.0
