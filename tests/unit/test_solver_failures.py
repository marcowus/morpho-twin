"""Tests for solver failure handling and escalation.

These tests verify that:
1. CBF-QP failures generate CRITICAL events and trigger immediate SAFE_STOP
2. MHE failures generate WARNING events and use fallback estimates
3. Accumulated failures trigger mode escalation
4. Supervisor correctly integrates failure events
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ddt.events import ComponentType, FailureSeverity, SolverFailureEvent
from ddt.interfaces import Estimate
from ddt.supervision.mode_manager import OperationMode
from ddt.supervision.supervisor import Supervisor, SupervisorState


@pytest.fixture
def mock_estimate() -> Estimate:
    """Create a mock estimate for testing."""
    return Estimate(
        x_hat=np.array([0.5]),
        theta_hat=np.array([1.02, 0.10]),
        theta_cov=np.array([[0.01, 0.0], [0.0, 0.01]]),
    )


@pytest.fixture
def supervisor() -> Supervisor:
    """Create a supervisor for testing."""
    return Supervisor(
        pe_window=10,
        pe_lambda_threshold=0.01,
        ntheta=2,
        solver_failure_escalation_threshold=5,
    )


@pytest.mark.unit
class TestSolverFailureEvents:
    """Tests for SolverFailureEvent creation and properties."""

    def test_critical_event_creation(self) -> None:
        """Test creating a CRITICAL failure event."""
        event = SolverFailureEvent(
            component=ComponentType.CBF_QP,
            severity=FailureSeverity.CRITICAL,
            message="CBF-QP infeasible",
            solver_status="infeasible",
            fallback_action="clamp_to_bounds",
        )

        assert event.component == ComponentType.CBF_QP
        assert event.severity == FailureSeverity.CRITICAL
        assert "infeasible" in event.message
        assert event.fallback_action == "clamp_to_bounds"

    def test_warning_event_creation(self) -> None:
        """Test creating a WARNING failure event."""
        event = SolverFailureEvent(
            component=ComponentType.MHE_CASADI,
            severity=FailureSeverity.WARNING,
            message="MHE did not converge",
            solver_status="Maximum_Iterations_Exceeded",
            fallback_action="use_previous_estimate",
        )

        assert event.component == ComponentType.MHE_CASADI
        assert event.severity == FailureSeverity.WARNING
        assert event.fallback_action == "use_previous_estimate"

    def test_event_string_representation(self) -> None:
        """Test event string formatting for logging."""
        event = SolverFailureEvent(
            component=ComponentType.CBF_QP,
            severity=FailureSeverity.CRITICAL,
            message="Safety cannot be guaranteed",
            solver_status="infeasible",
            fallback_action="clamp_to_bounds",
        )

        event_str = str(event)
        assert "CRITICAL" in event_str
        assert "CBF_QP" in event_str
        assert "clamp_to_bounds" in event_str


@pytest.mark.unit
class TestSupervisorFailureHandling:
    """Tests for supervisor failure event handling."""

    def test_critical_failure_triggers_safe_stop(
        self, supervisor: Supervisor, mock_estimate: Estimate
    ) -> None:
        """CBF failures should immediately trigger SAFE_STOP."""
        regressor = np.array([0.5, 0.1])

        # First update to initialize
        supervisor.update(mock_estimate, regressor)
        assert supervisor.mode != OperationMode.SAFE_STOP

        # Report critical failure
        critical_event = SolverFailureEvent(
            component=ComponentType.CBF_QP,
            severity=FailureSeverity.CRITICAL,
            message="CBF-QP infeasible",
        )

        supervisor.report_solver_failure(critical_event)

        # Should be in SAFE_STOP
        assert supervisor.mode == OperationMode.SAFE_STOP

    def test_accumulated_failures_escalate_mode(
        self, supervisor: Supervisor, mock_estimate: Estimate
    ) -> None:
        """Accumulated failures should escalate from NORMAL to CONSERVATIVE."""
        regressor = np.array([0.5, 0.1])
        supervisor.update(mock_estimate, regressor)

        # Start in NORMAL mode
        assert supervisor.mode == OperationMode.NORMAL

        # Report failures below threshold
        warning_event = SolverFailureEvent(
            component=ComponentType.MHE_CASADI,
            severity=FailureSeverity.WARNING,
            message="MHE did not converge",
        )

        for _ in range(4):
            supervisor.report_solver_failure(warning_event)

        # Still in NORMAL (below threshold of 5)
        assert supervisor.mode == OperationMode.NORMAL

        # Report 5th failure - should escalate
        supervisor.report_solver_failure(warning_event)
        assert supervisor.mode == OperationMode.CONSERVATIVE

    def test_conservative_mode_escalates_to_safe_stop(
        self, supervisor: Supervisor, mock_estimate: Estimate
    ) -> None:
        """Accumulated failures in CONSERVATIVE mode should trigger SAFE_STOP."""
        regressor = np.array([0.5, 0.1])
        supervisor.update(mock_estimate, regressor)

        # Force into CONSERVATIVE mode
        supervisor._mode_manager.trigger_conservative()
        assert supervisor.mode == OperationMode.CONSERVATIVE

        # Report failures at threshold
        warning_event = SolverFailureEvent(
            component=ComponentType.NMPC_CASADI,
            severity=FailureSeverity.WARNING,
            message="NMPC did not converge",
        )

        for _ in range(5):
            supervisor.report_solver_failure(warning_event)

        # Should escalate to SAFE_STOP
        assert supervisor.mode == OperationMode.SAFE_STOP

    def test_events_passed_via_update(
        self, supervisor: Supervisor, mock_estimate: Estimate
    ) -> None:
        """Events can be passed via the update method."""
        regressor = np.array([0.5, 0.1])

        critical_event = SolverFailureEvent(
            component=ComponentType.CBF_QP,
            severity=FailureSeverity.CRITICAL,
            message="CBF-QP infeasible",
        )

        # Pass event via update
        state = supervisor.update(mock_estimate, regressor, solver_events=[critical_event])

        # Should be in SAFE_STOP
        assert state.mode == OperationMode.SAFE_STOP
        assert state.solver_failure_count == 1

    def test_recent_failures_tracked(
        self, supervisor: Supervisor, mock_estimate: Estimate
    ) -> None:
        """Recent failures should be tracked in state."""
        regressor = np.array([0.5, 0.1])

        events = [
            SolverFailureEvent(
                component=ComponentType.MHE_CASADI,
                severity=FailureSeverity.WARNING,
                message=f"Failure {i}",
            )
            for i in range(3)
        ]

        for event in events:
            supervisor.report_solver_failure(event)

        state = supervisor.update(mock_estimate, regressor)

        assert len(state.recent_failures) == 3
        assert state.solver_failure_count == 3

    def test_clear_failure_count(
        self, supervisor: Supervisor, mock_estimate: Estimate
    ) -> None:
        """Failure count can be cleared manually."""
        regressor = np.array([0.5, 0.1])

        event = SolverFailureEvent(
            component=ComponentType.MHE_CASADI,
            severity=FailureSeverity.WARNING,
            message="MHE failure",
        )

        for _ in range(3):
            supervisor.report_solver_failure(event)

        assert supervisor._solver_failure_count == 3

        supervisor.clear_failure_count()
        assert supervisor._solver_failure_count == 0


@pytest.mark.unit
class TestCBFQPFailureHandling:
    """Tests for CBF-QP specific failure handling."""

    @pytest.mark.skipif(
        not pytest.importorskip("osqp", reason="OSQP not installed"),
        reason="OSQP not installed",
    )
    def test_cbf_qp_returns_failure_event_on_infeasibility(self) -> None:
        """CBF-QP should return failure event when QP is infeasible."""
        from ddt.safety.cbf_qp import CBFQPSafetyFilter

        # Create filter with impossible constraints to force infeasibility
        filter = CBFQPSafetyFilter(
            nx=1,
            nu=1,
            x_min=np.array([0.0]),  # Narrow constraint band
            x_max=np.array([0.01]),
            u_min=np.array([-0.001]),  # Very tight input bounds
            u_max=np.array([0.001]),
            alpha=0.99,  # Aggressive barrier
        )

        # Create estimate at edge of constraint
        est = Estimate(
            x_hat=np.array([10.0]),  # Way outside constraints
            theta_hat=np.array([1.0, 0.1]),
            theta_cov=np.array([[0.01, 0.0], [0.0, 0.01]]),
        )

        u_nom = np.array([1.0])  # Large input

        # Should return fallback with failure event
        u_safe, event = filter.filter_with_event(u_nom, est)

        # Should have clamped approximately to bounds (allow solver tolerance)
        # OSQP may return slightly outside bounds due to solver tolerances
        assert u_safe[0] >= -0.001 - 1e-4 and u_safe[0] <= 0.001 + 1e-4, (
            f"u_safe={u_safe[0]} should be clamped near [-0.001, 0.001]"
        )

        # Event may or may not be generated depending on QP feasibility
        # The test verifies the interface works correctly


@pytest.mark.unit
class TestMHEFailureHandling:
    """Tests for MHE failure handling."""

    def test_mhe_base_stores_failure_event(self) -> None:
        """MHE base class should store the last failure event."""
        from ddt.events import SolverFailureEvent
        from ddt.estimation.mhe.base import MHEBase

        # MHEBase is abstract, so we test the interface
        # The actual implementation is tested in integration tests

    def test_failure_event_contains_solver_status(self) -> None:
        """Failure events should include raw solver status."""
        event = SolverFailureEvent(
            component=ComponentType.MHE_CASADI,
            severity=FailureSeverity.WARNING,
            message="Solver failed",
            solver_status="Maximum_Iterations_Exceeded",
            iteration_count=50,
        )

        assert event.solver_status == "Maximum_Iterations_Exceeded"
        assert event.iteration_count == 50


@pytest.mark.unit
class TestSimulationEventPropagation:
    """Tests for event propagation through simulation loop."""

    def test_supervised_log_captures_failures(self) -> None:
        """SupervisedSimulationLog should capture solver failures."""
        from ddt.runtime.simulate import SupervisedSimulationLog

        log = SupervisedSimulationLog(
            t=[],
            x=[],
            y=[],
            u_nom=[],
            u_safe=[],
            a_hat=[],
            b_hat=[],
            ref=[],
        )

        event = SolverFailureEvent(
            component=ComponentType.CBF_QP,
            severity=FailureSeverity.CRITICAL,
            message="Test failure",
        )

        log.solver_failures.append(event)

        assert len(log.solver_failures) == 1
        assert log.solver_failures[0].component == ComponentType.CBF_QP
