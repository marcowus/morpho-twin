"""Tests for the Supervisor module."""

import warnings

import numpy as np
import pytest

from ddt.interfaces import Estimate
from ddt.supervision.supervisor import Supervisor, create_supervisor


def test_supervisor_initialization():
    """Test Supervisor initializes correctly."""
    supervisor = Supervisor()

    assert supervisor.pe_window == 100
    assert supervisor.pe_lambda_threshold == 0.1
    assert supervisor.pe_condition_threshold == 100.0
    assert supervisor.ntheta == 2


def test_supervisor_warns_when_no_regressor_provided():
    """Supervisor should warn when regressor not explicitly provided."""
    supervisor = Supervisor()

    estimate = Estimate(
        x_hat=np.array([1.0]),
        theta_hat=np.array([0.9, 0.2]),
        theta_cov=np.eye(2) * 0.1,
    )

    # Should warn when regressor is None
    with pytest.warns(UserWarning, match="No regressor provided"):
        supervisor.update(estimate, regressor=None)


def test_supervisor_no_warning_with_explicit_regressor():
    """Supervisor should not warn when regressor is provided."""
    supervisor = Supervisor()

    estimate = Estimate(
        x_hat=np.array([1.0]),
        theta_hat=np.array([0.9, 0.2]),
        theta_cov=np.eye(2) * 0.1,
    )

    # Should not warn when regressor is provided
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        supervisor.update(estimate, regressor=np.array([1.0, 0.5]))


def test_supervisor_condition_threshold_wiring():
    """Test that condition_threshold is properly wired through."""
    supervisor = Supervisor(
        pe_condition_threshold=50.0,  # Custom threshold
    )

    # The PE monitor should have received this threshold
    assert supervisor._pe_monitor.condition_threshold == 50.0


def test_create_supervisor_passes_condition_threshold():
    """Test factory function passes condition_threshold."""
    supervisor = create_supervisor(
        pe_condition_threshold=25.0,
    )

    assert supervisor._pe_monitor.condition_threshold == 25.0


def test_supervisor_mode_transitions():
    """Test supervisor mode transitions based on uncertainty."""
    supervisor = Supervisor()

    # Start in NORMAL mode
    from ddt.supervision.mode_manager import OperationMode

    assert supervisor.mode == OperationMode.NORMAL

    # High uncertainty should trigger mode change
    estimate = Estimate(
        x_hat=np.array([1.0]),
        theta_hat=np.array([0.9, 0.2]),
        theta_cov=np.eye(2) * 10.0,  # High uncertainty
    )

    for _ in range(10):
        supervisor.update(estimate, regressor=np.array([1.0, 0.5]))

    # Should have transitioned to CONSERVATIVE or higher
    assert supervisor.mode != OperationMode.NORMAL
