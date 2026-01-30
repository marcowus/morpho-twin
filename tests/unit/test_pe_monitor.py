"""Tests for PE monitoring."""

import numpy as np


def test_pe_monitor_initialization():
    """Test PEMonitor initialization."""
    from ddt.supervision.pe_monitor import PEMonitor

    monitor = PEMonitor(window=50, lambda_threshold=0.1, ntheta=2)

    status = monitor.get_status()
    assert status.lambda_min >= 0
    assert not status.is_pe_satisfied  # Initially not satisfied


def test_pe_monitor_update():
    """Test PEMonitor update with regressors."""
    from ddt.supervision.pe_monitor import PEMonitor

    monitor = PEMonitor(window=50, lambda_threshold=0.1, ntheta=2)

    # Add exciting regressors
    for i in range(100):
        regressor = np.array([np.sin(0.1 * i), np.cos(0.1 * i)])
        status = monitor.update(regressor)

    # Should have non-zero FIM
    assert status.lambda_min > 0


def test_pe_satisfied_with_excitation():
    """Test that PE is satisfied with sufficient excitation."""
    from ddt.supervision.pe_monitor import PEMonitor

    monitor = PEMonitor(window=50, lambda_threshold=0.01, ntheta=2)

    # Add diverse regressors
    for i in range(100):
        regressor = np.array([i * 0.1, 1.0])  # Growing x, constant u
        monitor.update(regressor)

    status = monitor.get_status()
    # With growing state, should have good excitation
    assert status.lambda_min > 0


def test_pe_not_satisfied_without_excitation():
    """Test that PE fails without excitation."""
    from ddt.supervision.pe_monitor import PEMonitor

    monitor = PEMonitor(window=50, lambda_threshold=1.0, ntheta=2)

    # Add constant regressors (no excitation)
    for _i in range(100):
        regressor = np.array([0.5, 0.5])  # Constant
        monitor.update(regressor)

    status = monitor.get_status()
    # Constant regressors give poor FIM
    assert status.recommended_probe_weight > 0  # Should recommend probing


def test_pe_monitor_reset():
    """Test PEMonitor reset."""
    from ddt.supervision.pe_monitor import PEMonitor

    monitor = PEMonitor(window=50, ntheta=2)

    # Add some data
    for i in range(20):
        monitor.update(np.array([i, i]))

    # Reset
    monitor.reset()

    # Should be back to initial state
    fim = monitor.get_fim()
    assert np.allclose(fim, np.eye(2) * 1e-6, atol=1e-5)


def test_condition_threshold_gating():
    """PE should be violated when FIM condition number exceeds threshold."""
    from ddt.supervision.pe_monitor import PEMonitor

    # Use a low condition threshold to test the gating behavior
    monitor = PEMonitor(
        window=50,
        lambda_threshold=0.001,  # Very low so lambda_min passes
        condition_threshold=5.0,  # Strict condition threshold
        ntheta=2,
    )

    # Add regressors that create an ill-conditioned FIM
    # One direction has much more variance than the other
    for i in range(50):
        # Large variance in first component, small in second
        regressor = np.array([10.0 * (i + 1), 0.01])
        monitor.update(regressor)

    status = monitor.get_status()

    # The FIM will be ill-conditioned - high condition number
    assert status.condition_number > 5.0, (
        f"Condition number {status.condition_number} should exceed threshold"
    )

    # PE should NOT be satisfied because condition_number > condition_threshold,
    # even though lambda_min might be above lambda_threshold
    assert not status.is_pe_satisfied, (
        "PE should be violated due to high condition number"
    )


def test_condition_threshold_passed_when_balanced():
    """PE should be satisfied when FIM is well-conditioned."""
    from ddt.supervision.pe_monitor import PEMonitor

    monitor = PEMonitor(
        window=50,
        lambda_threshold=0.1,
        condition_threshold=100.0,  # Reasonable threshold
        ntheta=2,
    )

    # Add balanced excitation
    for i in range(100):
        regressor = np.array([np.sin(0.1 * i), np.cos(0.1 * i)])
        monitor.update(regressor)

    status = monitor.get_status()

    # With balanced excitation, condition should be reasonable
    assert status.condition_number < 100.0
    assert status.lambda_min > 0.1
    assert status.is_pe_satisfied
