"""Tests for CBF-QP safety filter."""

import numpy as np
import pytest

# Check if OSQP is available
try:
    import osqp  # noqa: F401

    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_qp_initialization():
    """Test CBF-QP filter initialization."""
    from ddt.safety.cbf_qp import CBFQPSafetyFilter

    filter = CBFQPSafetyFilter(
        nx=1,
        nu=1,
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        u_min=np.array([-2.0]),
        u_max=np.array([2.0]),
    )

    assert filter is not None


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_qp_filter_safe_state():
    """Test that CBF-QP doesn't modify input when state is safe."""
    from ddt.interfaces import Estimate
    from ddt.safety.cbf_qp import CBFQPSafetyFilter

    filter = CBFQPSafetyFilter(
        nx=1,
        nu=1,
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        u_min=np.array([-2.0]),
        u_max=np.array([2.0]),
        alpha=0.5,
    )

    # State at origin (safe)
    est = Estimate(
        x_hat=np.array([0.0]),
        theta_hat=np.array([1.0, 0.1]),
        theta_cov=np.eye(2) * 0.01,
    )

    u_nom = np.array([0.5])
    u_safe = filter.filter(u_nom, est)

    # Should be close to nominal (state is far from boundary)
    assert np.abs(u_safe[0] - u_nom[0]) < 0.5


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_qp_filter_near_boundary():
    """Test that CBF-QP modifies input near boundary."""
    from ddt.interfaces import Estimate
    from ddt.safety.cbf_qp import CBFQPSafetyFilter

    filter = CBFQPSafetyFilter(
        nx=1,
        nu=1,
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        u_min=np.array([-2.0]),
        u_max=np.array([2.0]),
        alpha=0.5,
    )

    # State near upper boundary
    est = Estimate(
        x_hat=np.array([0.9]),
        theta_hat=np.array([1.0, 0.1]),  # a=1, b=0.1 means x_next = x + 0.1*u
        theta_cov=np.eye(2) * 0.01,
    )

    # Nominal input would push further toward boundary
    u_nom = np.array([2.0])
    u_safe = filter.filter(u_nom, est)

    # Should be reduced to prevent violation
    assert u_safe[0] < u_nom[0]


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_qp_respects_input_bounds():
    """Test that CBF-QP respects input bounds."""
    from ddt.interfaces import Estimate
    from ddt.safety.cbf_qp import CBFQPSafetyFilter

    u_max = 1.5
    u_min = -1.5

    filter = CBFQPSafetyFilter(
        nx=1,
        nu=1,
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        u_min=np.array([u_min]),
        u_max=np.array([u_max]),
    )

    est = Estimate(
        x_hat=np.array([0.0]),
        theta_hat=np.array([1.0, 0.1]),
        theta_cov=np.eye(2) * 0.01,
    )

    # Request input beyond bounds
    u_nom = np.array([5.0])
    u_safe = filter.filter(u_nom, est)

    assert u_safe[0] <= u_max + 1e-6
    assert u_safe[0] >= u_min - 1e-6


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_qp_margin_factor():
    """Test that margin factor affects filtering."""
    from ddt.interfaces import Estimate
    from ddt.safety.cbf_qp import CBFQPSafetyFilter

    filter = CBFQPSafetyFilter(
        nx=1,
        nu=1,
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        u_min=np.array([-2.0]),
        u_max=np.array([2.0]),
        gamma_robust=1.0,
    )

    est = Estimate(
        x_hat=np.array([0.7]),
        theta_hat=np.array([1.0, 0.1]),
        theta_cov=np.eye(2) * 1.0,  # High uncertainty
    )

    u_nom = np.array([1.0])

    # Normal margin
    filter.set_margin_factor(1.0)
    u_safe_normal = filter.filter(u_nom, est)

    # Conservative margin
    filter.set_margin_factor(2.0)
    u_safe_conservative = filter.filter(u_nom, est)

    # Conservative should be more restrictive or equal
    # (depends on constraint activity)
    assert u_safe_conservative[0] <= u_safe_normal[0] + 0.5


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_qp_barrier_value():
    """Test barrier value computation."""
    from ddt.safety.cbf_qp import CBFQPSafetyFilter

    filter = CBFQPSafetyFilter(
        nx=1,
        nu=1,
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        u_min=np.array([-2.0]),
        u_max=np.array([2.0]),
    )

    # Inside safe set
    assert filter.get_barrier_value(np.array([0.0])) > 0
    assert filter.is_safe(np.array([0.0]))

    # On boundary
    assert np.isclose(filter.get_barrier_value(np.array([1.0])), 0.0)

    # Outside safe set
    assert filter.get_barrier_value(np.array([1.5])) < 0
    assert not filter.is_safe(np.array([1.5]))
