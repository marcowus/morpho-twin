"""Tests for NMPC stop-gradient boundary."""

import importlib

import pytest


@pytest.mark.unit
def test_freeze_theta_numpy():
    """Test freeze_theta with numpy arrays."""
    import numpy as np

    from ddt.control.nmpc_base import freeze_theta

    theta = np.array([1.0, 0.5])
    theta_frozen = freeze_theta(theta)

    # Should return numpy array
    assert isinstance(theta_frozen, np.ndarray)
    np.testing.assert_array_equal(theta, theta_frozen)


@pytest.mark.unit
def test_freeze_theta_with_jax():
    """Test freeze_theta blocks gradients with JAX."""
    jax_spec = importlib.util.find_spec("jax")
    if jax_spec is None:
        pytest.skip("jax not installed (optional extra)")

    import jax
    import jax.numpy as jnp

    from ddt.control.nmpc_base import freeze_theta

    def control_loss(theta):
        """Control loss that uses frozen theta."""
        theta_frozen = freeze_theta(theta)
        # Simulate control computation using frozen theta
        u = 2.0 * theta_frozen[0] + theta_frozen[1]
        return (u - 1.0) ** 2

    # Gradient should be zero because theta is frozen
    theta = jnp.array([1.0, 0.5])
    grad = jax.grad(control_loss)(theta)

    assert float(grad[0]) == 0.0
    assert float(grad[1]) == 0.0


@pytest.mark.unit
def test_freeze_theta_preserves_values():
    """Test that freeze_theta preserves the parameter values."""
    import numpy as np

    from ddt.control.nmpc_base import freeze_theta

    theta = np.array([1.02, 0.10, -0.5])
    theta_frozen = freeze_theta(theta)

    np.testing.assert_array_almost_equal(theta, theta_frozen)


@pytest.mark.unit
def test_stop_gradient_in_nmpc_context():
    """Test stop-gradient in an NMPC-like computation."""
    jax_spec = importlib.util.find_spec("jax")
    if jax_spec is None:
        pytest.skip("jax not installed (optional extra)")

    import jax
    import jax.numpy as jnp

    def nmpc_cost(theta):
        """Simulated NMPC cost function."""
        # Freeze theta to prevent gradient flow
        theta_frozen = jax.lax.stop_gradient(theta)

        # Simulate dynamics: x_next = a * x + b * u
        a, b = theta_frozen[0], theta_frozen[1]
        x = 0.5
        u = 0.1

        x_next = a * x + b * u

        # Tracking cost
        x_ref = 0.8
        cost = (x_next - x_ref) ** 2

        return cost

    theta = jnp.array([1.02, 0.10])
    grad = jax.grad(nmpc_cost)(theta)

    # Gradients should be zero because theta is frozen
    assert float(grad[0]) == 0.0
    assert float(grad[1]) == 0.0


@pytest.mark.unit
def test_estimation_gradient_flows():
    """Test that gradients DO flow for estimation (not frozen)."""
    jax_spec = importlib.util.find_spec("jax")
    if jax_spec is None:
        pytest.skip("jax not installed (optional extra)")

    import jax
    import jax.numpy as jnp

    def estimation_loss(theta):
        """Estimation loss where gradients should flow."""
        # For estimation, we want gradients to flow
        a, b = theta[0], theta[1]

        # Simulated prediction
        x = 0.5
        u = 0.1
        x_pred = a * x + b * u

        # Measurement
        y_meas = 0.6

        # Measurement error
        loss = (x_pred - y_meas) ** 2

        return loss

    theta = jnp.array([1.02, 0.10])
    grad = jax.grad(estimation_loss)(theta)

    # Gradients should be non-zero for estimation
    assert float(grad[0]) != 0.0 or float(grad[1]) != 0.0
