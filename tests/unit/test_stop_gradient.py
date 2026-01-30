import importlib

import pytest


@pytest.mark.unit
def test_stop_gradient_blocks_control_to_theta():
    jax_spec = importlib.util.find_spec("jax")
    if jax_spec is None:
        pytest.skip("jax not installed (optional extra).")

    import jax
    import jax.numpy as jnp

    def control_loss(theta):
        # pretend theta affects future predictions; but we stop-gradient it
        theta_frozen = jax.lax.stop_gradient(theta)
        u = 2.0 * theta_frozen
        return (u - 1.0) ** 2

    g = jax.grad(control_loss)(jnp.array(3.0))
    assert float(g) == 0.0
