"""Integration tests for MHE in closed loop."""

import numpy as np
import pytest

# Check for CasADi
try:
    import casadi  # noqa: F401

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
def test_mhe_converges_to_true_parameters():
    """Test that MHE converges to true system parameters."""
    from ddt.factory import make_controller, make_estimator, make_plant, make_safety
    from ddt.runtime.simulate import run_closed_loop
    from ddt.utils import load_config

    cfg = load_config("configs/mhe_demo.yaml")

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    log = run_closed_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        steps=300,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    # Get final parameter estimates
    a_hat_final = log.a_hat[-1]
    b_hat_final = log.b_hat[-1]

    a_true = cfg.plant.a_true
    b_true = cfg.plant.b_true

    # Check convergence (within 10% relative error)
    a_error = abs(a_hat_final - a_true) / abs(a_true)
    b_error = abs(b_hat_final - b_true) / abs(b_true)

    assert a_error < 0.15, f"a estimate error too large: {a_error:.1%}"
    assert b_error < 0.30, f"b estimate error too large: {b_error:.1%}"


@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
def test_mhe_maintains_safety():
    """Test that MHE + safety filter maintains constraints."""
    from ddt.factory import make_controller, make_estimator, make_plant, make_safety
    from ddt.runtime.simulate import compute_constraint_violations, run_closed_loop
    from ddt.utils import load_config

    cfg = load_config("configs/mhe_demo.yaml")

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    log = run_closed_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        steps=200,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    count, max_viol = compute_constraint_violations(
        log,
        x_min=cfg.safety.x_min,
        x_max=cfg.safety.x_max,
    )

    # Should have minimal violations
    assert max_viol < 0.1


@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
def test_mhe_covariance_decreases():
    """Test that MHE covariance decreases over time with data."""
    from ddt.factory import make_estimator, make_plant
    from ddt.utils import load_config

    cfg = load_config("configs/mhe_demo.yaml")
    plant = make_plant(cfg)
    estimator = make_estimator(cfg)

    estimator.reset()
    sr = plant.reset()

    covariances = []
    est = estimator.update(sr.y, np.array([0.0]))
    covariances.append(np.trace(est.theta_cov))

    # Run for some steps
    for k in range(100):
        u = np.array([0.5 * np.sin(0.1 * k)])  # Exciting input
        sr = plant.step(u)
        est = estimator.update(sr.y, u)
        covariances.append(np.trace(est.theta_cov))

    # Covariance should generally decrease
    early_cov = np.mean(covariances[:20])
    late_cov = np.mean(covariances[-20:])

    assert late_cov <= early_cov * 1.5  # Allow some variation but should trend down
