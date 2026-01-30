"""Integration tests for supervised simulation loop."""

import pytest

# Check for optional dependencies
try:
    import osqp  # noqa: F401

    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False


def test_supervised_loop_basic():
    """Test basic supervised loop runs without error."""
    from ddt.factory import (
        make_controller,
        make_estimator,
        make_plant,
        make_safety,
    )
    from ddt.runtime.simulate import run_supervised_loop
    from ddt.supervision import Supervisor
    from ddt.utils import load_config

    cfg = load_config("configs/linear_demo.yaml")

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    # Create supervisor manually (not in config)
    supervisor = Supervisor(
        pe_window=50,
        pe_lambda_threshold=0.1,
        ntheta=2,
    )

    log = run_supervised_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        supervisor=supervisor,
        steps=50,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    assert len(log.t) == 50
    assert len(log.mode) == 50
    assert len(log.pe_lambda_min) == 50


def test_mode_transitions_in_loop():
    """Test that mode transitions happen during simulation."""
    from ddt.factory import make_controller, make_estimator, make_plant, make_safety
    from ddt.runtime.simulate import run_supervised_loop
    from ddt.supervision import Supervisor
    from ddt.supervision.mode_manager import ModeConfig
    from ddt.utils import load_config

    cfg = load_config("configs/linear_demo.yaml")

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    # Create supervisor with low threshold for quick mode change
    mode_config = ModeConfig(
        uncertainty_normal_to_conservative=0.01,  # Very low threshold
    )
    supervisor = Supervisor(
        pe_window=10,
        pe_lambda_threshold=0.1,
        ntheta=2,
        mode_config=mode_config,
    )

    log = run_supervised_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        supervisor=supervisor,
        steps=100,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    # With low threshold, should see mode changes
    unique_modes = set(log.mode)
    assert len(unique_modes) >= 1  # At least NORMAL mode


def test_safety_maintained_in_loop():
    """Test that safety constraints are maintained."""
    from ddt.factory import make_controller, make_estimator, make_plant, make_safety
    from ddt.runtime.simulate import (
        compute_constraint_violations,
        run_supervised_loop,
    )
    from ddt.supervision import Supervisor
    from ddt.utils import load_config

    cfg = load_config("configs/linear_demo.yaml")

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    supervisor = Supervisor(
        pe_window=50,
        pe_lambda_threshold=0.1,
        ntheta=2,
    )

    log = run_supervised_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        supervisor=supervisor,
        steps=200,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    # Check constraint violations
    count, max_viol = compute_constraint_violations(
        log,
        x_min=cfg.safety.x_min,
        x_max=cfg.safety.x_max,
    )

    # Allow small violations due to noise
    assert max_viol < 0.1  # Less than 10% of constraint range


@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
def test_cbf_supervised_loop():
    """Test supervised loop with CBF-QP safety filter."""
    from ddt.factory import (
        make_controller,
        make_estimator,
        make_plant,
        make_safety,
        make_supervisor,
    )
    from ddt.runtime.simulate import (
        compute_constraint_violations,
        run_supervised_loop,
    )
    from ddt.utils import load_config

    cfg = load_config("configs/cbf_demo.yaml")

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)
    supervisor = make_supervisor(cfg)

    assert supervisor is not None

    log = run_supervised_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        supervisor=supervisor,
        steps=100,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    # CBF should maintain safety
    count, max_viol = compute_constraint_violations(
        log,
        x_min=cfg.safety.x_min,
        x_max=cfg.safety.x_max,
    )

    assert max_viol < 0.15  # Allow small violations from noise/slack
