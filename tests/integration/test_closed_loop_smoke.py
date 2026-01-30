from ddt.factory import make_controller, make_estimator, make_plant, make_safety
from ddt.runtime.simulate import run_closed_loop
from ddt.utils import load_config


def test_linear_demo_runs():
    cfg = load_config("configs/linear_demo.yaml")
    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    log = run_closed_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        steps=50,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    assert len(log.t) == 50
