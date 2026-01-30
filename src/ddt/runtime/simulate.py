from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from loguru import logger

from ..interfaces import Controller, Estimator, Plant, SafetyFilter


def step_reference_fn(cfg: dict, t: float) -> float:
    if cfg.get("type", "step") == "step":
        t_step = float(cfg.get("t_step", 0.0))
        value = float(cfg.get("value", 0.0))
        return value if t >= t_step else 0.0
    raise ValueError(f"Unknown reference type: {cfg!r}")


@dataclass
class SimulationLog:
    t: list[float]
    x: list[float]
    y: list[float]
    u_nom: list[float]
    u_safe: list[float]
    a_hat: list[float]
    b_hat: list[float]
    ref: list[float]


def run_closed_loop(
    plant: Plant,
    estimator: Estimator,
    controller: Controller,
    safety: SafetyFilter,
    steps: int,
    dt: float,
    reference_cfg: dict,
) -> SimulationLog:
    estimator.reset()
    controller.reset()
    safety.reset()

    sr = plant.reset()
    t = 0.0

    log = SimulationLog(t=[], x=[], y=[], u_nom=[], u_safe=[], a_hat=[], b_hat=[], ref=[])

    est = estimator.update(sr.y, np.array([0.0]))

    for k in range(steps):
        r = step_reference_fn(reference_cfg, t)
        u_nom = controller.compute_u(np.array([r]), est)
        u_safe = safety.filter(u_nom, est)

        sr = plant.step(u_safe)
        est = estimator.update(sr.y, u_safe)

        log.t.append(t)
        log.x.append(float(sr.x.reshape(())))
        log.y.append(float(sr.y.reshape(())))
        log.u_nom.append(float(u_nom.reshape(())))
        log.u_safe.append(float(u_safe.reshape(())))
        log.a_hat.append(float(est.theta_hat.reshape(-1)[0]))
        log.b_hat.append(float(est.theta_hat.reshape(-1)[1]))
        log.ref.append(float(r))

        t += dt

    logger.info("Simulation complete: {} steps", steps)
    return log
