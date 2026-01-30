"""Simulation runtime for Morpho Twin closed-loop control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from ..interfaces import Controller, Estimator, Plant, SafetyFilter

if TYPE_CHECKING:
    from ..supervision import Supervisor


def step_reference_fn(cfg: dict, t: float) -> float:
    """Generate reference signal based on configuration."""
    if cfg.get("type", "step") == "step":
        t_step = float(cfg.get("t_step", 0.0))
        value = float(cfg.get("value", 0.0))
        return value if t >= t_step else 0.0
    raise ValueError(f"Unknown reference type: {cfg!r}")


@dataclass
class SimulationLog:
    """Log of simulation data."""

    t: list[float]
    x: list[float]
    y: list[float]
    u_nom: list[float]
    u_safe: list[float]
    a_hat: list[float]
    b_hat: list[float]
    ref: list[float]


@dataclass
class SupervisedSimulationLog(SimulationLog):
    """Extended log with supervision data."""

    mode: list[str] = field(default_factory=list)
    pe_lambda_min: list[float] = field(default_factory=list)
    uncertainty: list[float] = field(default_factory=list)
    safety_margin: list[float] = field(default_factory=list)


def run_closed_loop(
    plant: Plant,
    estimator: Estimator,
    controller: Controller,
    safety: SafetyFilter,
    steps: int,
    dt: float,
    reference_cfg: dict,
) -> SimulationLog:
    """Run standard closed-loop simulation.

    Args:
        plant: Plant model
        estimator: State/parameter estimator
        controller: Control law
        safety: Safety filter
        steps: Number of simulation steps
        dt: Time step
        reference_cfg: Reference configuration

    Returns:
        SimulationLog with recorded data
    """
    estimator.reset()
    controller.reset()
    safety.reset()

    sr = plant.reset()
    t = 0.0

    log = SimulationLog(t=[], x=[], y=[], u_nom=[], u_safe=[], a_hat=[], b_hat=[], ref=[])

    est = estimator.update(sr.y, np.array([0.0]))

    for _k in range(steps):
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


def run_supervised_loop(
    plant: Plant,
    estimator: Estimator,
    controller: Controller,
    safety: SafetyFilter,
    supervisor: Supervisor,
    steps: int,
    dt: float,
    reference_cfg: dict,
) -> SupervisedSimulationLog:
    """Run closed-loop simulation with supervision.

    The supervisor monitors PE and uncertainty, adjusting the
    safety filter margins based on operating mode.

    Args:
        plant: Plant model
        estimator: State/parameter estimator
        controller: Control law
        safety: Safety filter (should support set_margin_factor)
        supervisor: Supervision system
        steps: Number of simulation steps
        dt: Time step
        reference_cfg: Reference configuration

    Returns:
        SupervisedSimulationLog with extended data
    """
    estimator.reset()
    controller.reset()
    safety.reset()
    supervisor.reset()

    sr = plant.reset()
    t = 0.0

    log = SupervisedSimulationLog(
        t=[],
        x=[],
        y=[],
        u_nom=[],
        u_safe=[],
        a_hat=[],
        b_hat=[],
        ref=[],
        mode=[],
        pe_lambda_min=[],
        uncertainty=[],
        safety_margin=[],
    )

    est = estimator.update(sr.y, np.array([0.0]))

    for _k in range(steps):
        r = step_reference_fn(reference_cfg, t)

        # Compute control
        if not supervisor.is_safe_to_operate():
            # SAFE_STOP mode: apply zero control
            u_nom = supervisor.get_zero_control(1)
            u_safe = u_nom
            logger.warning("SAFE_STOP mode active at t={:.2f}", t)
        else:
            # Normal operation
            u_nom = controller.compute_u(np.array([r]), est)
            u_safe = safety.filter(u_nom, est)

        # Update supervisor with correct regressor [x, u] for PE monitoring
        regressor = np.concatenate([est.x_hat, u_safe])
        sup_state = supervisor.update(est, regressor)

        # Update safety filter margin based on mode for next iteration
        if hasattr(safety, "set_margin_factor"):
            safety.set_margin_factor(sup_state.safety_margin_factor)

        sr = plant.step(u_safe)
        est = estimator.update(sr.y, u_safe)

        # Log
        log.t.append(t)
        log.x.append(float(sr.x.reshape(())))
        log.y.append(float(sr.y.reshape(())))
        log.u_nom.append(float(u_nom.reshape(())))
        log.u_safe.append(float(u_safe.reshape(())))
        log.a_hat.append(float(est.theta_hat.reshape(-1)[0]))
        log.b_hat.append(float(est.theta_hat.reshape(-1)[1]))
        log.ref.append(float(r))
        log.mode.append(sup_state.mode.name)
        log.pe_lambda_min.append(sup_state.pe_status.lambda_min)
        log.uncertainty.append(sup_state.uncertainty_norm)
        log.safety_margin.append(sup_state.safety_margin_factor)

        t += dt

    logger.info("Supervised simulation complete: {} steps", steps)
    return log


def compute_constraint_violations(
    log: SimulationLog,
    x_min: float,
    x_max: float,
) -> tuple[int, float]:
    """Compute constraint violation statistics.

    Args:
        log: Simulation log
        x_min: State lower bound
        x_max: State upper bound

    Returns:
        (violation_count, max_violation)
    """
    x = np.array(log.x)
    violations_low = np.maximum(0, x_min - x)
    violations_high = np.maximum(0, x - x_max)
    all_violations = np.maximum(violations_low, violations_high)

    count = int(np.sum(all_violations > 1e-6))
    max_viol = float(np.max(all_violations))

    return count, max_viol


def compute_tracking_error(
    log: SimulationLog,
) -> tuple[float, float]:
    """Compute tracking error statistics.

    Returns:
        (iae, max_error) - Integrated Absolute Error and max error
    """
    y = np.array(log.y)
    ref = np.array(log.ref)
    error = np.abs(y - ref)

    iae = float(np.sum(error))
    max_err = float(np.max(error))

    return iae, max_err
