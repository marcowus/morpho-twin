from __future__ import annotations

from typing import Any

from .baselines.pid import PIDController
from .config import AppConfig
from .estimation.windowed_ls import WindowedLeastSquaresEstimator
from .interfaces import Controller, Estimator, Plant, SafetyFilter
from .safety.state_box import StateBoxSafetyFilter
from .sim.linear_scalar import LinearScalarPlant


def make_plant(cfg: AppConfig) -> Plant:
    if cfg.plant.type == "linear_scalar":
        assert cfg.plant.a_true is not None and cfg.plant.b_true is not None
        plant = LinearScalarPlant(
            dt=cfg.dt,
            a_true=cfg.plant.a_true,
            b_true=cfg.plant.b_true,
            process_noise_std=cfg.plant.process_noise_std,
            meas_noise_std=cfg.plant.meas_noise_std,
            x0=cfg.plant.x0,
        )
        plant.seed(cfg.seed)
        return plant
    raise ValueError(f"Unknown plant type: {cfg.plant.type}")


def make_estimator(cfg: AppConfig) -> Estimator:
    if cfg.estimation.type == "windowed_ls":
        return WindowedLeastSquaresEstimator(window=cfg.estimation.window)
    raise ValueError(f"Unknown estimator type: {cfg.estimation.type}")


def make_controller(cfg: AppConfig) -> Controller:
    if cfg.control.type == "pid":
        return PIDController(
            kp=cfg.control.kp,
            ki=cfg.control.ki,
            kd=cfg.control.kd,
            dt=cfg.dt,
            u_min=cfg.control.u_min,
            u_max=cfg.control.u_max,
        )
    raise ValueError(f"Unknown controller type: {cfg.control.type}")


def make_safety(cfg: AppConfig) -> SafetyFilter:
    if cfg.safety.type == "state_box":
        return StateBoxSafetyFilter(
            x_min=cfg.safety.x_min,
            x_max=cfg.safety.x_max,
            alpha=cfg.safety.alpha,
            slack_weight=cfg.safety.slack_weight,
        )
    raise ValueError(f"Unknown safety type: {cfg.safety.type}")
