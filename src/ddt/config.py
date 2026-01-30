from __future__ import annotations

from pydantic import BaseModel, Field


class PlantConfig(BaseModel):
    type: str = Field(..., description="Plant type identifier.")
    a_true: float | None = None
    b_true: float | None = None
    process_noise_std: float = 0.0
    meas_noise_std: float = 0.0
    x0: float = 0.0


class EstimationConfig(BaseModel):
    type: str
    window: int = 50


class ControlConfig(BaseModel):
    type: str
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    u_min: float = -1.0
    u_max: float = 1.0


class SafetyConfig(BaseModel):
    type: str
    x_min: float = -1.0
    x_max: float = 1.0
    alpha: float = 0.05
    slack_weight: float = 1000.0


class ReferenceConfig(BaseModel):
    type: str = "step"
    t_step: float = 0.0
    value: float = 0.0


class ScenarioConfig(BaseModel):
    steps: int = 1000
    reference: ReferenceConfig = ReferenceConfig()


class AppConfig(BaseModel):
    seed: int = 0
    dt: float = 0.1
    horizon_steps: int = 20

    plant: PlantConfig
    estimation: EstimationConfig
    control: ControlConfig
    safety: SafetyConfig
    scenario: ScenarioConfig
