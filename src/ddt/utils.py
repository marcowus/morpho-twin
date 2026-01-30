from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import AppConfig


def load_config(path: str | Path) -> AppConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(data)
