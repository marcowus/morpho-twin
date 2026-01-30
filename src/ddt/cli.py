from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich import print as rprint

from .factory import make_controller, make_estimator, make_plant, make_safety
from .metrics import compute_metrics
from .runtime.simulate import run_closed_loop
from .utils import load_config

app = typer.Typer(help="Differentiable Digital Twins scaffold CLI.")


@app.command()
def simulate(config: str = typer.Option(..., help="Path to YAML config.")) -> None:
    cfg = load_config(config)

    plant = make_plant(cfg)
    estimator = make_estimator(cfg)
    controller = make_controller(cfg)
    safety = make_safety(cfg)

    log = run_closed_loop(
        plant=plant,
        estimator=estimator,
        controller=controller,
        safety=safety,
        steps=cfg.scenario.steps,
        dt=cfg.dt,
        reference_cfg=cfg.scenario.reference.model_dump(),
    )

    y = np.array(log.y, dtype=float)
    ref = np.array(log.ref, dtype=float)
    m = compute_metrics(y=y, ref=ref, x_min=cfg.safety.x_min, x_max=cfg.safety.x_max)

    rprint("[bold]Metrics[/bold]")
    rprint(m)

    out_dir = Path(".ddt_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "last_run.json"
    out_path.write_text(json.dumps(log.__dict__, indent=2))
    rprint(f"[green]Saved[/green] {out_path}")


if __name__ == "__main__":
    app()
