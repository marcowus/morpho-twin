# Acceptance criteria (checklist)

## Build & hygiene
- `pip install -e ".[dev]"` succeeds on Linux/macOS.
- `ruff check .` passes.
- `mypy src/ddt` passes.
- `pytest` passes.

## Functional (simulation demo)
- `ddt simulate --config configs/linear_demo.yaml` runs end-to-end.
- Safety filter keeps `x` within `[x_min, x_max]` for the demo scenario.

## Stop-gradient / causal separation (optional JAX)
- If `jax` extra is installed, unit test `test_stop_gradient_blocks_control_to_theta` passes.

## Extensibility
- New plants/controllers can be added by implementing the corresponding `Protocol` in `ddt.interfaces`.
