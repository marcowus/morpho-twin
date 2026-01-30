# Differentiable Digital Twins (DDT) — Scaffold

This repository is a **project scaffold** for implementing a *causally-correct* differentiable digital twin:
Moving Horizon Estimation (MHE) for identification, dual-control NMPC (RTI) for control, and a Control Barrier Function (CBF) safety filter.

## What you get
- Clean module boundaries (**model**, **estimation**, **control**, **safety**, **runtime**, **metrics**)
- Config-driven experiments (YAML)
- Baseline controllers (PID, LQR)
- Unit + integration test stubs (including **stop-gradient / no causal leakage** tests)

## Quickstart (simulation-only)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m ddt.cli simulate --config configs/linear_demo.yaml
```

> Notes:
> - `acados` integration is optional; see `docs/acados.md`.
> - `jax` is optional; used for gradient-leak tests and differentiable components.

## Structure
See `docs/architecture.md` for the blueprint and `docs/acceptance_criteria.md` for acceptance tests.
