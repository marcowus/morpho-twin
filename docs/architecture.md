# Architecture

Morpho Twin is a modular framework for adaptive control with safety guarantees.

## Core Modules

| Module | Purpose |
|--------|---------|
| `ddt.estimation` | Moving Horizon Estimation, parameter uncertainty tracking |
| `ddt.control` | NMPC/RTI with dual-control (FIM) objective |
| `ddt.safety` | CBF-QP safety filters, barrier functions, robust margins |
| `ddt.supervision` | PE monitoring, mode management |
| `ddt.runtime` | Orchestration loop, profiling, logging |
| `ddt.baselines` | PID, LQR, baseline MPC |

## Data Flow

```
Plant → y → MHE → (θ̂, Σ_θ) → NMPC → u_nom → CBF-QP → u_safe → Plant
                      ↓                         ↑
                Supervision ──────────────────────
                (PE monitor, mode manager)
```

## Key Interfaces

- `Estimator` — Updates parameter estimates from measurements
- `Controller` — Computes nominal control from state and parameters
- `SafetyFilter` — Projects nominal control to safe set
- `Supervisor` — Monitors system health and adapts safety margins

## Solver Backends

- **CasADi + IPOPT** — Default, pure Python installation
- **acados + HPIPM** — Production, requires separate installation

See `docs/acados.md` for acados setup instructions.
