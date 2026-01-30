# Morpho Twin Operator Guide

## What Morpho Twin Does

Morpho Twin is an adaptive digital twin framework for real-time control with:

1. **Moving Horizon Estimation (MHE)**: Estimates system parameters in real-time
2. **Model Predictive Control (NMPC)**: Computes optimal control actions
3. **Control Barrier Functions (CBF-QP)**: Ensures safety constraints are satisfied
4. **Supervision**: Monitors uncertainty and adjusts safety margins

## System Requirements

| Requirement | Details |
|-------------|---------|
| **Sensors** | State measurements at fixed sample rate |
| **Actuators** | Controllable inputs with known bounds |
| **Model** | Approximate dynamics model (parameters will be learned) |
| **Constraints** | State and input limits |

## Architecture Overview

```
                    ┌─────────────┐
                    │   Plant     │
                    └──────┬──────┘
                           │ y (measurement)
                    ┌──────▼──────┐
                    │     MHE     │──► θ_hat, Σ_θ
                    └──────┬──────┘
                           │ x_hat
    ┌───────────┐   ┌──────▼──────┐
    │ Reference │──►│    NMPC     │──► u_nom
    └───────────┘   └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   CBF-QP    │──► u_safe
                    │ (Safety)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Supervisor  │──► mode, margins
                    └─────────────┘
```

## Operating Modes

The supervisor manages three operating modes:

| Mode | Trigger | Safety Margin | Description |
|------|---------|---------------|-------------|
| **NORMAL** | PE satisfied, low uncertainty | 1.0× | Standard operation |
| **CONSERVATIVE** | PE degraded or uncertainty rising | 2.0× | Tightened margins |
| **SAFE_STOP** | Critical failure or high uncertainty | 5.0× | Zero control, await intervention |

### Mode Transitions

```
             ┌─────────────────────────────────────┐
             │                                     │
             ▼                                     │
        ┌─────────┐    PE degraded or         ┌───┴──────────┐
   ────►│ NORMAL  │    uncertainty high       │ CONSERVATIVE │
        └────┬────┘ ─────────────────────────►└───────┬──────┘
             │                                        │
             │ Critical failure                       │ Continued failures
             │ or extreme uncertainty                 │ or extreme uncertainty
             │                                        │
             ▼                                        ▼
        ┌─────────────────────────────────────────────────┐
        │                  SAFE_STOP                       │
        └─────────────────────────────────────────────────┘
```

## Failure Modes and Handling

### Solver Failures

| Failure | Symptom | Automatic Action | Operator Action |
|---------|---------|------------------|-----------------|
| **CBF-QP infeasible** | Safety cannot be guaranteed | Immediate SAFE_STOP | Check constraints, restart |
| **NMPC timeout** | Control computation too slow | Replay previous trajectory | Check system load |
| **MHE divergence** | Parameter estimates unstable | Conservative mode + warning | Check model accuracy |
| **PE violation** | Insufficient excitation | Increase probing | Verify reference signal |

### Failure Escalation

1. **Single WARNING** (MHE/NMPC): Logged, fallback used
2. **5 consecutive WARNINGs**: Mode escalates (NORMAL → CONSERVATIVE)
3. **CRITICAL failure** (CBF-QP): Immediate SAFE_STOP
4. **5 failures in CONSERVATIVE**: Escalate to SAFE_STOP

## Configuration

### Key Parameters

```yaml
# Sample time (must match sensor rate)
dt: 0.1

# MHE configuration
estimation:
  mhe:
    horizon: 20  # Estimation window
    noise:
      Q_diag: [0.04]  # Process noise variance
      R_diag: [0.09]  # Measurement noise variance

# NMPC configuration
control:
  nmpc:
    horizon: 20  # Prediction horizon
    Q: [10.0]    # State tracking weight
    R_u: [0.1]   # Input regularization
    lambda_info: 0.01  # Probing weight for PE

# Safety configuration
safety:
  x_min: -1.0
  x_max: 1.0
  cbf:
    alpha: 0.5        # Barrier decay rate
    gamma_robust: 1.0 # Robust margin scaling

# Supervision thresholds
supervision:
  pe:
    lambda_threshold: 0.1  # Min eigenvalue for PE
  solver_failure_threshold: 5  # Failures before escalation
```

## Running Demos

### Basic Demo (Linear Scalar)

```bash
morpho --config configs/full_demo.yaml
```

### CSTR Benchmark (Nonlinear)

```bash
morpho --config configs/cstr_demo.yaml
```

### View Real-Time Status

```bash
morpho --config configs/full_demo.yaml --verbose
```

## Interpreting Logs

### Key Log Fields

| Field | Description | Healthy Range |
|-------|-------------|---------------|
| `mode` | Current operating mode | NORMAL |
| `pe_lambda_min` | Minimum eigenvalue of FIM | > threshold |
| `uncertainty` | Trace of parameter covariance | Decreasing over time |
| `safety_margin` | Current margin factor | 1.0 (NORMAL) |
| `solver_failures` | List of recent failures | Empty |

### Example Log Output

```json
{
  "t": 10.5,
  "mode": "NORMAL",
  "pe_lambda_min": 0.25,
  "uncertainty": 0.032,
  "safety_margin": 1.0,
  "x_hat": [0.78],
  "theta_hat": [1.018, 0.098],
  "u_safe": [0.15]
}
```

## Troubleshooting

| Issue | Check | Solution |
|-------|-------|----------|
| Poor tracking | Is PE satisfied? | Increase `lambda_info` |
| Frequent safety overrides | Are constraints too tight? | Check margin factor |
| Parameter drift | Is there enough excitation? | Verify reference variation |
| Mode thrashing | Are thresholds appropriate? | Tune mode config |
| RTI timing violations | Is computation fast enough? | Reduce horizon, use simpler model |
| NIS test failures | Is covariance accurate? | Check noise parameters |

### Diagnostic Commands

```bash
# Check PE status
cat .morpho_logs/last_run.json | jq '.pe_lambda_min | min'

# Check for solver failures
cat .morpho_logs/last_run.json | jq '.solver_failures | length'

# Check constraint violations
cat .morpho_logs/last_run.json | jq '.x | select(. > 1.0 or . < -1.0)'
```

## Safety Guidelines

### Before Operation

1. Verify model accuracy on test data
2. Check that constraints are physically achievable
3. Test in simulation before deployment
4. Verify sensor calibration

### During Operation

1. Monitor `mode` - escalation indicates issues
2. Watch `uncertainty` - should decrease over time
3. Check `pe_lambda_min` - should stay above threshold
4. Review `solver_failures` - should be empty

### Emergency Procedures

1. **SAFE_STOP triggered**: System applies zero control
2. **Manual intervention required**: Diagnose cause before restart
3. **Restart procedure**:
   - Clear failure count: `supervisor.clear_failure_count()`
   - Reset estimator: `estimator.reset()`
   - Resume from NORMAL mode

## API Reference

### Key Classes

- `Supervisor`: Mode management and failure handling
- `CBFQPSafetyFilter`: Safety constraint enforcement
- `MHEBase`: Parameter estimation base class
- `RTINMPC`: Real-time iteration NMPC

### Key Methods

```python
# Check operating mode
supervisor.mode  # OperationMode.NORMAL

# Check if safe to operate
supervisor.is_safe_to_operate()  # True/False

# Get safety margin
supervisor.safety_margin_factor  # 1.0, 2.0, or 5.0

# Manual safe stop
supervisor.trigger_safe_stop()

# Clear failure count after recovery
supervisor.clear_failure_count()

# Get timing statistics (RTI-NMPC)
controller.get_timing_stats()  # RTITimingStats

# Validate uncertainty (MHE)
estimator.validate_uncertainty()  # UncertaintyValidation
```

## Version Information

- Framework: Morpho Twin v0.1
- MHE backends: CasADi (fallback), acados (primary)
- NMPC backends: CasADi (fallback), acados RTI (primary)
- Safety: OSQP-based CBF-QP
