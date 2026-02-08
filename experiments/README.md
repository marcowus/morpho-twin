# Koopman Operator Control Experiments

This directory contains experiments using Koopman operator theory for system identification and control of the CSTR plant.

## Files

- `koopman_cstr.py`: Main experiment script.
  - Generates synthetic data from `src/ddt/sim/cstr.py`.
  - Trains a `rEDMDc` (Recursive Extended Dynamic Mode Decomposition with Control) model.
  - Runs a closed-loop MPC simulation using the learned linear lifted model.
  - Saves plots of states, inputs, and prediction errors to `experiments/results_koopman/`.

## Running the Experiment

Ensure you have installed the necessary dependencies:
```bash
pip install cvxpy matplotlib scikit-learn seaborn pandas numpy
```

Run the script from the repository root:
```bash
python experiments/koopman_cstr.py
```

## Results

The script will output the training progress and MPC simulation steps.
After completion, check `experiments/results_koopman/` for:
- `states.png`: Trajectories of Concentration ($C_A$) and Temperature ($T$) vs Target.
- `inputs.png`: Control inputs ($q$, $Q_c$) applied.
- `prediction_error.png`: One-step ahead prediction error of the Koopman model.
