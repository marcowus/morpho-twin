# Architecture (DDT Blueprint v2 — scaffold)

This doc is intentionally concise; the full blueprint is provided in the ChatGPT response that generated this repository scaffold.

Key modules:
- `ddt.estimation`: moving-horizon estimation & parameter uncertainty
- `ddt.control`: nominal NMPC (RTI) + dual-control (FIM) objective terms
- `ddt.safety`: CBF-QP safety filters & mode manager
- `ddt.runtime`: orchestration loop, profiling, logging
- `ddt.baselines`: PID / LQR / baseline MPC
