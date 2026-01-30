# acados integration notes (optional)

This scaffold assumes you may use `acados` for RTI NMPC and/or MHE.

Typical flow:
1. Build/install acados (C library).
2. Use `acados_template` + CasADi to generate an OCP solver.
3. Wrap it behind the `ddt.control.Controller` interface.

See https://docs.acados.org for reference.
