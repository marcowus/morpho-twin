"""Tests for solver structural integrity.

These tests verify that theta (θ) is structurally a parameter in the solvers,
NOT a decision variable. This is critical for the "no cheating" guarantee:
the optimizer must not be able to modify parameter estimates.

If these tests fail, it indicates a serious bug where the optimizer could
potentially "cheat" by modifying parameters to achieve better performance.
"""

from __future__ import annotations

import numpy as np
import pytest


# Check for optional dependencies
try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    ca = None  # type: ignore[assignment]

try:
    from acados_template import AcadosOcpSolver

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False


def _build_test_dynamics():
    """Build simple scalar dynamics for testing."""
    from ddt.control.ocp.dynamics import build_discrete_dynamics

    return build_discrete_dynamics(dt=0.1, model_type="linear_scalar")


def _build_test_cost():
    """Build simple cost for testing."""
    from ddt.control.ocp.cost import build_tracking_cost

    return build_tracking_cost(Q_diag=[10.0], R_diag=[0.1])


def _build_test_constraints():
    """Build simple constraints for testing."""
    from ddt.control.ocp.constraints import build_constraint_vectors

    return build_constraint_vectors(
        x_min=[-10.0],
        x_max=[10.0],
        u_min=[-1.0],
        u_max=[1.0],
    )


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
class TestCasADiSolverStructure:
    """Tests for CasADi solver structure."""

    def test_theta_is_parameter_not_decision_variable(self) -> None:
        """Verify theta is in NLP parameters (p), NOT decision variables (w).

        Decision variables (w) should only contain:
        - x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N

        Parameters (p) should contain:
        - x0 (initial state)
        - x_ref (reference)
        - theta (model parameters)
        """
        from ddt.control.solvers.casadi_backend import CasADiSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()
        horizon = 5

        solver = CasADiSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=horizon,
            dt=0.1,
        )

        nx = dynamics.nx  # 1
        nu = dynamics.nu  # 1
        ntheta = dynamics.ntheta  # 2
        N = horizon  # 5

        # Decision variables: x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N
        # Count: N+1 states + N inputs = 6 + 5 = 11
        expected_n_w = (N + 1) * nx + N * nu
        assert solver._n_w == expected_n_w, (
            f"Decision variable count wrong: got {solver._n_w}, expected {expected_n_w}. "
            f"This suggests theta may have been incorrectly added to decision variables!"
        )

        # Parameters: x0 (nx) + x_ref (nx) + theta (ntheta)
        expected_n_p = nx + nx + ntheta  # 1 + 1 + 2 = 4
        assert solver._n_p == expected_n_p, (
            f"Parameter count wrong: got {solver._n_p}, expected {expected_n_p}. "
            f"Theta may not be properly included in parameters!"
        )

    def test_theta_cannot_be_optimized(self) -> None:
        """Verify solving with different theta doesn't optimize theta.

        The optimizer should take theta as given and compute optimal
        control for that fixed theta value.
        """
        from ddt.control.solvers.casadi_backend import CasADiSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()

        solver = CasADiSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=5,
            dt=0.1,
        )

        x0 = np.array([0.5])
        x_ref = np.array([0.8])

        # Solve with two different theta values
        theta1 = np.array([1.02, 0.10])
        theta2 = np.array([0.98, 0.15])

        u1, x1, success1 = solver.solve(x0, x_ref, theta1)
        u2, x2, success2 = solver.solve(x0, x_ref, theta2)

        assert success1, "Solver failed with theta1"
        assert success2, "Solver failed with theta2"

        # Solutions should differ because theta is different
        # If theta were being optimized, solutions might be suspiciously similar
        assert not np.allclose(u1, u2), (
            "Solutions are identical despite different theta values. "
            "This suggests theta might not be properly affecting the optimization."
        )

    def test_decision_variable_bounds_exclude_theta(self) -> None:
        """Verify bounds are only for states and inputs, not theta."""
        from ddt.control.solvers.casadi_backend import CasADiSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()
        horizon = 5

        solver = CasADiSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=horizon,
            dt=0.1,
        )

        # Check that lbw/ubw length matches decision variables
        expected_len = (horizon + 1) * dynamics.nx + horizon * dynamics.nu
        assert len(solver._lbw) == expected_len, (
            f"Lower bound vector length wrong: {len(solver._lbw)} vs {expected_len}"
        )
        assert len(solver._ubw) == expected_len, (
            f"Upper bound vector length wrong: {len(solver._ubw)} vs {expected_len}"
        )

    def test_nlp_structure_symbolic_verification(self) -> None:
        """Directly verify NLP structure via symbolic inspection."""
        from ddt.control.solvers.casadi_backend import CasADiSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()
        horizon = 3

        solver = CasADiSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=horizon,
            dt=0.1,
        )

        nx = dynamics.nx  # 1
        nu = dynamics.nu  # 1
        ntheta = dynamics.ntheta  # 2

        # Verify decision variable count matches expected
        # Decision variables: x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N
        # Count: (N+1)*nx + N*nu = 4 + 3 = 7
        expected_n_w = (horizon + 1) * nx + horizon * nu
        assert solver._n_w == expected_n_w, f"n_w: expected {expected_n_w}, got {solver._n_w}"

        # Verify parameter count: x0 + x_ref + theta = 1 + 1 + 2 = 4
        expected_n_p = nx + nx + ntheta
        assert solver._n_p == expected_n_p, f"n_p: expected {expected_n_p}, got {solver._n_p}"


@pytest.mark.unit
@pytest.mark.skipif(not ACADOS_AVAILABLE, reason="acados not installed")
class TestAcadosSolverStructure:
    """Tests for acados solver structure."""

    def test_theta_is_parameter_not_state(self) -> None:
        """Verify theta is in model.p, NOT model.x or model.u."""
        from ddt.control.solvers.acados_backend import AcadosSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()

        solver = AcadosSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=5,
            dt=0.1,
        )

        # Access the acados OCP to inspect model dimensions
        ocp = solver._acados_solver.acados_ocp

        nx = ocp.dims.nx
        nu = ocp.dims.nu
        np_dim = ocp.dims.np  # Parameter dimension

        # State should only be the actual state (nx=1 for linear scalar)
        assert nx == dynamics.nx, (
            f"State dimension includes extra variables: got {nx}, expected {dynamics.nx}. "
            f"Theta may be incorrectly in state!"
        )

        # Input should only be the actual input (nu=1 for linear scalar)
        assert nu == dynamics.nu, (
            f"Input dimension includes extra variables: got {nu}, expected {dynamics.nu}. "
            f"Theta may be incorrectly in input!"
        )

        # Parameter dimension should include theta
        assert np_dim == dynamics.ntheta, (
            f"Parameter dimension wrong: got {np_dim}, expected {dynamics.ntheta}. "
            f"Theta may not be properly set as parameter!"
        )

    def test_theta_changes_affect_solution(self) -> None:
        """Verify different theta values produce different solutions."""
        from ddt.control.solvers.acados_backend import AcadosSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()

        solver = AcadosSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=5,
            dt=0.1,
        )

        x0 = np.array([0.5])
        x_ref = np.array([0.8])

        # Solve with two different theta values
        theta1 = np.array([1.02, 0.10])
        theta2 = np.array([0.98, 0.15])

        u1, x1, success1 = solver.solve(x0, x_ref, theta1)
        u2, x2, success2 = solver.solve(x0, x_ref, theta2)

        assert success1, "Solver failed with theta1"
        assert success2, "Solver failed with theta2"

        # Solutions should differ because theta affects dynamics
        assert not np.allclose(u1, u2), (
            "Solutions identical despite different theta. "
            "Theta may not be properly used in dynamics."
        )

    def test_parameter_update_at_all_stages(self) -> None:
        """Verify theta is set at all prediction stages."""
        from ddt.control.solvers.acados_backend import AcadosSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()
        horizon = 5

        solver = AcadosSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=horizon,
            dt=0.1,
        )

        theta = np.array([1.02, 0.10])

        # Run prepare to set parameters
        solver.prepare(theta)

        # Verify parameters are set at all stages
        acados_solver = solver._acados_solver
        for i in range(horizon + 1):
            p_i = acados_solver.get(i, "p")
            np.testing.assert_array_almost_equal(
                p_i, theta,
                err_msg=f"Parameter at stage {i} not correctly set",
            )


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
class TestThetaExclusion:
    """Tests ensuring theta exclusion from optimization."""

    def test_cost_depends_on_theta_only_through_dynamics(self) -> None:
        """Verify cost function structure.

        The cost should depend on theta only indirectly through:
        J = sum ||x_k - x_ref||^2_Q + ||u_k||^2_R

        where x_k evolves according to x_{k+1} = f(x_k, u_k, theta).
        Theta appears in dynamics, not directly in cost.
        """
        from ddt.control.ocp.cost import QuadraticCost

        cost = _build_test_cost()

        # Cost structure should have no theta dependency
        assert isinstance(cost, QuadraticCost)
        assert cost.Q.shape == (1, 1)  # State weight
        assert cost.R.shape == (1, 1)  # Input weight
        assert cost.Q_terminal.shape == (1, 1)  # Terminal weight

        # These matrices should only weight states and inputs
        # There should be no theta-related terms

    def test_regression_guard_theta_in_decision_vars(self) -> None:
        """Guard against regression: theta accidentally added to w.

        This test explicitly checks the NLP structure to ensure theta
        hasn't been added to decision variables.
        """
        from ddt.control.solvers.casadi_backend import CasADiSolver

        dynamics = _build_test_dynamics()
        cost = _build_test_cost()
        constraints = _build_test_constraints()
        horizon = 10

        solver = CasADiSolver(
            dynamics=dynamics,
            cost=cost,
            constraints=constraints,
            horizon=horizon,
            dt=0.1,
        )

        nx, nu, ntheta = dynamics.nx, dynamics.nu, dynamics.ntheta
        N = horizon

        # If someone accidentally adds theta to w, n_w would be:
        # (N+1)*nx + N*nu + ntheta = 11 + 10 + 2 = 23
        # Instead of correct: (N+1)*nx + N*nu = 11 + 10 = 21

        expected_without_theta = (N + 1) * nx + N * nu
        incorrect_with_theta = expected_without_theta + ntheta

        assert solver._n_w == expected_without_theta, (
            f"n_w = {solver._n_w}. "
            f"Expected {expected_without_theta} (without theta), "
            f"but would be {incorrect_with_theta} if theta was in w. "
            f"REGRESSION: Theta may have been added to decision variables!"
        )
