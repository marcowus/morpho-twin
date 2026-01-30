"""Tests for FIM computation in dual-control."""

import numpy as np


def test_propagate_sensitivity():
    """Test sensitivity propagation."""
    from ddt.control.dual_control.fim_compute import propagate_sensitivity

    nx, ntheta = 1, 2

    S_prev = np.zeros((nx, ntheta))
    A = np.array([[0.9]])
    C_theta = np.array([[0.1, 0.2]])

    S_next = propagate_sensitivity(S_prev, A, C_theta)

    # S_next = A @ S_prev + C_theta = 0 + C_theta
    np.testing.assert_array_almost_equal(S_next, C_theta)


def test_propagate_sensitivity_accumulated():
    """Test that sensitivity accumulates over time."""
    from ddt.control.dual_control.fim_compute import propagate_sensitivity

    nx, ntheta = 1, 2

    S = np.zeros((nx, ntheta))
    A = np.array([[0.9]])
    C_theta = np.array([[0.1, 0.2]])

    # Propagate multiple steps
    for _ in range(10):
        S = propagate_sensitivity(S, A, C_theta)

    # Should have non-zero sensitivity
    assert np.linalg.norm(S) > 0


def test_compute_fim_prediction():
    """Test FIM computation from trajectory."""
    from ddt.control.dual_control.fim_compute import compute_fim_prediction

    N = 10
    nx, _nu, _ntheta = 1, 1, 2

    x_traj = np.linspace(0, 1, N + 1).reshape(-1, 1)
    u_traj = np.ones((N, 1)) * 0.5
    theta = np.array([1.0, 0.1])

    # Mock Jacobian functions
    def A_func(x, u, th):
        return np.array([[th[0]]])

    def C_func(x, u, th):
        return np.array([[x[0], u[0]]])

    R_inv = np.eye(nx)

    fim = compute_fim_prediction(
        x_traj=x_traj,
        u_traj=u_traj,
        theta=theta,
        A_func=A_func,
        C_func=C_func,
        R_inv=R_inv,
    )

    # FIM should be positive semi-definite
    eigvals = np.linalg.eigvalsh(fim)
    assert np.all(eigvals >= -1e-10)

    # FIM should be symmetric
    np.testing.assert_array_almost_equal(fim, fim.T)


def test_fim_grows_with_excitation():
    """Test that FIM has better conditioning with diverse trajectories."""
    from ddt.control.dual_control.fim_compute import compute_fim_prediction

    N = 20
    nx, _nu, _ntheta = 1, 1, 2
    theta = np.array([1.0, 0.1])

    def A_func(x, u, th):
        return np.array([[th[0]]])

    def C_func(x, u, th):
        return np.array([[x[0], u[0]]])

    R_inv = np.eye(nx)

    # Constant trajectory - poor excitation (rank-deficient regressors)
    x_const = np.ones((N + 1, 1)) * 0.5
    u_const = np.ones((N, 1)) * 0.5

    fim_const = compute_fim_prediction(x_const, u_const, theta, A_func, C_func, R_inv)

    # Varying trajectory - good excitation (diverse regressors)
    x_vary = np.linspace(-1, 1, N + 1).reshape(-1, 1)
    u_vary = np.sin(np.linspace(0, 4 * np.pi, N)).reshape(-1, 1)

    fim_vary = compute_fim_prediction(x_vary, u_vary, theta, A_func, C_func, R_inv)

    # Varying trajectory should have better conditioning (larger min eigenvalue)
    # Constant trajectory has rank-1 structure from repeated identical regressors
    eigvals_const = np.linalg.eigvalsh(fim_const)
    eigvals_vary = np.linalg.eigvalsh(fim_vary)

    # Minimum eigenvalue should be larger for diverse trajectories
    # (constant trajectory has near-zero min eigenvalue due to collinearity)
    assert eigvals_vary.min() >= eigvals_const.min() - 1e-6
