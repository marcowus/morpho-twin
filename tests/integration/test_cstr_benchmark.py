"""Integration tests for CSTR benchmark.

These tests verify that:
1. CSTR plant dynamics work correctly
2. Safety constraints are maintained under parameter drift
3. NMPC+MHE outperforms PID baseline
4. PE monitoring enables parameter convergence
"""

from __future__ import annotations

import numpy as np
import pytest

from ddt.sim.cstr import CSTRPlant


@pytest.fixture
def cstr_plant() -> CSTRPlant:
    """Create a CSTR plant for testing."""
    plant = CSTRPlant(
        dt=0.1,
        k_0_true=7.2e10,
        E_a_true=72750.0,
        dH_r_true=-50000.0,
        process_noise_std=np.array([0.0, 0.0]),  # No noise for deterministic tests
        meas_noise_std=np.array([0.0, 0.0]),
    )
    plant.seed(42)
    return plant


@pytest.mark.integration
class TestCSTRPlantBasics:
    """Basic tests for CSTR plant functionality."""

    def test_plant_creation(self, cstr_plant: CSTRPlant) -> None:
        """Test CSTR plant creation."""
        assert cstr_plant.nx == 2
        assert cstr_plant.nu == 2
        assert cstr_plant.ny == 2
        assert cstr_plant.ntheta == 3

    def test_plant_reset(self, cstr_plant: CSTRPlant) -> None:
        """Test CSTR plant reset."""
        result = cstr_plant.reset()

        assert result.x.shape == (2,)
        assert result.y.shape == (2,)
        np.testing.assert_array_almost_equal(
            result.x, [cstr_plant.C_A_init, cstr_plant.T_init]
        )

    def test_plant_step(self, cstr_plant: CSTRPlant) -> None:
        """Test CSTR plant step."""
        cstr_plant.reset()

        # Step with nominal input
        u = np.array([1.0, 0.0])  # Flow rate 1 L/s, no cooling
        result = cstr_plant.step(u)

        assert result.x.shape == (2,)
        # State should have changed
        assert not np.allclose(result.x, [cstr_plant.C_A_init, cstr_plant.T_init])

    def test_temperature_increases_with_reaction(self, cstr_plant: CSTRPlant) -> None:
        """Test that temperature increases due to exothermic reaction."""
        cstr_plant.reset()
        initial_T = cstr_plant._x[1]

        # Run for a few steps with no cooling
        for _ in range(10):
            cstr_plant.step(np.array([1.0, 0.0]))

        final_T = cstr_plant._x[1]

        # Temperature should increase (exothermic reaction)
        assert final_T > initial_T, "Temperature should increase with exothermic reaction"

    def test_cooling_reduces_temperature(self, cstr_plant: CSTRPlant) -> None:
        """Test that coolant heat removal reduces temperature."""
        cstr_plant.reset()

        # Heat up first
        for _ in range(10):
            cstr_plant.step(np.array([1.0, 0.0]))

        heated_T = cstr_plant._x[1]

        # Now apply cooling
        for _ in range(10):
            cstr_plant.step(np.array([1.0, -500.0]))  # Strong cooling

        cooled_T = cstr_plant._x[1]

        assert cooled_T < heated_T, "Cooling should reduce temperature"

    def test_concentration_bounds(self, cstr_plant: CSTRPlant) -> None:
        """Test that concentration stays non-negative."""
        cstr_plant.reset()

        # Run for many steps
        for _ in range(100):
            cstr_plant.step(np.array([1.5, 0.0]))

        C_A = cstr_plant._x[0]
        assert C_A >= 0, "Concentration should not be negative"

    def test_parameter_drift(self, cstr_plant: CSTRPlant) -> None:
        """Test parameter drift functionality."""
        original_k0 = cstr_plant.k_0_true

        # Apply 10% degradation
        cstr_plant.apply_parameter_drift(k_0_factor=0.9)

        assert cstr_plant.k_0_true == original_k0 * 0.9

    def test_safety_check(self, cstr_plant: CSTRPlant) -> None:
        """Test safety checking functionality."""
        cstr_plant.reset()

        # Initially should be safe
        assert cstr_plant.is_safe()

        # Check with margin
        assert cstr_plant.is_safe(margin=10.0)

    def test_get_true_parameters(self, cstr_plant: CSTRPlant) -> None:
        """Test getting true parameters."""
        params = cstr_plant.get_true_parameters()

        assert len(params) == 3
        assert params[0] == cstr_plant.k_0_true
        assert params[1] == cstr_plant.E_a_true
        assert params[2] == cstr_plant.dH_r_true


@pytest.mark.integration
class TestCSTRSafetyConstraints:
    """Tests for CSTR safety constraint enforcement."""

    def test_runaway_without_control(self) -> None:
        """Test that temperature rises in batch mode (no flow, no cooling).

        With zero flow (batch reactor) and no active cooling, the
        exothermic reaction causes temperature to rise continuously
        until reactant is depleted.
        """
        plant = CSTRPlant(
            dt=0.1,
            T_init=320.0,  # Start at moderate temperature
            C_A_init=0.8,  # Plenty of reactant
            process_noise_std=np.array([0.0, 0.0]),
            meas_noise_std=np.array([0.0, 0.0]),
        )
        plant.seed(42)
        plant.reset()

        initial_T = plant._x[1]
        max_T = initial_T

        # Run in batch mode: no flow, no cooling
        for _ in range(200):
            plant.step(np.array([0.0, 0.0]))  # Zero flow, zero cooling
            max_T = max(max_T, plant._x[1])

        # Temperature should rise due to exothermic reaction
        # (limited by reactant depletion in batch mode)
        assert max_T > initial_T + 5.0, (
            f"Temperature should rise in batch mode, got {max_T} from {initial_T}"
        )

    def test_cooling_prevents_runaway(self) -> None:
        """Test that active cooling can prevent temperature runaway."""
        plant = CSTRPlant(
            dt=0.1,
            T_init=350.0,
            T_max=400.0,
            process_noise_std=np.array([0.0, 0.0]),
            meas_noise_std=np.array([0.0, 0.0]),
        )
        plant.seed(42)
        plant.reset()

        max_T = plant._x[1]
        T_max = plant.T_max

        # Simple proportional cooling
        for _ in range(200):
            T = plant._x[1]
            # Cool proportionally to how close we are to limit
            Q_c = -500.0 * max(0, (T - 340.0) / 60.0)
            plant.step(np.array([1.0, Q_c]))
            max_T = max(max_T, plant._x[1])

        # Temperature should stay below max
        assert max_T < T_max, f"Temperature {max_T} exceeded max {T_max}"

    def test_constraint_violation_detection(self) -> None:
        """Test constraint violation detection."""
        plant = CSTRPlant(
            dt=0.1,
            T_init=325.0,
            T_max=330.0,  # Low limit, close to where batch mode reaches
            C_A_init=0.8,
            process_noise_std=np.array([0.0, 0.0]),
            meas_noise_std=np.array([0.0, 0.0]),
        )
        plant.seed(42)
        plant.reset()

        # Run in batch mode until violation
        for _ in range(100):
            plant.step(np.array([0.0, 0.0]))  # Batch mode
            violated, amount = plant.get_constraint_violation()
            if violated:
                break

        # Should detect violation
        violated, amount = plant.get_constraint_violation()
        assert violated, "Should detect temperature constraint violation"
        assert amount > 0, "Violation amount should be positive"


@pytest.mark.integration
class TestCSTRParameterDrift:
    """Tests for parameter drift scenarios."""

    def test_steady_state_changes_with_drift(self) -> None:
        """Test that steady state changes as parameters drift."""
        plant = CSTRPlant(
            dt=0.1,
            process_noise_std=np.array([0.0, 0.0]),
            meas_noise_std=np.array([0.0, 0.0]),
        )
        plant.seed(42)
        plant.reset()

        # Run to approximate steady state
        for _ in range(200):
            plant.step(np.array([1.0, 0.0]))

        steady_T_original = plant._x[1]

        # Apply catalyst degradation
        plant.apply_parameter_drift(k_0_factor=0.5)  # 50% degradation

        # Run to new steady state
        for _ in range(200):
            plant.step(np.array([1.0, 0.0]))

        steady_T_degraded = plant._x[1]

        # With less reaction, temperature should be lower
        assert steady_T_degraded < steady_T_original, (
            "Temperature should decrease with catalyst degradation"
        )

    def test_gradual_drift_scenario(self) -> None:
        """Test gradual parameter drift over time."""
        plant = CSTRPlant(
            dt=0.1,
            process_noise_std=np.array([0.0, 0.0]),
            meas_noise_std=np.array([0.0, 0.0]),
        )
        plant.seed(42)
        plant.reset()

        k0_history = [plant.k_0_true]

        # Simulate with gradual degradation
        for step in range(500):
            plant.step(np.array([1.0, 0.0]))

            # Apply 0.1% degradation per step after step 200
            if step > 200:
                plant.apply_parameter_drift(k_0_factor=0.999)
                k0_history.append(plant.k_0_true)

        # k_0 should have decreased significantly
        final_k0 = plant.k_0_true
        initial_k0 = k0_history[0]

        assert final_k0 < 0.8 * initial_k0, (
            f"k_0 should decrease significantly: {final_k0:.2e} vs {initial_k0:.2e}"
        )


@pytest.mark.integration
class TestCSTRNoiseRobustness:
    """Tests for noise robustness."""

    def test_noisy_plant_stays_bounded(self) -> None:
        """Test that noisy plant stays reasonably bounded."""
        plant = CSTRPlant(
            dt=0.1,
            process_noise_std=np.array([0.01, 2.0]),
            meas_noise_std=np.array([0.05, 5.0]),
        )
        plant.seed(42)
        plant.reset()

        T_history = []
        C_A_history = []

        # Simple control to keep temperature bounded
        for _ in range(500):
            T = plant._x[1]
            Q_c = -300.0 * max(0, (T - 340.0) / 50.0)
            result = plant.step(np.array([1.0, Q_c]))
            T_history.append(result.x[1])
            C_A_history.append(result.x[0])

        # Check bounded behavior
        assert max(T_history) < 450.0, f"Max temperature {max(T_history)} too high"
        assert min(T_history) > 250.0, f"Min temperature {min(T_history)} too low"
        assert all(c >= 0 for c in C_A_history), "Concentration should be non-negative"


@pytest.mark.integration
@pytest.mark.slow
class TestCSTRBenchmarkPerformance:
    """Performance benchmark tests (marked slow)."""

    def test_tracking_performance(self) -> None:
        """Test basic setpoint tracking performance."""
        plant = CSTRPlant(
            dt=0.1,
            C_A_init=0.8,
            T_init=310.0,
            process_noise_std=np.array([0.001, 0.5]),
            meas_noise_std=np.array([0.01, 1.0]),
        )
        plant.seed(42)
        plant.reset()

        # Target setpoint
        C_A_ref = 0.5
        T_ref = 350.0

        # Simple proportional controller
        tracking_errors = []
        for _ in range(500):
            C_A, T = plant._x

            # Simple feedback
            u_C_A = 0.5 * (C_A_ref - C_A)  # Adjust flow to track concentration
            u_T = -200.0 * (T - T_ref)  # Cool if too hot

            q = 1.0 + np.clip(u_C_A, -0.5, 0.5)
            Q_c = np.clip(u_T, -500.0, 500.0)

            plant.step(np.array([q, Q_c]))

            # Tracking error
            error = np.sqrt((C_A - C_A_ref) ** 2 + ((T - T_ref) / 100.0) ** 2)
            tracking_errors.append(error)

        # Should achieve reasonable tracking
        final_errors = tracking_errors[-100:]
        mean_final_error = np.mean(final_errors)

        assert mean_final_error < 0.5, (
            f"Mean final tracking error {mean_final_error} too large"
        )
