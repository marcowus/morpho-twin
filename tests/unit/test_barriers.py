"""Tests for barrier functions."""

import numpy as np


def test_box_barrier_evaluate():
    """Test BoxBarrier evaluation."""
    from ddt.safety.barriers import BoxBarrier

    barrier = BoxBarrier(
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        component=0,
    )

    # Inside safe set
    assert barrier.evaluate(np.array([0.0])) > 0
    assert barrier.evaluate(np.array([0.5])) > 0
    assert barrier.evaluate(np.array([-0.5])) > 0

    # On boundary
    assert np.isclose(barrier.evaluate(np.array([1.0])), 0.0)
    assert np.isclose(barrier.evaluate(np.array([-1.0])), 0.0)

    # Outside safe set
    assert barrier.evaluate(np.array([1.5])) < 0
    assert barrier.evaluate(np.array([-1.5])) < 0


def test_box_barrier_gradient():
    """Test BoxBarrier gradient computation."""
    from ddt.safety.barriers import BoxBarrier

    barrier = BoxBarrier(
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        component=0,
    )

    # Closer to upper bound: gradient points inward (negative)
    grad = barrier.gradient(np.array([0.6]))
    assert grad[0] < 0

    # Closer to lower bound: gradient points inward (positive)
    grad = barrier.gradient(np.array([-0.6]))
    assert grad[0] > 0


def test_composite_barrier():
    """Test CompositeBarrier min-composition."""
    from ddt.safety.barriers import BoxBarrier, CompositeBarrier

    b1 = BoxBarrier(x_min=np.array([-1.0]), x_max=np.array([1.0]), component=0)
    b2 = BoxBarrier(x_min=np.array([-0.5]), x_max=np.array([0.5]), component=0)

    composite = CompositeBarrier(barriers=[b1, b2])

    # At x=0.4, both are positive but b2 is smaller
    h1 = b1.evaluate(np.array([0.4]))
    h2 = b2.evaluate(np.array([0.4]))
    h_composite = composite.evaluate(np.array([0.4]))

    assert h_composite == min(h1, h2)


def test_create_box_barriers():
    """Test factory function for box barriers."""
    from ddt.safety.barriers import LowerBoundBarrier, UpperBoundBarrier, create_box_barriers

    composite = create_box_barriers(
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
    )

    # Should create 2 barriers per dimension (one lower, one upper)
    assert len(composite.barriers) == 2

    # Verify barrier types
    assert isinstance(composite.barriers[0], LowerBoundBarrier)
    assert isinstance(composite.barriers[1], UpperBoundBarrier)

    # Should be safe at origin
    assert composite.evaluate(np.array([0.0])) > 0


def test_box_barriers_enforce_both_bounds():
    """Verify both lower and upper bounds generate separate constraints."""
    from ddt.safety.barriers import create_box_barriers

    composite = create_box_barriers(x_min=np.array([-1.0]), x_max=np.array([1.0]))

    x = np.array([0.0])
    f = np.array([0.1])
    g = np.array([[1.0]])
    alpha = 0.5

    # Get all constraints - should have 2 (one per bound)
    constraints = composite.get_all_constraints(x, f, g, alpha)
    assert len(constraints) == 2

    # Each constraint should have different gradients (opposite signs)
    _, Lg_h_lower, _ = constraints[0]
    _, Lg_h_upper, _ = constraints[1]
    assert Lg_h_lower[0] > 0  # Lower bound gradient is positive
    assert Lg_h_upper[0] < 0  # Upper bound gradient is negative


def test_box_barrier_lie_derivatives():
    """Test Lie derivative computation."""
    from ddt.safety.barriers import BoxBarrier

    barrier = BoxBarrier(
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
        component=0,
    )

    x = np.array([0.5])
    f = np.array([0.1])  # Drift
    g = np.array([[1.0]])  # Input matrix

    Lf_h, Lg_h = barrier.lie_derivatives(x, f, g)

    # Verify types
    assert isinstance(Lf_h, float)
    assert isinstance(Lg_h, np.ndarray)
