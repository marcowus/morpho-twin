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
    from ddt.safety.barriers import create_box_barriers

    composite = create_box_barriers(
        x_min=np.array([-1.0]),
        x_max=np.array([1.0]),
    )

    # Should create composite with barrier for each dimension
    assert len(composite.barriers) == 1

    # Should be safe at origin
    assert composite.evaluate(np.array([0.0])) > 0


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
