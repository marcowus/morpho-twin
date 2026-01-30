"""Barrier function implementations for CBF-QP safety."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class BarrierFunction(ABC):
    """Abstract base class for Control Barrier Functions.

    A CBF h(x) satisfies:
        h(x) >= 0  implies  x in safe set

    The CBF condition for safety:
        Lf·h + Lg·h·u + α·h >= 0
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate barrier function h(x)."""
        ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient ∇h(x)."""
        ...

    @abstractmethod
    def lie_derivatives(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute Lie derivatives Lf·h and Lg·h.

        Args:
            x: State
            f: Drift dynamics f(x)
            g: Input matrix g(x) such that x_dot = f + g*u

        Returns:
            Lf_h: Lie derivative along f
            Lg_h: Lie derivative along g (row vector)
        """
        ...


@dataclass
class LowerBoundBarrier(BarrierFunction):
    """Lower bound barrier: h(x) = x - x_min.

    Enforces x >= x_min for a single state component.
    """

    x_min: np.ndarray
    component: int = 0

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate h(x) = x_i - x_min_i."""
        x = np.atleast_1d(x)
        x_i = x[self.component] if len(x) > self.component else x[0]
        return float(x_i - self.x_min[self.component])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient is +1 in the component direction."""
        x = np.atleast_1d(x)
        grad = np.zeros_like(x)
        idx = self.component if len(x) > self.component else 0
        grad[idx] = 1.0
        return grad

    def lie_derivatives(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute Lie derivatives."""
        grad = self.gradient(x)
        Lf_h = float(np.dot(grad, f))
        Lg_h = grad @ g
        return Lf_h, Lg_h


@dataclass
class UpperBoundBarrier(BarrierFunction):
    """Upper bound barrier: h(x) = x_max - x.

    Enforces x <= x_max for a single state component.
    """

    x_max: np.ndarray
    component: int = 0

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate h(x) = x_max_i - x_i."""
        x = np.atleast_1d(x)
        x_i = x[self.component] if len(x) > self.component else x[0]
        return float(self.x_max[self.component] - x_i)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient is -1 in the component direction."""
        x = np.atleast_1d(x)
        grad = np.zeros_like(x)
        idx = self.component if len(x) > self.component else 0
        grad[idx] = -1.0
        return grad

    def lie_derivatives(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute Lie derivatives."""
        grad = self.gradient(x)
        Lf_h = float(np.dot(grad, f))
        Lg_h = grad @ g
        return Lf_h, Lg_h


@dataclass
class BoxBarrier(BarrierFunction):
    """Box constraint barrier: x_min <= x <= x_max.

    NOTE: This class returns min(h_low, h_high) which only generates
    one constraint for the closer bound. For proper CBF-QP safety,
    use create_box_barriers() which creates separate LowerBoundBarrier
    and UpperBoundBarrier instances for each dimension.

    Implements separate barriers for lower and upper bounds:
        h_low(x) = x - x_min
        h_high(x) = x_max - x
    """

    x_min: np.ndarray
    x_max: np.ndarray
    component: int = 0  # Which state component

    def evaluate(self, x: np.ndarray) -> float:
        """Return minimum of lower and upper barriers."""
        x = np.atleast_1d(x)
        x_i = x[self.component] if len(x) > self.component else x[0]
        h_low = x_i - self.x_min[self.component]
        h_high = self.x_max[self.component] - x_i
        return min(float(h_low), float(h_high))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient points away from nearest boundary."""
        x = np.atleast_1d(x)
        x_i = x[self.component] if len(x) > self.component else x[0]
        h_low = x_i - self.x_min[self.component]
        h_high = self.x_max[self.component] - x_i

        grad = np.zeros_like(x)
        if h_low < h_high:
            # Lower bound is closer
            grad[self.component if len(x) > self.component else 0] = 1.0
        else:
            # Upper bound is closer
            grad[self.component if len(x) > self.component else 0] = -1.0
        return grad

    def lie_derivatives(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute Lie derivatives."""
        grad = self.gradient(x)
        Lf_h = float(np.dot(grad, f))
        Lg_h = grad @ g
        return Lf_h, Lg_h


@dataclass
class EllipsoidBarrier(BarrierFunction):
    """Ellipsoidal safe set barrier.

    h(x) = 1 - (x - c)^T P (x - c)

    where P defines the ellipsoid shape and c is the center.
    Safe set is the interior of the ellipsoid.
    """

    center: np.ndarray  # Ellipsoid center
    P: np.ndarray  # Shape matrix (positive definite)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate h(x) = 1 - (x-c)^T P (x-c)."""
        x = np.atleast_1d(x)
        dx = x - self.center
        return float(1.0 - dx.T @ self.P @ dx)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient ∇h = -2 P (x - c)."""
        x = np.atleast_1d(x)
        dx = x - self.center
        result: np.ndarray = -2.0 * self.P @ dx
        return result

    def lie_derivatives(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute Lie derivatives."""
        grad = self.gradient(x)
        Lf_h = float(np.dot(grad, f))
        Lg_h = grad @ g
        return Lf_h, Lg_h


@dataclass
class CompositeBarrier(BarrierFunction):
    """Composition of multiple barrier functions.

    The safe set is the intersection of individual safe sets.
    Uses min-composition: h(x) = min_i h_i(x)
    """

    barriers: Sequence[BarrierFunction]

    def evaluate(self, x: np.ndarray) -> float:
        """Return minimum barrier value."""
        return min(b.evaluate(x) for b in self.barriers)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the active (minimum) barrier."""
        values = [(b.evaluate(x), b) for b in self.barriers]
        _, active_barrier = min(values, key=lambda t: t[0])
        return active_barrier.gradient(x)

    def lie_derivatives(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Lie derivatives of active barrier."""
        values = [(b.evaluate(x), b) for b in self.barriers]
        _, active_barrier = min(values, key=lambda t: t[0])
        return active_barrier.lie_derivatives(x, f, g)

    def get_all_constraints(
        self,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        alpha: float,
    ) -> list[tuple[float, np.ndarray, float]]:
        """Get CBF constraints for all barriers.

        Returns list of (Lf_h, Lg_h, h) tuples for constraint:
            Lf_h + Lg_h @ u + alpha * h >= 0

        Args:
            x: Current state
            f: Drift dynamics
            g: Input matrix
            alpha: CBF class-K function parameter

        Returns:
            List of constraint tuples
        """
        constraints = []
        for barrier in self.barriers:
            h = barrier.evaluate(x)
            Lf_h, Lg_h = barrier.lie_derivatives(x, f, g)
            constraints.append((Lf_h, Lg_h, h))
        return constraints


def create_box_barriers(
    x_min: np.ndarray,
    x_max: np.ndarray,
) -> CompositeBarrier:
    """Create composite barrier for box constraints.

    Creates TWO barriers per state component: one for the lower bound
    and one for the upper bound. This ensures both bounds are enforced
    as separate CBF constraints in the QP.

    Args:
        x_min: Lower bounds (nx,)
        x_max: Upper bounds (nx,)

    Returns:
        CompositeBarrier with 2*nx box constraints
    """
    x_min = np.atleast_1d(x_min)
    x_max = np.atleast_1d(x_max)
    nx = len(x_min)

    barriers: list[BarrierFunction] = []
    for i in range(nx):
        barriers.append(LowerBoundBarrier(x_min=x_min, component=i))
        barriers.append(UpperBoundBarrier(x_max=x_max, component=i))

    return CompositeBarrier(barriers=barriers)
