"""RTI timing monitoring for real-time performance validation.

This module provides timing instrumentation for RTI-NMPC to ensure
the solver meets real-time deadlines.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class RTITimingStats:
    """Statistics for RTI timing performance.

    Attributes:
        prepare_mean_ms: Mean prepare phase time in milliseconds
        prepare_p95_ms: 95th percentile prepare time
        feedback_mean_ms: Mean feedback phase time
        feedback_p95_ms: 95th percentile feedback time
        total_mean_ms: Mean total time (prepare + feedback)
        total_p95_ms: 95th percentile total time
        is_within_budget: Whether p95 total is within budget
        budget_ms: Timing budget in milliseconds
        budget_violations: Number of timing budget violations
        sample_count: Number of samples collected
    """

    prepare_mean_ms: float
    prepare_p95_ms: float
    feedback_mean_ms: float
    feedback_p95_ms: float
    total_mean_ms: float
    total_p95_ms: float
    is_within_budget: bool
    budget_ms: float
    budget_violations: int
    sample_count: int


@dataclass
class RTITimingMonitor:
    """Monitor for RTI timing performance.

    Tracks prepare and feedback phase timings to validate
    real-time performance requirements.

    The timing budget is typically a fraction of the sample time dt:
    - budget_fraction=0.2 means p95 < 20% of dt
    - This leaves headroom for other computation and jitter

    Attributes:
        dt: Sample time in seconds
        budget_fraction: Maximum fraction of dt for p95 timing
        window_size: Number of samples for rolling statistics
    """

    dt: float
    budget_fraction: float = 0.2
    window_size: int = 100

    _prepare_times: deque[float] = field(init=False)
    _feedback_times: deque[float] = field(init=False)
    _total_times: deque[float] = field(init=False)
    _budget_violations: int = field(default=0, init=False)
    _current_prepare_start: float | None = field(default=None, init=False)
    _current_feedback_start: float | None = field(default=None, init=False)
    _current_prepare_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._prepare_times = deque(maxlen=self.window_size)
        self._feedback_times = deque(maxlen=self.window_size)
        self._total_times = deque(maxlen=self.window_size)

    @property
    def budget_ms(self) -> float:
        """Timing budget in milliseconds."""
        return self.dt * self.budget_fraction * 1000.0

    def reset(self) -> None:
        """Reset timing statistics."""
        self._prepare_times.clear()
        self._feedback_times.clear()
        self._total_times.clear()
        self._budget_violations = 0
        self._current_prepare_start = None
        self._current_feedback_start = None
        self._current_prepare_time = 0.0

    def start_prepare(self) -> None:
        """Mark start of prepare phase."""
        self._current_prepare_start = time.perf_counter()

    def end_prepare(self) -> None:
        """Mark end of prepare phase."""
        if self._current_prepare_start is None:
            return

        elapsed = time.perf_counter() - self._current_prepare_start
        self._current_prepare_time = elapsed
        self._prepare_times.append(elapsed * 1000.0)  # Store in ms
        self._current_prepare_start = None

    def start_feedback(self) -> None:
        """Mark start of feedback phase."""
        self._current_feedback_start = time.perf_counter()

    def end_feedback(self) -> None:
        """Mark end of feedback phase and check budget."""
        if self._current_feedback_start is None:
            return

        elapsed = time.perf_counter() - self._current_feedback_start
        elapsed_ms = elapsed * 1000.0
        self._feedback_times.append(elapsed_ms)

        # Total time for this iteration
        total_ms = self._current_prepare_time * 1000.0 + elapsed_ms
        self._total_times.append(total_ms)

        # Check budget
        if total_ms > self.budget_ms:
            self._budget_violations += 1
            logger.warning(
                "RTI timing budget exceeded | total={:.2f}ms | budget={:.2f}ms",
                total_ms,
                self.budget_ms,
            )

        self._current_feedback_start = None
        self._current_prepare_time = 0.0

    def get_stats(self) -> RTITimingStats:
        """Compute timing statistics.

        Returns:
            RTITimingStats with current performance metrics
        """
        if len(self._total_times) == 0:
            # No data yet
            return RTITimingStats(
                prepare_mean_ms=0.0,
                prepare_p95_ms=0.0,
                feedback_mean_ms=0.0,
                feedback_p95_ms=0.0,
                total_mean_ms=0.0,
                total_p95_ms=0.0,
                is_within_budget=True,
                budget_ms=self.budget_ms,
                budget_violations=0,
                sample_count=0,
            )

        prepare_arr = np.array(self._prepare_times)
        feedback_arr = np.array(self._feedback_times)
        total_arr = np.array(self._total_times)

        p95_total = float(np.percentile(total_arr, 95)) if len(total_arr) > 0 else 0.0

        return RTITimingStats(
            prepare_mean_ms=float(np.mean(prepare_arr)) if len(prepare_arr) > 0 else 0.0,
            prepare_p95_ms=float(np.percentile(prepare_arr, 95)) if len(prepare_arr) > 0 else 0.0,
            feedback_mean_ms=float(np.mean(feedback_arr)) if len(feedback_arr) > 0 else 0.0,
            feedback_p95_ms=float(np.percentile(feedback_arr, 95)) if len(feedback_arr) > 0 else 0.0,
            total_mean_ms=float(np.mean(total_arr)),
            total_p95_ms=p95_total,
            is_within_budget=p95_total < self.budget_ms,
            budget_ms=self.budget_ms,
            budget_violations=self._budget_violations,
            sample_count=len(self._total_times),
        )

    def check_budget_and_warn(self) -> bool:
        """Check if timing is within budget and warn if not.

        Returns:
            True if within budget, False otherwise
        """
        stats = self.get_stats()

        if not stats.is_within_budget:
            warnings.warn(
                f"RTI timing budget exceeded: p95={stats.total_p95_ms:.2f}ms > "
                f"budget={stats.budget_ms:.2f}ms ({stats.budget_violations} violations)",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        return True
