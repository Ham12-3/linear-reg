"""
Performance profiling module for Linear Regression Studio.
Provides lightweight timing utilities for measuring execution time.
"""

import time
import functools
from contextlib import contextmanager
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import streamlit as st


@dataclass
class TimingRecord:
    """Record of a single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float


class PerformanceTracker:
    """
    Singleton performance tracker that collects timing data.
    Thread-safe for Streamlit's execution model.
    """
    _instance: Optional['PerformanceTracker'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()

    def reset(self):
        """Reset all timing data."""
        self._timings: List[TimingRecord] = []
        self._aggregated: Dict[str, List[float]] = defaultdict(list)

    def record(self, name: str, duration_ms: float):
        """Record a timing measurement."""
        record = TimingRecord(
            name=name,
            duration_ms=duration_ms,
            timestamp=time.time()
        )
        self._timings.append(record)
        self._aggregated[name].append(duration_ms)

    def get_timings(self) -> List[TimingRecord]:
        """Get all timing records."""
        return self._timings.copy()

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated summary statistics."""
        summary = {}
        for name, durations in self._aggregated.items():
            if durations:
                summary[name] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations)
                }
        return summary

    def get_slowest(self, n: int = 3) -> List[tuple]:
        """Get the N slowest operations by total time."""
        summary = self.get_summary()
        sorted_ops = sorted(
            summary.items(),
            key=lambda x: x[1]['total_ms'],
            reverse=True
        )
        return sorted_ops[:n]


# Global tracker instance
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """Get the global performance tracker."""
    return _tracker


def reset_tracker():
    """Reset the global tracker."""
    _tracker.reset()


@contextmanager
def timer(name: str, track: bool = True):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("data_load"):
            load_data()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        if track:
            _tracker.record(name, duration_ms)


def timed(name: Optional[str] = None):
    """
    Decorator for timing functions.

    Usage:
        @timed("data_load")
        def load_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with timer(op_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def is_debug_mode() -> bool:
    """Check if debug performance mode is enabled."""
    return st.session_state.get('debug_performance', False)


def show_performance_ui():
    """
    Display performance metrics in the Streamlit UI.
    Only shows when debug mode is enabled.
    """
    if not is_debug_mode():
        return

    with st.expander("🔍 Performance Debug", expanded=False):
        tracker = get_tracker()

        # Summary metrics
        summary = tracker.get_summary()
        if not summary:
            st.info("No timing data collected yet.")
            return

        # Slowest operations
        st.markdown("**Slowest Operations:**")
        slowest = tracker.get_slowest(5)

        for name, stats in slowest:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(name)
            with col2:
                st.text(f"{stats['total_ms']:.1f}ms total")
            with col3:
                st.text(f"{stats['avg_ms']:.1f}ms avg")

        # Detailed breakdown
        if st.checkbox("Show detailed breakdown", key="perf_detail"):
            import pandas as pd

            rows = []
            for name, stats in summary.items():
                rows.append({
                    'Operation': name,
                    'Count': stats['count'],
                    'Total (ms)': round(stats['total_ms'], 2),
                    'Avg (ms)': round(stats['avg_ms'], 2),
                    'Min (ms)': round(stats['min_ms'], 2),
                    'Max (ms)': round(stats['max_ms'], 2)
                })

            df = pd.DataFrame(rows)
            df = df.sort_values('Total (ms)', ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Reset button
        if st.button("Reset Timings", key="reset_perf"):
            reset_tracker()
            st.rerun()


def get_performance_summary_text() -> str:
    """Get a text summary of performance for export."""
    tracker = get_tracker()
    summary = tracker.get_summary()

    if not summary:
        return "No timing data available."

    lines = ["Performance Summary", "=" * 40]

    for name, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        lines.append(
            f"{name}: {stats['total_ms']:.1f}ms total, "
            f"{stats['avg_ms']:.1f}ms avg ({stats['count']} calls)"
        )

    return "\n".join(lines)
