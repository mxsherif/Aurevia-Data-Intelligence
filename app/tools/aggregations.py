"""The aggregation registry.

Aggregations are data, not code branches: each one is an :class:`Aggregation`
entry in :data:`AGGREGATIONS`, so adding "variance" or "p95" later is a single
:func:`register_aggregation` call and every tool picks it up -- the grouping,
statistics, ranking, time-trend and chart tools all resolve names through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.tools.exceptions import UnsupportedOperationError

#: Aggregation functions receive a Series and return a scalar.
AggFunc = Callable[[pd.Series], Any]


def _percentage_change(series: pd.Series) -> float:
    """Percent change from the first to the last value, in the given order.

    Used as a scalar aggregation ("how much did this group move?"). For
    period-over-period change inside a series, see :func:`add_percentage_change`.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    first, last = float(values.iloc[0]), float(values.iloc[-1])
    if first == 0:
        return float("nan")
    return (last - first) / abs(first) * 100.0


@dataclass(frozen=True)
class Aggregation:
    """One named way to reduce a series to a single value."""

    name: str
    label: str
    func: AggFunc
    #: Requires a numeric column (``count`` and ``min``/``max`` do not).
    numeric_only: bool = True
    #: Result is already expressed as a percentage.
    is_percentage: bool = False
    #: Suffix used when naming an output column (``revenue_sum``).
    suffix: str | None = None
    #: Axis/title template; ``{measure}`` receives the humanised column name.
    #: Keeping it here means a new aggregation gets readable labels for free.
    measure_template: str = "{label} of {measure}"

    @property
    def column_suffix(self) -> str:
        return self.suffix or self.name

    def apply(self, series: pd.Series) -> Any:
        return self.func(series)


def _std(series: pd.Series) -> float:
    """Sample standard deviation; 0.0 for a single value rather than NaN."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return float(values.std(ddof=1))


def _numeric_reduce(reducer: Callable[[pd.Series], Any]) -> AggFunc:
    """Wrap a reducer so it always sees clean numeric input."""

    def apply(series: pd.Series) -> Any:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return float("nan")
        return reducer(values)

    return apply


AGGREGATIONS: dict[str, Aggregation] = {}


def register_aggregation(aggregation: Aggregation, *, overwrite: bool = False) -> Aggregation:
    """Add an aggregation to the registry (the extension point)."""
    if aggregation.name in AGGREGATIONS and not overwrite:
        raise ValueError(f"Aggregation '{aggregation.name}' is already registered.")
    AGGREGATIONS[aggregation.name] = aggregation
    return aggregation


def _register_builtins() -> None:
    builtins = [
        Aggregation(
            "sum", "Sum", _numeric_reduce(lambda s: float(s.sum())),
            measure_template="total {measure}",
        ),
        Aggregation(
            "mean", "Average", _numeric_reduce(lambda s: float(s.mean())),
            measure_template="average {measure}",
        ),
        Aggregation(
            "median", "Median", _numeric_reduce(lambda s: float(s.median())),
            measure_template="median {measure}",
        ),
        Aggregation(
            "count",
            "Count",
            lambda s: int(s.notna().sum()),
            numeric_only=False,
            measure_template="count of {measure}",
        ),
        Aggregation(
            "min", "Minimum", lambda s: _extreme(s, "min"),
            numeric_only=False, measure_template="minimum {measure}",
        ),
        Aggregation(
            "max", "Maximum", lambda s: _extreme(s, "max"),
            numeric_only=False, measure_template="maximum {measure}",
        ),
        Aggregation(
            "std", "Std. deviation", _std,
            measure_template="std. deviation of {measure}",
        ),
        Aggregation(
            "percentage_change",
            "Percentage change",
            _percentage_change,
            is_percentage=True,
            suffix="pct_change",
            measure_template="change in {measure} (%)",
        ),
    ]
    for aggregation in builtins:
        register_aggregation(aggregation)


def _extreme(series: pd.Series, which: str) -> Any:
    """min/max that also works for dates and ordered strings."""
    values = series.dropna()
    if values.empty:
        return float("nan")
    return values.min() if which == "min" else values.max()


_register_builtins()

#: Aliases accepted from callers (and, later, from LLM-generated tool calls).
AGGREGATION_ALIASES: dict[str, str] = {
    "average": "mean",
    "avg": "mean",
    "total": "sum",
    "stdev": "std",
    "std_dev": "std",
    "standard_deviation": "std",
    "pct_change": "percentage_change",
    "percent_change": "percentage_change",
    "size": "count",
    "n": "count",
    "minimum": "min",
    "maximum": "max",
}


def resolve_aggregation(name: Any) -> Aggregation:
    """Look up an aggregation by name or alias, case-insensitively."""
    if isinstance(name, Aggregation):
        return name
    if not isinstance(name, str):
        raise UnsupportedOperationError("aggregation", name, AGGREGATIONS)
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = AGGREGATION_ALIASES.get(key, key)
    if key not in AGGREGATIONS:
        raise UnsupportedOperationError("aggregation", name, AGGREGATIONS)
    return AGGREGATIONS[key]


def supported_aggregations(*, numeric_only: bool | None = None) -> list[str]:
    """Registered aggregation names, optionally filtered by their input needs."""
    names = [
        name
        for name, aggregation in AGGREGATIONS.items()
        if numeric_only is None or aggregation.numeric_only is numeric_only
    ]
    return sorted(names)


def add_percentage_change(values: pd.Series) -> pd.Series:
    """Period-over-period percentage change for an already-ordered series.

    Returns a float series aligned to `values`, with ``NaN`` for the first
    element and wherever the previous value was zero or missing.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    previous = numeric.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        change = (numeric - previous) / previous.abs() * 100.0
    return change.replace([np.inf, -np.inf], np.nan)
