"""Time-series aggregation.

Resamples a dataset onto a calendar frequency (daily … yearly), guaranteeing
chronological order and reporting how many rows had missing or unparseable
dates rather than silently dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.tools.aggregations import add_percentage_change, resolve_aggregation
from app.tools.exceptions import UnsupportedOperationError
from app.tools.validation import (
    require_column,
    require_dataframe,
    require_datetime_column,
)


@dataclass(frozen=True)
class Frequency:
    """A calendar frequency and how to render its period labels."""

    name: str
    label: str
    #: pandas offset alias used for resampling.
    alias: str
    #: strftime pattern, or ``None`` when the label needs custom formatting.
    date_format: str | None = "%Y-%m-%d"
    axis_label: str = "Date"


FREQUENCIES: dict[str, Frequency] = {
    "daily": Frequency("daily", "Daily", "D", "%Y-%m-%d", "Day"),
    "weekly": Frequency("weekly", "Weekly", "W-MON", "%Y-%m-%d", "Week beginning"),
    "monthly": Frequency("monthly", "Monthly", "MS", "%Y-%m", "Month"),
    "quarterly": Frequency("quarterly", "Quarterly", "QS", None, "Quarter"),
    "yearly": Frequency("yearly", "Yearly", "YS", "%Y", "Year"),
}

FREQUENCY_ALIASES: dict[str, str] = {
    "day": "daily",
    "d": "daily",
    "week": "weekly",
    "w": "weekly",
    "month": "monthly",
    "m": "monthly",
    "quarter": "quarterly",
    "q": "quarterly",
    "year": "yearly",
    "y": "yearly",
    "annual": "yearly",
    "annually": "yearly",
}


def resolve_frequency(frequency: Any) -> Frequency:
    """Look up a frequency by name or alias, case-insensitively."""
    if isinstance(frequency, Frequency):
        return frequency
    if not isinstance(frequency, str):
        raise UnsupportedOperationError("frequency", frequency, FREQUENCIES)
    key = frequency.strip().lower()
    key = FREQUENCY_ALIASES.get(key, key)
    if key not in FREQUENCIES:
        raise UnsupportedOperationError("frequency", frequency, FREQUENCIES)
    return FREQUENCIES[key]


def supported_frequencies() -> list[str]:
    """Frequency names, ordered from finest to coarsest."""
    return list(FREQUENCIES)


def _format_period(timestamp: pd.Timestamp, frequency: Frequency) -> str:
    if frequency.name == "quarterly":
        return f"{timestamp.year}-Q{timestamp.quarter}"
    return timestamp.strftime(frequency.date_format or "%Y-%m-%d")


@dataclass
class TimeTrendResult:
    """An ordered time series plus the bookkeeping behind it."""

    date_column: str
    value_column: str | None
    aggregation: str
    frequency: str
    #: Chronologically ordered, with columns
    #: ``period`` (timestamp), ``period_label`` (str), ``value``,
    #: ``row_count`` and ``pct_change``.
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows_used: int = 0
    rows_missing_date: int = 0
    rows_unparseable_date: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.dataframe.empty

    @property
    def periods(self) -> list[str]:
        if self.dataframe.empty:
            return []
        return self.dataframe["period_label"].tolist()

    @property
    def values(self) -> list[float]:
        if self.dataframe.empty:
            return []
        return self.dataframe["value"].tolist()

    @property
    def first_period(self) -> str | None:
        return self.periods[0] if self.periods else None

    @property
    def last_period(self) -> str | None:
        return self.periods[-1] if self.periods else None

    @property
    def total_change_pct(self) -> float | None:
        """Percentage change from the first period to the last."""
        values = [v for v in self.values if pd.notna(v)]
        if len(values) < 2 or values[0] == 0:
            return None
        return (values[-1] - values[0]) / abs(values[0]) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_column": self.date_column,
            "value_column": self.value_column,
            "aggregation": self.aggregation,
            "frequency": self.frequency,
            "period_count": len(self.dataframe),
            "rows_used": self.rows_used,
            "rows_missing_date": self.rows_missing_date,
            "rows_unparseable_date": self.rows_unparseable_date,
            "first_period": self.first_period,
            "last_period": self.last_period,
            "total_change_pct": self.total_change_pct,
            "points": self.dataframe.drop(columns=["period"], errors="ignore").to_dict(
                "records"
            ),
            "notes": list(self.notes),
        }


def calculate_time_trend(
    df: pd.DataFrame,
    date_column: str,
    value_column: str | None = None,
    *,
    aggregation: str = "sum",
    frequency: str = "monthly",
    fill_gaps: bool = False,
) -> TimeTrendResult:
    """Aggregate `value_column` over `date_column` at the given `frequency`.

    With no `value_column` the result counts rows per period. Rows whose date is
    missing or unparseable are excluded and counted in the result rather than
    raising, and the output is always sorted oldest-first.

    `fill_gaps` inserts missing periods so the x-axis is evenly spaced; the
    inserted periods carry ``0`` for counts and sums, and ``NaN`` otherwise.
    """
    require_dataframe(df)
    date_name = require_column(df, date_column)
    aggregator = resolve_aggregation("count" if value_column is None else aggregation)
    freq = resolve_frequency(frequency)

    notes: list[str] = []
    missing_dates = int(df[date_name].isna().sum())
    parsed = require_datetime_column(df, date_name)
    unparseable = int(parsed.isna().sum() - missing_dates)

    if value_column is None:
        value_name: str | None = None
        working = pd.DataFrame({"__period": parsed, "__value": 1.0})
    else:
        value_name = require_column(df, value_column)
        if aggregator.numeric_only:
            from app.tools.validation import require_numeric_column

            values = require_numeric_column(df, value_name)
        else:
            values = df[value_name]
        working = pd.DataFrame({"__period": parsed, "__value": values.to_numpy()})

    working = working.dropna(subset=["__period"])
    if missing_dates:
        notes.append(f"{missing_dates:,} row(s) had no date and were excluded.")
    if unparseable > 0:
        notes.append(
            f"{unparseable:,} row(s) had an unrecognisable date and were excluded."
        )

    result = TimeTrendResult(
        date_column=date_name,
        value_column=value_name,
        aggregation=aggregator.name,
        frequency=freq.name,
        rows_used=int(len(working)),
        rows_missing_date=missing_dates,
        rows_unparseable_date=max(unparseable, 0),
        notes=notes,
    )

    if working.empty:
        result.notes.append("No rows with a usable date remained, so the trend is empty.")
        result.dataframe = _empty_trend_frame()
        return result

    working["__period"] = working["__period"].dt.to_period(
        _period_alias(freq)
    ).dt.to_timestamp()

    grouped = working.groupby("__period", sort=True)["__value"]
    aggregated = grouped.apply(aggregator.apply)
    row_counts = grouped.size()

    frame = pd.DataFrame(
        {
            "period": aggregated.index,
            "value": aggregated.to_numpy(),
            "row_count": row_counts.to_numpy(),
        }
    ).sort_values("period", kind="stable").reset_index(drop=True)

    if fill_gaps and len(frame) > 1:
        frame = _fill_period_gaps(frame, freq, aggregator.name)

    frame["period_label"] = [_format_period(ts, freq) for ts in frame["period"]]
    frame["pct_change"] = add_percentage_change(frame["value"])
    frame = frame[["period", "period_label", "value", "row_count", "pct_change"]]

    result.dataframe = frame
    return result


def _period_alias(frequency: Frequency) -> str:
    """``to_period`` alias for a frequency (differs from the resample alias)."""
    return {
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }[frequency.name]


def _fill_period_gaps(
    frame: pd.DataFrame, frequency: Frequency, aggregation: str
) -> pd.DataFrame:
    """Insert absent periods so the axis is evenly spaced."""
    full = pd.date_range(
        frame["period"].min(), frame["period"].max(), freq=frequency.alias
    )
    if len(full) <= len(frame):
        return frame

    filled = (
        frame.set_index("period")
        .reindex(full)
        .rename_axis("period")
        .reset_index()
    )
    filled["row_count"] = filled["row_count"].fillna(0).astype(int)
    if aggregation in ("count", "sum"):
        filled["value"] = filled["value"].fillna(0.0)
    return filled


def _empty_trend_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": pd.Series(dtype="datetime64[ns]"),
            "period_label": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
            "row_count": pd.Series(dtype="int64"),
            "pct_change": pd.Series(dtype="float64"),
        }
    )
