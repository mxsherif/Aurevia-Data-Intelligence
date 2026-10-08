"""The deterministic analysis executor.

Takes a *validated* :class:`~app.models.plans.AnalysisPlan` and runs it with the
Phase 2 tools. One handler per intent, dispatched through an explicit router —
the LLM never selects or executes code, it only names an intent from a closed
set, and this module decides which Python functions that means.

Every number in the returned :class:`~app.models.results.AnalysisResult` was
computed here. Nothing in this file calls an LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.models.plans import AnalysisPlan, Intent, SortDirection, TIME_INTENTS
from app.models.results import AnalysisResult
from app.tools.aggregations import resolve_aggregation
from app.tools.analysis import (
    calculate_correlation,
    calculate_statistics,
    compare_segments,
    detect_outliers,
    get_column_summary,
    get_dataset_schema,
    group_and_aggregate,
    rank_values,
)
from app.tools.charts import lower_label, supported_chart_types
from app.tools.exceptions import ToolError
from app.tools.filters import describe_conditions, filter_rows
from app.tools.timeseries import calculate_time_trend, resolve_frequency
from app.tools.validation import (
    categorical_columns,
    numeric_columns,
    require_dataframe,
)

logger = logging.getLogger(__name__)

#: Rows kept in a result table handed to the UI.
MAX_TABLE_ROWS = 200

#: Result shapes whose chart is drawn from the source column rather than from
#: the result table, so a short table does not mean a pointless chart.
COLUMN_DRAWN_SHAPES = frozenset({"distribution", "distribution_by_group", "outliers"})


# --------------------------------------------------------------------------- #
# Chart selection
# --------------------------------------------------------------------------- #

def select_visualization(
    *,
    intent: Intent,
    requested: str | None,
    shape: str,
    row_count: int,
) -> str | None:
    """Decide the chart type from the **result shape**, not the model's word.

    The planner's suggestion is honoured only when it fits what was actually
    computed; `shape` is what the handler produced ("time_series", "ranked",
    "correlation_matrix", "distribution", "scalar"). When nothing would help,
    this returns ``None`` rather than forcing a chart.
    """
    natural: dict[str, str | None] = {
        "time_series": "line",
        "ranked": "bar",
        "correlation_matrix": "heatmap",
        "distribution": "histogram",
        "distribution_by_group": "box",
        "outliers": "box",
        "scatter": "scatter",
        "scalar": None,
        "schema": None,
        "none": None,
    }
    default = natural.get(shape, None)

    if default is None:
        # A single number or a schema listing has no useful chart.
        return None
    # The row-count guard only applies to charts drawn from the result table.
    # A distribution or outlier chart is drawn from the source column, so it is
    # meaningful even when the table holds a single row.
    if row_count < 2 and shape not in COLUMN_DRAWN_SHAPES:
        return None

    if requested in supported_chart_types():
        # Accept the suggestion when it is compatible with the shape.
        compatible: dict[str, set[str]] = {
            "time_series": {"line", "bar"},
            "ranked": {"bar", "line"},
            "correlation_matrix": {"heatmap"},
            "distribution": {"histogram", "box"},
            "distribution_by_group": {"box", "histogram"},
            "outliers": {"box", "histogram"},
            "scatter": {"scatter"},
        }
        if requested in compatible.get(shape, set()):
            return requested
        logger.debug(
            "Ignoring chart suggestion %r: incompatible with a %s result",
            requested, shape,
        )
    return default


def normalize_chart_spec(spec: dict[str, Any], chart_type: str) -> dict[str, Any]:
    """Make a chart spec coherent for `chart_type`.

    Overriding the chart type is not enough on its own: a histogram reads its
    column from ``x`` while a box plot reads it from ``y``, and a heatmap takes
    no axes at all. Without this, swapping the type leaves keys behind that
    belong to the chart we decided *not* to draw.
    """
    repaired = dict(spec)
    repaired["chart_type"] = chart_type

    if chart_type == "histogram":
        if not repaired.get("x") and repaired.get("y"):
            repaired["x"] = repaired["y"]
        repaired.pop("y", None)
        for key in ("aggregation", "frequency", "top_n"):
            repaired.pop(key, None)

    elif chart_type == "box":
        if not repaired.get("y") and repaired.get("x"):
            # A lone column on x belongs on y for a box plot.
            repaired["y"] = repaired.pop("x")
        for key in ("aggregation", "frequency", "top_n"):
            repaired.pop(key, None)

    elif chart_type == "heatmap":
        repaired = {"chart_type": "heatmap"}

    elif chart_type == "bar":
        # Bars are categorical: a date axis with a frequency is a line chart's
        # shape, so fall back to the grouping dimension when one is available.
        if repaired.get("frequency") and repaired.get("group_by"):
            repaired["x"] = repaired.pop("group_by")
        repaired.pop("frequency", None)

    elif chart_type == "line":
        repaired.pop("top_n", None)

    return repaired


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _as_number(value: Any) -> Any:
    """Convert numpy scalars so the result model stays JSON-friendly."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame, limit: int = MAX_TABLE_ROWS) -> list[dict[str, Any]]:
    """A dataframe as JSON-friendly records, capped in length."""
    if frame is None or frame.empty:
        return []
    trimmed = frame.head(limit)
    return [
        {str(key): _as_number(value) for key, value in row.items()}
        for row in trimmed.to_dict("records")
    ]


def _measure_name(plan: AnalysisPlan) -> str:
    if plan.metric is None:
        return "Number of rows"
    aggregator = resolve_aggregation(plan.aggregation or "sum")
    return aggregator.measure_template.format(
        label=aggregator.label, measure=lower_label(plan.metric)
    ).capitalize()


def _ascending(plan: AnalysisPlan) -> bool:
    return plan.sort_direction is SortDirection.ASCENDING


def _apply_filters(
    df: pd.DataFrame, plan: AnalysisPlan
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter the data and record what that did."""
    if not plan.filters:
        return df, {}

    conditions = plan.filter_conditions()
    filtered = filter_rows(df, conditions)
    meta = {
        "filters": describe_conditions(conditions),
        "rows_before_filter": int(len(df)),
        "rows_after_filter": int(len(filtered)),
    }
    return filtered, meta


def _is_rate_metric(df: pd.DataFrame, metric: str | None, aggregation: str | None) -> bool:
    """True when the mean of `metric` is really a rate (a 0/1 flag averaged).

    "Churn rate" is the archetype: the column holds 0 and 1, so its mean is a
    proportion. Without this, the only way to answer in percent is for the
    model to multiply by 100 -- which is arithmetic it is forbidden to do, and
    which the grounding check correctly flags. Computing it here keeps the
    natural phrasing available and verifiable.
    """
    if metric is None or metric not in df.columns:
        return False
    if (aggregation or "").lower() not in {"mean", "median"}:
        return False
    values = df[metric].dropna()
    if values.empty:
        return False
    try:
        distinct = set(pd.to_numeric(values, errors="coerce").dropna().unique())
    except (TypeError, ValueError):
        return False
    return bool(distinct) and distinct <= {0.0, 1.0}


def _add_rate_percentage(
    summary: dict[str, Any],
    df: pd.DataFrame,
    plan: AnalysisPlan,
    value: Any,
    label: str,
) -> None:
    """Add a percentage reading of a rate metric to the headline figures."""
    if not _is_rate_metric(df, plan.metric, plan.aggregation):
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    summary[label] = round(float(value) * 100, 2)


def _apply_time_window(
    df: pd.DataFrame, plan: AnalysisPlan
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict the data to the most recent `plan.periods` periods.

    "Which region earned most *last quarter*?" is a ranking with a time scope.
    The temporal handlers do their own windowing, but a ranking, comparison or
    count would otherwise aggregate the whole dataset while the answer claimed
    to be about one quarter -- stating a full-period total as a quarterly one.
    """
    if not plan.periods or plan.time_column is None:
        return df, {}
    if plan.intent in TIME_INTENTS:
        return df, {}  # these scope themselves
    if plan.time_column not in df.columns:
        return df, {}

    frequency = resolve_frequency(plan.time_granularity or "monthly")
    alias = {
        "daily": "D", "weekly": "W", "monthly": "M",
        "quarterly": "Q", "yearly": "Y",
    }[frequency.name]

    try:
        parsed = pd.to_datetime(
            df[plan.time_column], errors="coerce", format="mixed"
        )
    except Exception:  # noqa: BLE001 - an unreadable date means no window
        return df, {}

    usable = parsed.notna()
    if not usable.any():
        return df, {}

    periods = parsed[usable].dt.to_period(alias)
    distinct = periods.sort_values().unique()
    if len(distinct) <= plan.periods:
        return df, {}  # the window is the whole dataset

    kept = set(distinct[-plan.periods:])
    mask = pd.Series(False, index=df.index)
    mask.loc[periods.index] = periods.isin(kept)
    windowed = df.loc[mask].copy()

    labels = sorted(str(p) for p in kept)
    return windowed, {
        "time_window": (
            labels[0] if len(labels) == 1 else f"{labels[0]} to {labels[-1]}"
        ),
        "time_window_periods": plan.periods,
        "time_window_granularity": frequency.name,
        "rows_before_window": int(len(df)),
        "rows_after_window": int(len(windowed)),
    }


def _dimension(plan: AnalysisPlan, df: pd.DataFrame) -> str | None:
    """The grouping column, falling back to a sensible categorical one."""
    if plan.primary_dimension:
        return plan.primary_dimension
    options = categorical_columns(df, max_unique=50)
    return options[0] if options else None


# --------------------------------------------------------------------------- #
# Handlers -- one per intent
# --------------------------------------------------------------------------- #

def _handle_ranking(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    dimension = _dimension(plan, df)
    if dimension is None:
        return AnalysisResult.failure(
            "This dataset has no categorical column to rank by."
        )

    ranking = rank_values(
        df,
        dimension,
        plan.metric,
        aggregation=plan.aggregation or ("count" if plan.metric is None else "sum"),
        top_n=plan.limit or 10,
        ascending=_ascending(plan),
    )
    measure = _measure_name(plan)
    frame = ranking.dataframe

    summary: dict[str, Any] = {}
    if not ranking.is_empty:
        top = frame.iloc[0]
        leader = str(top[dimension])
        summary = {
            f"Top {lower_label(dimension)}": leader,
            f"{measure} ({leader})": _as_number(top["value"]),
            f"Distinct {lower_label(dimension)} values": ranking.total_candidates,
        }
        _add_rate_percentage(
            summary, df, plan, top["value"], f"{measure} ({leader}) as a percentage"
        )
        if len(frame) > 1 and not _is_rate_metric(df, plan.metric, plan.aggregation):
            total = float(pd.to_numeric(frame["value"], errors="coerce").sum())
            if total:
                summary["Share of the listed total"] = round(
                    float(top["value"]) / total * 100, 2
                )

    display = frame.rename(columns={"value": measure})
    if _is_rate_metric(df, plan.metric, plan.aggregation) and measure in display:
        display[f"{measure} %"] = (
            pd.to_numeric(display[measure], errors="coerce") * 100
        ).round(2)

    return AnalysisResult(
        success=True,
        title=f"{measure} by {lower_label(dimension)}",
        summary_data=summary,
        table_data=_records(display),
        metadata={
            "tools": ["rank_values"],
            "dimension": dimension,
            "metric": plan.metric,
            "aggregation": ranking.aggregation,
            "limit": plan.limit,
            "sort": str(plan.sort_direction),
            "total_candidates": ranking.total_candidates,
            "notes": list(ranking.notes),
            "result_shape": "ranked",
        },
        chart_spec={
            "chart_type": "bar",
            "x": dimension,
            "y": plan.metric,
            "aggregation": ranking.aggregation,
            "top_n": plan.limit or 10,
        },
    )


def _handle_comparison(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    dimension = _dimension(plan, df)
    if dimension is None:
        return AnalysisResult.failure(
            "This dataset has no categorical column to compare across."
        )

    aggregation = plan.aggregation or ("count" if plan.metric is None else "mean")
    comparison = compare_segments(
        df,
        dimension,
        plan.metric,
        aggregation=aggregation,
        top_n=plan.limit or 20,
    )
    measure = _measure_name(plan)

    summary: dict[str, Any] = {}
    if not comparison.is_empty:
        best, worst = comparison.best, comparison.worst
        summary = {
            f"Highest {lower_label(dimension)}": str(best["segment"]),
            f"Highest {lower_label(measure)}": _as_number(best["value"]),
            f"Lowest {lower_label(dimension)}": str(worst["segment"]),
            f"Lowest {lower_label(measure)}": _as_number(worst["value"]),
            "Overall baseline": _as_number(comparison.baseline_value),
        }
        _add_rate_percentage(
            summary, df, plan, best["value"],
            f"Highest {lower_label(measure)} as a percentage",
        )
        _add_rate_percentage(
            summary, df, plan, worst["value"],
            f"Lowest {lower_label(measure)} as a percentage",
        )
        spread = _as_number(best["value"]), _as_number(worst["value"])
        if all(isinstance(v, (int, float)) for v in spread) and spread[1]:
            summary["Highest vs lowest (x)"] = round(spread[0] / spread[1], 2)

    frame = comparison.dataframe.rename(
        columns={
            "segment": dimension,
            "row_count": "Rows",
            "value": measure,
            "share_pct": "Share %",
            "difference": "vs baseline",
            "difference_pct": "vs baseline %",
        }
    )
    if _is_rate_metric(df, plan.metric, plan.aggregation) and measure in frame:
        frame.insert(
            frame.columns.get_loc(measure) + 1,
            f"{measure} %",
            (pd.to_numeric(frame[measure], errors="coerce") * 100).round(2),
        )

    return AnalysisResult(
        success=True,
        title=f"{measure} by {lower_label(dimension)}",
        summary_data=summary,
        table_data=_records(frame),
        metadata={
            "tools": ["compare_segments"],
            "dimension": dimension,
            "metric": plan.metric,
            "aggregation": comparison.aggregation,
            "baseline": comparison.baseline,
            "baseline_value": _as_number(comparison.baseline_value),
            "segment_count": len(comparison.dataframe),
            "notes": list(comparison.notes),
            "result_shape": "ranked",
        },
        chart_spec={
            "chart_type": "bar",
            "x": dimension,
            "y": plan.metric,
            "aggregation": comparison.aggregation,
            "top_n": plan.limit or 20,
        },
    )


def _handle_trend(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    if plan.time_column is None:
        return AnalysisResult.failure(
            "Aurevia understood the request, but this dataset does not contain "
            "a valid date field required for trend analysis."
        )

    aggregation = plan.aggregation or ("count" if plan.metric is None else "sum")
    trend = calculate_time_trend(
        df,
        plan.time_column,
        plan.metric,
        aggregation=aggregation,
        frequency=plan.time_granularity or "monthly",
    )
    if trend.is_empty:
        return AnalysisResult.failure(
            "No rows with a usable date remained, so no trend could be computed.",
            notes=list(trend.notes),
        )

    frame = trend.dataframe
    # "the last six months" scopes the window.
    if plan.periods:
        frame = frame.tail(plan.periods)

    measure = _measure_name(plan)
    values = pd.to_numeric(frame["value"], errors="coerce")
    peak = frame.loc[values.idxmax()] if values.notna().any() else None
    trough = frame.loc[values.idxmin()] if values.notna().any() else None

    first, last = float(values.iloc[0]), float(values.iloc[-1])
    change = ((last - first) / abs(first) * 100) if first else None

    summary: dict[str, Any] = {
        "Periods covered": int(len(frame)),
        "First period": str(frame["period_label"].iloc[0]),
        "Last period": str(frame["period_label"].iloc[-1]),
        f"{measure} in the first period": _as_number(first),
        f"{measure} in the last period": _as_number(last),
    }
    if change is not None:
        summary["Change first to last (%)"] = round(change, 2)
    if peak is not None:
        summary["Peak period"] = str(peak["period_label"])
        summary[f"Peak {lower_label(measure)}"] = _as_number(peak["value"])
    if trough is not None:
        summary["Lowest period"] = str(trough["period_label"])
        summary[f"Lowest {lower_label(measure)}"] = _as_number(trough["value"])

    display = frame.rename(
        columns={
            "period_label": "Period",
            "value": measure,
            "row_count": "Rows",
            "pct_change": "Change vs previous (%)",
        }
    ).drop(columns=["period"], errors="ignore")

    return AnalysisResult(
        success=True,
        title=f"{measure} by {plan.time_granularity or 'month'}",
        summary_data=summary,
        table_data=_records(display),
        metadata={
            "tools": ["calculate_time_trend"],
            "time_column": plan.time_column,
            "granularity": trend.frequency,
            "metric": plan.metric,
            "aggregation": trend.aggregation,
            "rows_used": trend.rows_used,
            "rows_excluded": trend.rows_missing_date + trend.rows_unparseable_date,
            "periods": int(len(frame)),
            "notes": list(trend.notes),
            "result_shape": "time_series",
        },
        chart_spec={
            "chart_type": "line",
            "x": plan.time_column,
            "y": plan.metric,
            "aggregation": trend.aggregation,
            "frequency": trend.frequency,
        },
    )


def _handle_time_comparison(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Compare the most recent period with the one before it."""
    if plan.time_column is None:
        return AnalysisResult.failure(
            "Aurevia understood the request, but this dataset does not contain "
            "a valid date field required for the comparison."
        )

    granularity = plan.time_granularity or "quarterly"
    aggregation = plan.aggregation or ("count" if plan.metric is None else "sum")
    trend = calculate_time_trend(
        df, plan.time_column, plan.metric,
        aggregation=aggregation, frequency=granularity,
    )
    if len(trend.dataframe) < 2:
        return AnalysisResult.failure(
            f"The data covers fewer than two {granularity} periods, so there is "
            "nothing to compare against.",
            notes=list(trend.notes),
        )

    frame = trend.dataframe
    latest, previous = frame.iloc[-1], frame.iloc[-2]
    measure = _measure_name(plan)

    latest_value = float(latest["value"])
    previous_value = float(previous["value"])
    absolute = latest_value - previous_value
    percentage = (absolute / abs(previous_value) * 100) if previous_value else None

    summary: dict[str, Any] = {
        "Latest period": str(latest["period_label"]),
        f"{measure} (latest)": _as_number(latest_value),
        "Previous period": str(previous["period_label"]),
        f"{measure} (previous)": _as_number(previous_value),
        "Absolute change": _as_number(absolute),
    }
    if percentage is not None:
        summary["Percentage change"] = round(percentage, 2)
    summary["Direction"] = (
        "increase" if absolute > 0 else "decrease" if absolute < 0 else "no change"
    )

    display = frame.tail(plan.periods or 8).rename(
        columns={
            "period_label": "Period",
            "value": measure,
            "row_count": "Rows",
            "pct_change": "Change vs previous (%)",
        }
    ).drop(columns=["period"], errors="ignore")

    return AnalysisResult(
        success=True,
        title=f"{measure}: {latest['period_label']} vs {previous['period_label']}",
        summary_data=summary,
        table_data=_records(display),
        metadata={
            "tools": ["calculate_time_trend"],
            "time_column": plan.time_column,
            "granularity": trend.frequency,
            "metric": plan.metric,
            "aggregation": trend.aggregation,
            "comparison": "period_over_period",
            "notes": list(trend.notes),
            "result_shape": "time_series",
        },
        chart_spec={
            "chart_type": "line",
            "x": plan.time_column,
            "y": plan.metric,
            "aggregation": trend.aggregation,
            "frequency": trend.frequency,
        },
    )


def _handle_percentage_change(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Rank groups by how much they moved between the first and last period."""
    dimension = plan.primary_dimension
    if dimension is None:
        return _handle_time_comparison(df, plan)
    if plan.time_column is None:
        return AnalysisResult.failure(
            "Aurevia understood the request, but this dataset does not contain "
            "a valid date field required to measure change over time."
        )

    aggregation = plan.aggregation or ("count" if plan.metric is None else "sum")
    granularity = plan.time_granularity or "monthly"
    measure = _measure_name(plan)

    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for group, chunk in df.groupby(dimension, dropna=True, observed=True):
        trend = calculate_time_trend(
            chunk, plan.time_column, plan.metric,
            aggregation=aggregation, frequency=granularity,
        )
        frame = trend.dataframe
        if plan.periods:
            frame = frame.tail(plan.periods)
        if len(frame) < 2:
            continue
        first = float(frame["value"].iloc[0])
        last = float(frame["value"].iloc[-1])
        rows.append(
            {
                dimension: str(group),
                f"{measure} (first period)": _as_number(first),
                f"{measure} (last period)": _as_number(last),
                "Absolute change": _as_number(last - first),
                "Change (%)": round((last - first) / abs(first) * 100, 2)
                if first else None,
                "Periods": int(len(frame)),
            }
        )

    if not rows:
        return AnalysisResult.failure(
            f"No {lower_label(dimension)} value has at least two {granularity} "
            "periods of data, so change over time cannot be measured."
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(
        "Change (%)", ascending=_ascending(plan), na_position="last", kind="stable"
    ).reset_index(drop=True)
    limited = table.head(plan.limit or 10)

    leader = limited.iloc[0]
    summary = {
        f"{'Largest decline' if _ascending(plan) else 'Largest increase'} "
        f"({lower_label(dimension)})": str(leader[dimension]),
        "Change (%)": _as_number(leader["Change (%)"]),
        f"{measure} (first period)": _as_number(leader[f"{measure} (first period)"]),
        f"{measure} (last period)": _as_number(leader[f"{measure} (last period)"]),
        f"{lower_label(dimension).capitalize()} values compared": int(len(table)),
    }

    return AnalysisResult(
        success=True,
        title=f"Change in {lower_label(measure)} by {lower_label(dimension)}",
        summary_data=summary,
        table_data=_records(limited),
        metadata={
            "tools": ["calculate_time_trend", "add_percentage_change"],
            "dimension": dimension,
            "time_column": plan.time_column,
            "granularity": granularity,
            "metric": plan.metric,
            "aggregation": aggregation,
            "limit": plan.limit,
            "notes": notes,
            # The table is a ranking, but the *picture* that answers "what
            # declined" is one trend line per group, so the shape is temporal.
            "result_shape": "time_series",
        },
        chart_spec={
            "chart_type": "line",
            "x": plan.time_column,
            "y": plan.metric,
            "aggregation": aggregation,
            "frequency": granularity,
            "group_by": dimension,
        },
    )


def _handle_summary(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    columns = [plan.metric] if plan.metric else None
    stats = calculate_statistics(df, columns)
    if stats.is_empty:
        return AnalysisResult.failure(
            "This dataset has no numeric column to summarise."
        )

    frame = stats.to_frame()
    if plan.metric and plan.metric in stats.values:
        values = stats.values[plan.metric]
        summary = {
            "Rows": stats.row_count,
            f"Mean {lower_label(plan.metric)}": _as_number(values.get("mean")),
            f"Median {lower_label(plan.metric)}": _as_number(values.get("median")),
            f"Minimum {lower_label(plan.metric)}": _as_number(values.get("min")),
            f"Maximum {lower_label(plan.metric)}": _as_number(values.get("max")),
            f"Total {lower_label(plan.metric)}": _as_number(values.get("sum")),
            "Standard deviation": _as_number(values.get("std")),
        }
        shape = "distribution"
    else:
        summary = {
            "Rows": stats.row_count,
            "Numeric columns summarised": len(stats.columns),
        }
        shape = "none"

    return AnalysisResult(
        success=True,
        title=(
            f"Summary of {lower_label(plan.metric)}" if plan.metric
            else "Dataset summary statistics"
        ),
        summary_data=summary,
        table_data=_records(frame.rename(columns={"column": "Column"})),
        metadata={
            "tools": ["calculate_statistics"],
            "metric": plan.metric,
            "columns": stats.columns,
            "skipped": stats.skipped,
            "result_shape": shape,
        },
        chart_spec=(
            {"chart_type": "histogram", "x": plan.metric} if plan.metric else None
        ),
    )


def _handle_distribution(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    metric = plan.metric or (numeric_columns(df)[0] if numeric_columns(df) else None)
    if metric is None:
        return AnalysisResult.failure(
            "This dataset has no numeric column whose distribution could be shown."
        )

    column = get_column_summary(df, metric)
    dimension = plan.primary_dimension

    summary: dict[str, Any] = {
        "Values": column.count,
        "Mean": _as_number(column.mean),
        "Median": _as_number(column.median),
        "Standard deviation": _as_number(column.std),
        "Minimum": _as_number(column.min),
        "25th percentile": _as_number(column.q1),
        "75th percentile": _as_number(column.q3),
        "Maximum": _as_number(column.max),
        "Possible outliers": column.outlier_count,
    }
    if column.skew is not None:
        summary["Skew"] = round(column.skew, 3)
    if column.missing_count:
        summary["Missing values"] = column.missing_count

    table: list[dict[str, Any]] | None = None
    shape = "distribution"
    if dimension:
        grouped = group_and_aggregate(
            df, dimension,
            {metric: ["count", "mean", "median", "min", "max", "std"]},
            sort_by=f"{metric}_mean",
        )
        table = _records(grouped)
        shape = "distribution_by_group"

    return AnalysisResult(
        success=True,
        title=f"Distribution of {lower_label(metric)}",
        summary_data=summary,
        table_data=table,
        metadata={
            "tools": ["get_column_summary"] + (
                ["group_and_aggregate"] if dimension else []
            ),
            "metric": metric,
            "dimension": dimension,
            "result_shape": shape,
        },
        chart_spec=(
            {"chart_type": "box", "x": dimension, "y": metric}
            if dimension
            else {"chart_type": "histogram", "x": metric}
        ),
    )


def _handle_correlation(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    result = calculate_correlation(df, threshold=0.0)
    if result.is_empty:
        return AnalysisResult.failure(
            "At least two numeric columns with variation are needed to measure "
            "correlation; this dataset does not have them.",
            notes=list(result.notes),
        )

    pairs = result.pairs
    focus = plan.metric
    if focus:
        pairs = [p for p in pairs if focus in (p["left"], p["right"])]
        if not pairs:
            return AnalysisResult.failure(
                f"`{focus}` could not be correlated with any other numeric "
                "column in this dataset.",
                notes=list(result.notes),
            )

    limited = pairs[: plan.limit or 10]
    strongest = limited[0]

    summary: dict[str, Any] = {
        "Strongest pair": f"{strongest['left']} and {strongest['right']}",
        "Correlation (r)": strongest["correlation"],
        "Direction": strongest["direction"],
        "Strength": strongest["strength"],
        "Numeric columns compared": len(result.columns),
        "Pairs examined": len(result.pairs),
    }
    if focus:
        summary["Correlated with"] = focus

    table = [
        {
            "Column A": p["left"],
            "Column B": p["right"],
            "Correlation (r)": p["correlation"],
            "Direction": p["direction"],
            "Strength": p["strength"],
        }
        for p in limited
    ]

    return AnalysisResult(
        success=True,
        title=(
            f"Correlations with {lower_label(focus)}" if focus
            else "Correlations between numeric columns"
        ),
        summary_data=summary,
        table_data=table,
        metadata={
            "tools": ["calculate_correlation"],
            "method": result.method,
            "metric": focus,
            "columns": result.columns,
            "notes": list(result.notes)
            + ["Correlation measures association, not causation."],
            "result_shape": "correlation_matrix",
        },
        chart_spec={"chart_type": "heatmap"},
    )


def _handle_count(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    dimension = _dimension(plan, df)
    if dimension is None:
        return AnalysisResult(
            success=True,
            title="Row count",
            summary_data={"Rows": int(len(df))},
            metadata={"tools": [], "result_shape": "scalar"},
        )

    counts = group_and_aggregate(df, dimension, "count", ascending=False)
    total = int(counts["row_count"].sum())
    top = counts.iloc[0]

    frame = counts.rename(columns={"row_count": "Rows"}).copy()
    frame["Share %"] = (frame["Rows"] / total * 100).round(2) if total else 0.0

    return AnalysisResult(
        success=True,
        title=f"Row counts by {lower_label(dimension)}",
        summary_data={
            f"Distinct {lower_label(dimension)} values": int(len(counts)),
            "Rows counted": total,
            f"Most common {lower_label(dimension)}": str(top[dimension]),
            "Rows in the most common group": int(top["row_count"]),
            "Share of the most common group (%)": round(
                int(top["row_count"]) / total * 100, 2
            ) if total else 0.0,
        },
        table_data=_records(frame),
        metadata={
            "tools": ["group_and_aggregate"],
            "dimension": dimension,
            "aggregation": "count",
            "result_shape": "ranked",
        },
        chart_spec={
            "chart_type": "bar",
            "x": dimension,
            "aggregation": "count",
            "top_n": plan.limit or 20,
        },
    )


def _handle_anomaly(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    metric = plan.metric or (numeric_columns(df)[0] if numeric_columns(df) else None)
    if metric is None:
        return AnalysisResult.failure(
            "This dataset has no numeric column in which to look for outliers."
        )

    outliers = detect_outliers(df, metric, method="iqr", max_rows=plan.limit or 25)

    summary: dict[str, Any] = {
        "Column examined": metric,
        "Values examined": outliers.values_considered,
        "Outliers found": outliers.count,
        "Share of values (%)": outliers.pct,
        "Method": "1.5 x IQR fence",
    }
    if outliers.lower_bound is not None:
        summary["Expected range (low)"] = _as_number(outliers.lower_bound)
        summary["Expected range (high)"] = _as_number(outliers.upper_bound)

    return AnalysisResult(
        success=True,
        title=f"Possible outliers in {lower_label(metric)}",
        summary_data=summary,
        table_data=_records(outliers.rows),
        metadata={
            "tools": ["detect_outliers"],
            "metric": metric,
            "method": outliers.method,
            "threshold": outliers.threshold,
            "notes": list(outliers.notes),
            "result_shape": "outliers",
        },
        chart_spec={"chart_type": "box", "y": metric},
    )


def _handle_dataset_question(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    schema = get_dataset_schema(df, name="dataset")

    return AnalysisResult(
        success=True,
        title="What this dataset contains",
        summary_data={
            "Rows": schema.row_count,
            "Columns": schema.column_count,
            "Numeric columns": len(schema.numeric_columns),
            "Categorical columns": len(schema.categorical_columns),
            "Date columns": len(schema.date_columns),
        },
        table_data=[
            {
                "Column": column.name,
                "Type": column.inferred_type,
                "Missing %": round(column.missing_pct, 2),
                "Distinct values": column.unique_count,
                "Examples": ", ".join(str(v) for v in column.sample_values[:3]),
            }
            for column in schema.columns
        ],
        metadata={
            "tools": ["get_dataset_schema"],
            "roles": {
                "numeric": schema.numeric_columns,
                "categorical": schema.categorical_columns,
                "date": schema.date_columns,
                "identifier": schema.identifier_columns,
            },
            "result_shape": "schema",
        },
    )


#: The intent router. The LLM picks a key; this maps it to Python.
HANDLERS: dict[Intent, Callable[[pd.DataFrame, AnalysisPlan], AnalysisResult]] = {
    Intent.SUMMARY: _handle_summary,
    Intent.RANKING: _handle_ranking,
    Intent.COMPARISON: _handle_comparison,
    Intent.SEGMENTATION: _handle_comparison,
    Intent.TREND: _handle_trend,
    Intent.DISTRIBUTION: _handle_distribution,
    Intent.CORRELATION: _handle_correlation,
    Intent.TIME_COMPARISON: _handle_time_comparison,
    Intent.PERCENTAGE_CHANGE: _handle_percentage_change,
    Intent.COUNT: _handle_count,
    Intent.ANOMALY: _handle_anomaly,
    Intent.DATASET_QUESTION: _handle_dataset_question,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def execute_plan(df: pd.DataFrame, plan: AnalysisPlan) -> AnalysisResult:
    """Run a validated plan and return the computed result.

    Never raises: a tool failure becomes a failed :class:`AnalysisResult`
    carrying a message the UI can show, with the technical detail logged.
    """
    require_dataframe(df)

    handler = HANDLERS.get(plan.intent)
    if handler is None:
        return AnalysisResult.failure(
            f"Aurevia does not yet support '{plan.intent}' analysis."
        )

    try:
        working, filter_meta = _apply_filters(df, plan)
        working, window_meta = _apply_time_window(working, plan)
        filter_meta.update(window_meta)
    except ToolError as exc:
        logger.info("Filtering failed: %s", exc)
        return AnalysisResult.failure(str(exc))

    if working.empty:
        described = ", ".join(describe_conditions(plan.filter_conditions()))
        return AnalysisResult.failure(
            "Aurevia understood the request, but no rows match the filters "
            f"({described})."
        )

    try:
        result = handler(working, plan)
    except ToolError as exc:
        logger.info("Analysis tool rejected the plan: %s", exc)
        return AnalysisResult.failure(str(exc))
    except Exception as exc:  # noqa: BLE001 - the UI must never see a traceback
        logger.exception("Unexpected failure executing a %s plan", plan.intent)
        return AnalysisResult.failure(
            "The analysis failed unexpectedly. The details were logged.",
            exception=type(exc).__name__,
        )

    if not result.success:
        result.metadata.setdefault("intent", str(plan.intent))
        return result

    # Record provenance and settle the chart choice.
    result.metadata.update(filter_meta)
    result.metadata.setdefault("intent", str(plan.intent))
    result.metadata["rows_analysed"] = int(len(working))

    shape = str(result.metadata.get("result_shape", "none"))
    chart_type = select_visualization(
        intent=plan.intent,
        requested=plan.visualization,
        shape=shape,
        row_count=result.row_count or len(result.summary_data),
    )
    if chart_type is None:
        result.chart_spec = None
    elif result.chart_spec is not None:
        original = result.chart_spec.get("chart_type")
        if original != chart_type:
            result.metadata["chart_overridden_from"] = original
        result.chart_spec = normalize_chart_spec(result.chart_spec, chart_type)
    result.metadata["visualization"] = chart_type
    if plan.visualization and plan.visualization != chart_type:
        result.metadata["visualization_requested"] = plan.visualization

    return result


__all__ = [
    "COLUMN_DRAWN_SHAPES",
    "HANDLERS",
    "MAX_TABLE_ROWS",
    "execute_plan",
    "normalize_chart_spec",
    "select_visualization",
]
