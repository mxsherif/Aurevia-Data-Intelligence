"""Chart data preparation.

:func:`generate_chart_data` turns an analysis request into a plot-ready tidy
frame plus the metadata a renderer needs (axis roles, labels, ordering). It is
deliberately *explicit*: each chart type declares what it requires, and an
invalid combination raises instead of silently falling back to some other chart.

The result is consumed by :mod:`app.services.visualization`, which does the
Plotly work. Keeping the two apart means chart *content* is testable without a
rendering engine, and future agents can request data without drawing anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from app.tools.aggregations import resolve_aggregation, supported_aggregations
from app.tools.analysis import calculate_correlation, group_and_aggregate
from app.tools.exceptions import (
    InvalidParameterError,
    UnsupportedOperationError,
)
from app.tools.timeseries import calculate_time_trend, resolve_frequency
from app.tools.validation import (
    categorical_columns,
    numeric_columns,
    require_column,
    require_dataframe,
    require_numeric_column,
    require_positive_int,
)


@dataclass(frozen=True)
class ChartSpec:
    """What a chart type needs, and what it does with it."""

    name: str
    label: str
    needs_x: bool = True
    needs_y: bool = True
    #: y must be numeric (false for count-style charts).
    numeric_y: bool = True
    #: x must be numeric (scatter, histogram).
    numeric_x: bool = False
    #: x may be a date column and will then be ordered chronologically.
    supports_time_axis: bool = False
    #: Rows are aggregated by x (and group) before plotting.
    aggregates: bool = False
    supports_grouping: bool = True
    description: str = ""


CHART_SPECS: dict[str, ChartSpec] = {
    "line": ChartSpec(
        "line", "Line chart",
        supports_time_axis=True, aggregates=True,
        description="Trend of a measure over time or an ordered dimension.",
    ),
    "bar": ChartSpec(
        "bar", "Bar chart",
        aggregates=True,
        description="Compare an aggregated measure across categories.",
    ),
    "scatter": ChartSpec(
        "scatter", "Scatter plot",
        numeric_x=True,
        description="Relationship between two numeric columns, row by row.",
    ),
    "histogram": ChartSpec(
        "histogram", "Histogram",
        needs_y=False, numeric_x=True, numeric_y=False,
        description="Distribution of a single numeric column.",
    ),
    "box": ChartSpec(
        "box", "Box plot",
        needs_x=False,
        description="Spread and outliers of a numeric column, optionally by category.",
    ),
    "heatmap": ChartSpec(
        "heatmap", "Correlation heatmap",
        needs_x=False, needs_y=False, numeric_y=False, supports_grouping=False,
        description="Correlation matrix across the numeric columns.",
    ),
}

CHART_ALIASES: dict[str, str] = {
    "line_chart": "line",
    "lines": "line",
    "trend": "line",
    "bar_chart": "bar",
    "bars": "bar",
    "column": "bar",
    "scatter_plot": "scatter",
    "scatterplot": "scatter",
    "points": "scatter",
    "hist": "histogram",
    "distribution": "histogram",
    "boxplot": "box",
    "box_plot": "box",
    "correlation": "heatmap",
    "correlation_heatmap": "heatmap",
    "corr": "heatmap",
}

#: Rows above which a scatter plot is sampled, to keep the browser responsive.
SCATTER_SAMPLE_LIMIT = 5_000
#: Categories kept on a bar chart before the rest are grouped into "Other".
DEFAULT_MAX_CATEGORIES = 20


def resolve_chart_type(chart_type: Any) -> ChartSpec:
    """Look up a chart type by name or alias, case-insensitively."""
    if isinstance(chart_type, ChartSpec):
        return chart_type
    if not isinstance(chart_type, str):
        raise UnsupportedOperationError("chart type", chart_type, CHART_SPECS)
    key = chart_type.strip().lower().replace(" ", "_").replace("-", "_")
    key = CHART_ALIASES.get(key, key)
    if key not in CHART_SPECS:
        raise UnsupportedOperationError("chart type", chart_type, CHART_SPECS)
    return CHART_SPECS[key]


def supported_chart_types() -> list[str]:
    return list(CHART_SPECS)


@dataclass
class ChartData:
    """A plot-ready frame plus the roles and labels for its axes."""

    chart_type: str
    #: Tidy data: one row per mark.
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    x: str | None = None
    y: str | None = None
    color: str | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    #: ``"category"``, ``"numeric"``, or ``"time"`` -- drives axis formatting.
    x_kind: str = "category"
    aggregation: str | None = None
    frequency: str | None = None
    #: Explicit category order; renderers must not re-sort.
    category_order: list[Any] | None = None
    #: Extra columns to show on hover.
    hover_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.dataframe.empty

    @property
    def point_count(self) -> int:
        return int(len(self.dataframe))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_kind": self.x_kind,
            "aggregation": self.aggregation,
            "frequency": self.frequency,
            "point_count": self.point_count,
            "category_order": self.category_order,
            "data": self.dataframe.to_dict("records"),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

#: Tokens that should keep their conventional casing in an axis label.
UNIT_TOKENS = {
    "gb": "GB", "mb": "MB", "tb": "TB", "kb": "KB",
    "id": "ID", "url": "URL", "arpu": "ARPU", "usd": "USD", "eur": "EUR",
    "pct": "%", "qty": "qty", "avg": "avg",
}

#: The subset of UNIT_TOKENS rendered as a parenthesised suffix.
PARENTHESISED_UNITS = {"gb", "mb", "tb", "kb", "usd", "eur", "pct"}


def humanize(name: str) -> str:
    """``data_usage_gb`` -> ``Data usage (GB)``, ``monthly_charge`` -> ``Monthly charge``."""
    raw = str(name).strip()
    if not raw:
        return raw
    if raw.isupper():
        return raw

    words = [w for w in raw.replace("-", "_").split("_") if w]
    if not words:
        return raw

    # A trailing *unit of measure* reads better in parentheses; an acronym
    # like "customer_id" does not ("Customer ID", never "Customer (ID)").
    suffix = ""
    if len(words) > 1 and words[-1].lower() in PARENTHESISED_UNITS:
        suffix = f" ({UNIT_TOKENS[words[-1].lower()]})"
        words = words[:-1]

    rendered = [UNIT_TOKENS.get(w.lower(), w) for w in words]
    text = " ".join(rendered)
    return text[0].upper() + text[1:] + suffix


def lower_label(name: str) -> str:
    """:func:`humanize` for mid-sentence use, preserving units and acronyms.

    ``data_usage_gb`` -> ``data usage (GB)`` -- a plain ``.lower()`` would ruin
    the unit.
    """
    label = humanize(name)
    if not label:
        return label
    first = label.split(" ", 1)[0]
    if first.isupper() and len(first) > 1:
        return label
    return label[0].lower() + label[1:]


def _measure_label(column: str | None, aggregation: str | None) -> str:
    """Axis label for an aggregated measure, e.g. ``Average data usage (GB)``."""
    if column is None:
        return "Number of rows"
    if aggregation is None:
        return humanize(column)
    aggregator = resolve_aggregation(aggregation)
    text = aggregator.measure_template.format(
        label=aggregator.label, measure=lower_label(column)
    )
    return text[0].upper() + text[1:] if text else text


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def generate_chart_data(
    df: pd.DataFrame,
    chart_type: str,
    *,
    x: str | None = None,
    y: str | None = None,
    group_by: str | None = None,
    aggregation: str | None = None,
    frequency: str = "monthly",
    top_n: int | None = None,
    bins: int = 30,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    correlation_method: str = "pearson",
    title: str | None = None,
) -> ChartData:
    """Prepare the data for `chart_type` and return it with its axis metadata.

    The chart type decides what the other arguments mean, and an unusable
    combination raises :class:`~app.tools.exceptions.InvalidParameterError`
    rather than quietly drawing something else.
    """
    require_dataframe(df)
    spec = resolve_chart_type(chart_type)

    if group_by is not None and not spec.supports_grouping:
        raise InvalidParameterError(
            f"A {spec.label.lower()} does not support grouping; drop 'group_by'."
        )

    builders = {
        "line": _build_line,
        "bar": _build_bar,
        "scatter": _build_scatter,
        "histogram": _build_histogram,
        "box": _build_box,
        "heatmap": _build_heatmap,
    }
    data = builders[spec.name](
        df,
        spec=spec,
        x=x,
        y=y,
        group_by=group_by,
        aggregation=aggregation,
        frequency=frequency,
        top_n=top_n,
        bins=bins,
        max_categories=max_categories,
        correlation_method=correlation_method,
    )

    if title:
        data.title = title
    if data.is_empty and not data.notes:
        data.notes.append("The selection produced no data to plot.")
    return data


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _build_line(df, *, spec, x, y, group_by, aggregation, frequency, **_) -> ChartData:
    if x is None:
        raise InvalidParameterError(
            "A line chart needs an x-axis column (a date or an ordered measure)."
        )
    x_name = require_column(df, x)
    aggregator = resolve_aggregation(aggregation or ("count" if y is None else "sum"))
    y_name = require_column(df, y) if y is not None else None

    is_time_axis = _is_time_like(df, x_name)

    if is_time_axis and group_by is None:
        trend = calculate_time_trend(
            df, x_name, y_name, aggregation=aggregator.name, frequency=frequency
        )
        frame = trend.dataframe.copy()
        if not frame.empty:
            frame = frame.rename(columns={"period": x_name, "value": "value"})
        data = ChartData(
            chart_type="line",
            dataframe=frame,
            x=x_name,
            y="value",
            title=(
                f"{_measure_label(y_name, aggregator.name)} by "
                f"{resolve_frequency(frequency).axis_label.lower()}"
            ),
            x_label=resolve_frequency(frequency).axis_label,
            y_label=_measure_label(y_name, aggregator.name),
            x_kind="time",
            aggregation=aggregator.name,
            frequency=resolve_frequency(frequency).name,
            hover_columns=["row_count", "pct_change"],
            notes=list(trend.notes),
        )
        return data

    if is_time_axis:
        group_name = require_column(df, group_by)
        frame = _grouped_time_series(
            df, x_name, y_name, group_name, aggregator.name, frequency
        )
        return ChartData(
            chart_type="line",
            dataframe=frame,
            x=x_name,
            y="value",
            color=group_name,
            title=(
                f"{_measure_label(y_name, aggregator.name)} by "
                f"{resolve_frequency(frequency).axis_label.lower()} and "
                f"{lower_label(group_name)}"
            ),
            x_label=resolve_frequency(frequency).axis_label,
            y_label=_measure_label(y_name, aggregator.name),
            x_kind="time",
            aggregation=aggregator.name,
            frequency=resolve_frequency(frequency).name,
        )

    # Non-temporal x: aggregate per x value and order by x so the line is sane.
    keys = [x_name] + ([require_column(df, group_by)] if group_by else [])
    frame = _aggregate(df, keys, y_name, aggregator.name)
    frame = frame.sort_values(x_name, kind="stable").reset_index(drop=True)
    return ChartData(
        chart_type="line",
        dataframe=frame,
        x=x_name,
        y="value",
        color=keys[1] if len(keys) > 1 else None,
        title=f"{_measure_label(y_name, aggregator.name)} by {lower_label(x_name)}",
        x_label=humanize(x_name),
        y_label=_measure_label(y_name, aggregator.name),
        x_kind="numeric" if _is_numeric(df, x_name) else "category",
        aggregation=aggregator.name,
        category_order=frame[x_name].tolist() if not frame.empty else None,
    )


def _build_bar(
    df, *, spec, x, y, group_by, aggregation, top_n, max_categories, **_
) -> ChartData:
    if x is None:
        raise InvalidParameterError("A bar chart needs an x-axis column to group by.")
    x_name = require_column(df, x)
    aggregator = resolve_aggregation(aggregation or ("count" if y is None else "sum"))
    y_name = require_column(df, y) if y is not None else None
    group_name = require_column(df, group_by) if group_by else None

    keys = [x_name] + ([group_name] if group_name else [])
    frame = _aggregate(df, keys, y_name, aggregator.name)

    notes: list[str] = []
    limit = require_positive_int(top_n, "top_n") if top_n is not None else None
    cap = require_positive_int(max_categories, "max_categories", maximum=200)

    if not frame.empty:
        # Order categories by the measure, keeping groups of a category together.
        totals = (
            frame.groupby(x_name, observed=True)["value"].sum().sort_values(ascending=False)
        )
        keep = list(totals.index[: limit or cap])
        if len(totals) > len(keep):
            notes.append(
                f"Showing the top {len(keep)} of {len(totals):,} "
                f"{lower_label(x_name)} values."
            )
            frame = frame[frame[x_name].isin(keep)].copy()
        frame[x_name] = pd.Categorical(frame[x_name], categories=keep, ordered=True)
        frame = frame.sort_values(
            [x_name] + ([group_name] if group_name else []), kind="stable"
        ).reset_index(drop=True)
        frame[x_name] = frame[x_name].astype(object)
    else:
        keep = []

    return ChartData(
        chart_type="bar",
        dataframe=frame,
        x=x_name,
        y="value",
        color=group_name,
        title=f"{_measure_label(y_name, aggregator.name)} by {lower_label(x_name)}",
        x_label=humanize(x_name),
        y_label=_measure_label(y_name, aggregator.name),
        x_kind="category",
        aggregation=aggregator.name,
        category_order=keep or None,
        hover_columns=["row_count"] if "row_count" in frame.columns else [],
        notes=notes,
    )


def _build_scatter(df, *, spec, x, y, group_by, **_) -> ChartData:
    if x is None or y is None:
        raise InvalidParameterError(
            "A scatter plot needs both an x and a y column, and both must be numeric."
        )
    x_name = require_column(df, x)
    y_name = require_column(df, y)
    require_numeric_column(df, x_name)
    require_numeric_column(df, y_name)
    group_name = require_column(df, group_by) if group_by else None

    wanted = [x_name, y_name] + ([group_name] if group_name else [])
    frame = df.loc[:, list(dict.fromkeys(wanted))].copy()
    frame[x_name] = pd.to_numeric(frame[x_name], errors="coerce")
    frame[y_name] = pd.to_numeric(frame[y_name], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=[x_name, y_name])
    notes: list[str] = []
    if len(frame) < before:
        notes.append(f"{before - len(frame):,} row(s) with missing values were excluded.")

    if len(frame) > SCATTER_SAMPLE_LIMIT:
        notes.append(
            f"Sampled {SCATTER_SAMPLE_LIMIT:,} of {len(frame):,} rows to keep the "
            "chart responsive."
        )
        frame = frame.sample(SCATTER_SAMPLE_LIMIT, random_state=42).sort_index()

    return ChartData(
        chart_type="scatter",
        dataframe=frame.reset_index(drop=True),
        x=x_name,
        y=y_name,
        color=group_name,
        title=f"{humanize(y_name)} vs {lower_label(x_name)}",
        x_label=humanize(x_name),
        y_label=humanize(y_name),
        x_kind="numeric",
        notes=notes,
    )


def _build_histogram(df, *, spec, x, y, group_by, bins, **_) -> ChartData:
    column = x if x is not None else y
    if column is None:
        raise InvalidParameterError("A histogram needs one numeric column.")
    name = require_column(df, column)
    require_numeric_column(df, name)
    bin_count = require_positive_int(bins, "bins", maximum=200)
    group_name = require_column(df, group_by) if group_by else None

    wanted = [name] + ([group_name] if group_name else [])
    frame = df.loc[:, list(dict.fromkeys(wanted))].copy()
    frame[name] = pd.to_numeric(frame[name], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=[name])
    notes: list[str] = []
    if len(frame) < before:
        notes.append(f"{before - len(frame):,} missing value(s) were excluded.")

    return ChartData(
        chart_type="histogram",
        dataframe=frame.reset_index(drop=True),
        x=name,
        y=None,
        color=group_name,
        title=f"Distribution of {lower_label(name)}",
        x_label=humanize(name),
        y_label="Number of rows",
        x_kind="numeric",
        notes=notes + [f"{bin_count} bins."],
    )


def _build_box(df, *, spec, x, y, group_by, max_categories, **_) -> ChartData:
    if y is None:
        raise InvalidParameterError("A box plot needs a numeric y column.")
    y_name = require_column(df, y)
    require_numeric_column(df, y_name)

    category = x if x is not None else group_by
    category_name = require_column(df, category) if category else None
    cap = require_positive_int(max_categories, "max_categories", maximum=200)

    wanted = [y_name] + ([category_name] if category_name else [])
    frame = df.loc[:, list(dict.fromkeys(wanted))].copy()
    frame[y_name] = pd.to_numeric(frame[y_name], errors="coerce")
    frame = frame.dropna(subset=[y_name])

    notes: list[str] = []
    order: list[Any] | None = None

    if category_name and not frame.empty:
        medians = (
            frame.groupby(category_name, observed=True)[y_name]
            .median()
            .sort_values(ascending=False)
        )
        order = list(medians.index[:cap])
        if len(medians) > len(order):
            notes.append(
                f"Showing the {len(order)} {lower_label(category_name)} values "
                f"with the highest median, of {len(medians):,}."
            )
            frame = frame[frame[category_name].isin(order)].copy()
        frame[category_name] = pd.Categorical(
            frame[category_name], categories=order, ordered=True
        )
        frame = frame.sort_values(category_name, kind="stable")
        frame[category_name] = frame[category_name].astype(object)

    title = (
        f"{humanize(y_name)} by {lower_label(category_name)}"
        if category_name
        else f"Spread of {lower_label(y_name)}"
    )
    return ChartData(
        chart_type="box",
        dataframe=frame.reset_index(drop=True),
        x=category_name,
        y=y_name,
        title=title,
        x_label=humanize(category_name) if category_name else "",
        y_label=humanize(y_name),
        x_kind="category",
        category_order=order,
        notes=notes,
    )


def _build_heatmap(df, *, spec, correlation_method, **_) -> ChartData:
    result = calculate_correlation(df, method=correlation_method)
    matrix = result.matrix

    return ChartData(
        chart_type="heatmap",
        dataframe=matrix.round(4) if not matrix.empty else pd.DataFrame(),
        x=None,
        y=None,
        title=f"Correlation heatmap ({result.method})",
        x_label="",
        y_label="",
        x_kind="category",
        category_order=result.columns or None,
        notes=list(result.notes),
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _aggregate(
    df: pd.DataFrame,
    keys: list[str],
    value_column: str | None,
    aggregation: str,
) -> pd.DataFrame:
    """Group by `keys` and produce a single ``value`` column."""
    spec: Any = "count" if value_column is None else {value_column: aggregation}
    frame = group_and_aggregate(df, keys, spec, ascending=False)
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "value"])
    return frame.rename(columns={frame.columns[-1]: "value"})


def _grouped_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str | None,
    group_column: str,
    aggregation: str,
    frequency: str,
) -> pd.DataFrame:
    """One ordered time series per group, concatenated into a tidy frame."""
    frames: list[pd.DataFrame] = []
    for group_value, chunk in df.groupby(group_column, dropna=True, observed=True):
        trend = calculate_time_trend(
            chunk, date_column, value_column, aggregation=aggregation, frequency=frequency
        )
        if trend.is_empty:
            continue
        part = trend.dataframe.loc[:, ["period", "value", "row_count"]].copy()
        part[group_column] = group_value
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=[date_column, "value", "row_count", group_column])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"period": date_column})
    return combined.sort_values([group_column, date_column], kind="stable").reset_index(
        drop=True
    )


def _is_numeric(df: pd.DataFrame, column: str) -> bool:
    series = df[column]
    return bool(pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series))


def _is_time_like(df: pd.DataFrame, column: str) -> bool:
    """True when `column` can be read as a date (so the axis is chronological)."""
    from app.tools.validation import require_datetime_column

    try:
        require_datetime_column(df, column)
    except Exception:  # noqa: BLE001 - "not a date" is the answer, not an error
        return False
    return True


def suggest_chart_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Column lists the Visualize page uses to populate its selectors."""
    require_dataframe(df)
    from app.tools.validation import datetime_columns

    return {
        "numeric": numeric_columns(df),
        "categorical": categorical_columns(df, max_unique=DEFAULT_MAX_CATEGORIES * 5),
        "datetime": datetime_columns(df),
        "all": [str(c) for c in df.columns],
        "aggregations": supported_aggregations(),
    }
