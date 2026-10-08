"""The Explore page.

Guided exploratory analysis, built entirely on :mod:`app.tools`. The page is
organised as five tabs rather than a wall of charts: each one answers a specific
question, and the heavier visuals are driven by a selector so exactly one chart
is drawn per question instead of one per column.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.config import Settings, get_settings
from app.services.visualization import bar_chart, box_plot, build_chart, empty_figure
from app.tools import (
    ToolError,
    calculate_correlation,
    calculate_statistics,
    calculate_time_trend,
    categorical_columns,
    compare_segments,
    datetime_columns,
    detect_outliers,
    generate_chart_data,
    get_column_summary,
    get_dataset_schema,
    numeric_columns,
    rank_values,
    supported_frequencies,
)
from app.tools.charts import humanize
from app.ui.components import render_chart
from app.ui.state import (
    get_dataframe,
    get_load_result,
    get_profile,
    has_dataset,
    render_no_dataset_notice,
)

logger = logging.getLogger(__name__)

#: Correlation strength we consider worth surfacing on this page.
CORRELATION_THRESHOLD = 0.3
#: Categories shown in a breakdown before the tail is dropped.
TOP_CATEGORIES = 12


def _tool_guard(label: str, func, *args, **kwargs):
    """Run a tool, turning a ToolError into an inline message, not a traceback."""
    try:
        return func(*args, **kwargs)
    except ToolError as exc:
        st.warning(f"{label}: {exc}")
    except Exception as exc:  # noqa: BLE001 - the page must stay usable
        logger.exception("Unexpected failure in %s", label)
        st.error(f"{label} failed unexpectedly: {exc}")
    return None


# --------------------------------------------------------------------------- #
# Tab 1 - Summary
# --------------------------------------------------------------------------- #

def _render_summary(df: pd.DataFrame) -> None:
    schema = get_dataset_schema(df, name="dataset")

    st.subheader("Field roles")
    roles = {
        "Numeric": schema.numeric_columns,
        "Categorical": schema.categorical_columns,
        "Boolean": schema.boolean_columns,
        "Date / time": schema.date_columns,
        "Identifier": schema.identifier_columns,
    }
    columns = st.columns(len(roles))
    for container, (label, names) in zip(columns, roles.items()):
        container.metric(label, len(names))
        container.caption(", ".join(names[:4]) + (" …" if len(names) > 4 else "") or "—")

    st.subheader("Summary statistics")
    stats = _tool_guard("Statistics", calculate_statistics, df)
    if stats is None:
        return
    if stats.is_empty:
        st.info("No numeric columns to summarise.")
    else:
        frame = stats.to_frame()
        frame.insert(0, "Column", frame.pop("column").map(humanize))
        st.dataframe(
            frame.style.format(precision=2, thousands=","),
            width="stretch",
            hide_index=True,
        )
    if stats.skipped:
        with st.expander(f"{len(stats.skipped)} column(s) skipped"):
            for column, reason in stats.skipped.items():
                st.caption(f"`{column}` — {reason}")


# --------------------------------------------------------------------------- #
# Tab 2 - Data quality
# --------------------------------------------------------------------------- #

def _render_quality(df: pd.DataFrame) -> None:
    profile = get_profile()

    st.subheader("Missing values")
    missing = [
        {"Column": c.name, "Missing": c.missing_count, "Missing %": round(c.missing_pct, 2)}
        for c in profile.columns
        if c.missing_count
    ]
    if not missing:
        st.success("No missing values anywhere in the dataset.")
    else:
        frame = pd.DataFrame(missing).sort_values("Missing", ascending=False)
        left, right = st.columns([1, 1.3])
        with left:
            st.dataframe(frame, width="stretch", hide_index=True)
        with right:
            data = generate_chart_data(
                frame.rename(columns={"Column": "column", "Missing %": "missing_pct"}),
                "bar",
                x="column",
                y="missing_pct",
                aggregation="max",
                title="Missing values by column (%)",
            )
            data.y_label = "Missing (%)"
            render_chart(bar_chart(data))

    st.subheader("Duplicates")
    if profile.duplicate_row_count:
        st.warning(
            f"{profile.duplicate_row_count:,} duplicate row(s) "
            f"({profile.duplicate_row_pct:.2f}% of the dataset)."
        )
    else:
        st.success("No duplicate rows.")

    st.subheader("Date ranges")
    dates = datetime_columns(df)
    if not dates:
        st.info("No date column was detected, so there is no time span to report.")
    else:
        rows = []
        for column in dates:
            summary = get_column_summary(df, column)
            rows.append(
                {
                    "Column": column,
                    "From": summary.min_date,
                    "To": summary.max_date,
                    "Span (days)": summary.date_range_days,
                    "Missing": summary.missing_count,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 3 - Distributions
# --------------------------------------------------------------------------- #

def _render_distributions(df: pd.DataFrame) -> None:
    numeric = numeric_columns(df)
    if not numeric:
        st.info("No numeric columns to chart.")
        return

    left, right = st.columns([2, 1])
    column = left.selectbox("Numeric column", numeric, key="explore_dist_column")
    bins = right.slider("Bins", min_value=10, max_value=100, value=30, step=5,
                        key="explore_dist_bins")

    summary = _tool_guard("Column summary", get_column_summary, df, column)
    if summary is None:
        return

    tiles = st.columns(5)
    tiles[0].metric("Mean", f"{summary.mean:,.2f}" if summary.mean is not None else "—")
    tiles[1].metric("Median", f"{summary.median:,.2f}" if summary.median is not None else "—")
    tiles[2].metric("Std. dev", f"{summary.std:,.2f}" if summary.std is not None else "—")
    tiles[3].metric(
        "Range",
        f"{summary.min:,.0f} … {summary.max:,.0f}"
        if summary.min is not None else "—",
    )
    tiles[4].metric("Outliers", f"{summary.outlier_count:,}")

    histogram = _tool_guard(
        "Histogram", generate_chart_data, df, "histogram", x=column, bins=bins
    )
    if histogram is not None:
        render_chart(build_chart(histogram, bins=bins))

    # A box plot split by a category is where distributions get interesting.
    groupers = categorical_columns(df, max_unique=TOP_CATEGORIES)
    if groupers:
        split = st.selectbox(
            "Split by (optional)",
            ["— none —", *groupers],
            key="explore_dist_split",
        )
        box = _tool_guard(
            "Box plot",
            generate_chart_data,
            df,
            "box",
            x=None if split == "— none —" else split,
            y=column,
        )
        if box is not None:
            render_chart(box_plot(box))


# --------------------------------------------------------------------------- #
# Tab 4 - Categories
# --------------------------------------------------------------------------- #

def _render_categories(df: pd.DataFrame) -> None:
    groupers = categorical_columns(df, max_unique=200)
    if not groupers:
        st.info("No categorical columns to break down.")
        return

    numeric = numeric_columns(df)
    left, middle, right = st.columns(3)
    column = left.selectbox("Category", groupers, key="explore_cat_column")
    metric = middle.selectbox(
        "Measure", ["Row count", *numeric], key="explore_cat_metric"
    )
    aggregation = right.selectbox(
        "Aggregation",
        ["sum", "mean", "median", "min", "max"],
        index=1,
        key="explore_cat_agg",
        disabled=metric == "Row count",
    )

    metric_column = None if metric == "Row count" else metric
    ranking = _tool_guard(
        "Ranking",
        rank_values,
        df,
        column,
        metric_column,
        aggregation="count" if metric_column is None else aggregation,
        top_n=TOP_CATEGORIES,
    )
    if ranking is None:
        return

    if ranking.is_empty:
        st.info("Nothing to rank for this selection.")
        return

    st.subheader(f"Top {len(ranking.dataframe)} by {humanize(column).lower()}")
    frame = ranking.dataframe.rename(
        columns={"rank": "Rank", column: humanize(column), "value": "Value",
                 "row_count": "Rows"}
    )
    left, right = st.columns([1, 1.3])
    with left:
        st.dataframe(
            frame.style.format({"Value": "{:,.2f}", "Rows": "{:,.0f}"}),
            width="stretch",
            hide_index=True,
        )
    with right:
        chart = _tool_guard(
            "Breakdown chart",
            generate_chart_data,
            df,
            "bar",
            x=column,
            y=metric_column,
            aggregation="count" if metric_column is None else aggregation,
            top_n=TOP_CATEGORIES,
        )
        render_chart(build_chart(chart) if chart is not None else empty_figure())
    for note in ranking.notes:
        st.caption(note)

    # Segment comparison against the overall baseline.
    if numeric:
        st.subheader("Segment comparison")
        measure = st.selectbox(
            "Compare which measure across segments?",
            numeric,
            key="explore_segment_metric",
        )
        comparison = _tool_guard(
            "Segment comparison",
            compare_segments,
            df,
            column,
            measure,
            aggregation="mean",
            top_n=TOP_CATEGORIES,
        )
        if comparison is not None and not comparison.is_empty:
            st.caption(
                f"Baseline: overall average {humanize(measure).lower()} = "
                f"{comparison.baseline_value:,.2f}"
            )
            table = comparison.dataframe.rename(
                columns={
                    "segment": humanize(column),
                    "row_count": "Rows",
                    "value": "Average",
                    "share_pct": "Share %",
                    "difference": "vs baseline",
                    "difference_pct": "vs baseline %",
                }
            )
            st.dataframe(
                table.style.format(
                    {
                        "Rows": "{:,.0f}",
                        "Average": "{:,.2f}",
                        "Share %": "{:,.1f}",
                        "vs baseline": "{:+,.2f}",
                        "vs baseline %": "{:+,.1f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


# --------------------------------------------------------------------------- #
# Tab 5 - Relationships
# --------------------------------------------------------------------------- #

def _render_relationships(df: pd.DataFrame) -> None:
    st.subheader("Correlations")
    result = _tool_guard(
        "Correlation", calculate_correlation, df, threshold=CORRELATION_THRESHOLD
    )
    if result is None:
        return

    for note in result.notes:
        st.caption(note)

    if result.is_empty:
        st.info("At least two numeric columns with variation are needed.")
    else:
        left, right = st.columns([1.2, 1])
        with left:
            heatmap = generate_chart_data(df, "heatmap")
            render_chart(build_chart(heatmap))
        with right:
            if not result.pairs:
                st.info(
                    f"No pair reaches |r| ≥ {CORRELATION_THRESHOLD}; the numeric "
                    "columns look largely independent."
                )
            else:
                st.caption(f"Pairs with |r| ≥ {CORRELATION_THRESHOLD}, strongest first.")
                pairs = pd.DataFrame(result.pairs).rename(
                    columns={
                        "left": "Column A",
                        "right": "Column B",
                        "correlation": "r",
                        "strength": "Strength",
                    }
                )[["Column A", "Column B", "r", "Strength"]]
                st.dataframe(
                    pairs.style.format({"r": "{:+.3f}"}),
                    width="stretch",
                    hide_index=True,
                )

    st.subheader("Trend over time")
    dates = datetime_columns(df)
    numeric = numeric_columns(df)
    if not dates:
        st.info("No date column was detected, so no trend can be drawn.")
    else:
        left, middle, right = st.columns(3)
        date_column = left.selectbox("Date column", dates, key="explore_trend_date")
        measure = middle.selectbox(
            "Measure", ["Row count", *numeric], key="explore_trend_metric"
        )
        frequency = right.selectbox(
            "Frequency",
            supported_frequencies(),
            index=supported_frequencies().index("monthly"),
            key="explore_trend_freq",
        )
        metric_column = None if measure == "Row count" else measure
        trend = _tool_guard(
            "Time trend",
            calculate_time_trend,
            df,
            date_column,
            metric_column,
            aggregation="sum",
            frequency=frequency,
        )
        if trend is not None:
            if trend.is_empty:
                st.info("No rows with a usable date, so the trend is empty.")
            else:
                chart = generate_chart_data(
                    df,
                    "line",
                    x=date_column,
                    y=metric_column,
                    aggregation="sum",
                    frequency=frequency,
                )
                render_chart(build_chart(chart))
                tiles = st.columns(3)
                tiles[0].metric("Periods", f"{len(trend.dataframe):,}")
                tiles[1].metric(
                    "First → last",
                    f"{trend.first_period} → {trend.last_period}",
                )
                change = trend.total_change_pct
                tiles[2].metric(
                    "Total change",
                    f"{change:+.1f}%" if change is not None else "—",
                )
            for note in trend.notes:
                st.caption(note)

    st.subheader("Outliers")
    if not numeric:
        st.info("No numeric columns to check for outliers.")
        return

    left, middle, right = st.columns(3)
    column = left.selectbox("Column", numeric, key="explore_outlier_column")
    method = middle.selectbox("Method", ["iqr", "zscore"], key="explore_outlier_method")
    threshold = right.number_input(
        "Threshold",
        min_value=0.5,
        max_value=10.0,
        value=1.5 if method == "iqr" else 3.0,
        step=0.5,
        key=f"explore_outlier_threshold_{method}",
        help="IQR fence multiplier, or the z-score cut-off.",
    )

    outliers = _tool_guard(
        "Outlier detection",
        detect_outliers,
        df,
        column,
        method=method,
        threshold=float(threshold),
        max_rows=50,
    )
    if outliers is None:
        return

    tiles = st.columns(3)
    tiles[0].metric("Outliers", f"{outliers.count:,}", delta=f"{outliers.pct:.2f}%",
                    delta_color="off")
    tiles[1].metric(
        "Lower bound",
        f"{outliers.lower_bound:,.2f}" if outliers.lower_bound is not None else "—",
    )
    tiles[2].metric(
        "Upper bound",
        f"{outliers.upper_bound:,.2f}" if outliers.upper_bound is not None else "—",
    )
    for note in outliers.notes:
        st.caption(note)

    if outliers.is_empty:
        st.success(f"No outliers found in `{column}` with this method.")
    else:
        st.caption("The most extreme rows, furthest from the fence first.")
        st.dataframe(outliers.rows, width="stretch")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def render_explore(settings: Settings | None = None) -> None:
    """Render the Explore page."""
    settings = settings or get_settings()

    st.title("Explore")
    st.caption(
        "Exploratory analysis of the loaded dataset — all computed "
        "deterministically, no AI involved."
    )

    if not has_dataset():
        render_no_dataset_notice("Explore")
        return

    df = get_dataframe()
    result = get_load_result()
    st.caption(f"**{result.source_name}** · {result.rows:,} rows × {result.columns} columns")

    summary, quality, distributions, categories, relationships = st.tabs(
        ["Summary", "Data quality", "Distributions", "Categories", "Relationships"]
    )
    with summary:
        _render_summary(df)
    with quality:
        _render_quality(df)
    with distributions:
        _render_distributions(df)
    with categories:
        _render_categories(df)
    with relationships:
        _render_relationships(df)
