"""The Visualize page: a manual chart builder.

The control panel is driven by the selected chart type's
:class:`~app.tools.charts.ChartSpec`, so the inputs on screen are exactly the
ones that chart can use -- a histogram never asks for a y-axis, a heatmap asks
for nothing, and only the charts that aggregate show an aggregation selector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from app.config import Settings, get_settings
from app.services.visualization import build_chart, empty_figure
from app.tools import (
    CHART_SPECS,
    ChartSpec,
    ToolError,
    generate_chart_data,
    resolve_chart_type,
    suggest_chart_options,
    supported_aggregations,
    supported_frequencies,
)
from app.tools.charts import DEFAULT_MAX_CATEGORIES
from app.ui.components import render_chart
from app.ui.state import (
    get_dataframe,
    get_load_result,
    has_dataset,
    render_no_dataset_notice,
)

logger = logging.getLogger(__name__)

NONE_LABEL = "— none —"
COUNT_LABEL = "Row count"

#: Aggregations offered in the UI, in a sensible reading order.
UI_AGGREGATIONS = ("sum", "mean", "median", "count", "min", "max", "std")


@dataclass
class ChartRequest:
    """The options collected from the control panel."""

    chart_type: str
    x: str | None = None
    y: str | None = None
    group_by: str | None = None
    aggregation: str | None = None
    frequency: str = "monthly"
    bins: int = 30
    top_n: int | None = None
    max_categories: int = DEFAULT_MAX_CATEGORIES

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "group_by": self.group_by,
            "aggregation": self.aggregation,
            "frequency": self.frequency,
            "bins": self.bins,
            "top_n": self.top_n,
            "max_categories": self.max_categories,
        }


def _pick(label: str, options: list[str], key: str, *, optional: bool = False,
          help_text: str | None = None) -> str | None:
    """A selectbox that returns ``None`` for the optional "none" choice."""
    if not options:
        st.selectbox(label, ["— no suitable column —"], key=key, disabled=True)
        return None
    choices = [NONE_LABEL, *options] if optional else options
    chosen = st.selectbox(label, choices, key=key, help=help_text)
    return None if chosen == NONE_LABEL else chosen


def _render_controls(df: pd.DataFrame, spec: ChartSpec) -> ChartRequest:
    """Render only the controls `spec` can actually use."""
    options = suggest_chart_options(df)
    numeric = options["numeric"]
    categorical = options["categorical"]
    dates = options["datetime"]
    request = ChartRequest(chart_type=spec.name)

    if spec.name == "heatmap":
        st.caption(
            f"The heatmap correlates all {len(numeric)} numeric column(s); "
            "it takes no axis selection."
        )
        return request

    if spec.name == "histogram":
        left, right = st.columns([2, 1])
        with left:
            request.x = _pick("Numeric column", numeric, "viz_hist_x")
        with right:
            request.bins = st.slider("Bins", 10, 100, 30, 5, key="viz_hist_bins")
        request.group_by = _pick(
            "Split by (optional)", categorical, "viz_hist_group", optional=True,
            help_text="Overlays one histogram per category.",
        )
        return request

    if spec.name == "box":
        left, right = st.columns(2)
        with left:
            request.y = _pick("Numeric column", numeric, "viz_box_y")
        with right:
            request.x = _pick(
                "Group by (optional)", categorical, "viz_box_x", optional=True,
                help_text="Categories are ordered by median.",
            )
        return request

    if spec.name == "scatter":
        left, middle, right = st.columns(3)
        with left:
            request.x = _pick("X axis (numeric)", numeric, "viz_scatter_x")
        with middle:
            # Default y to a different column than x where possible.
            y_options = [c for c in numeric if c != request.x] or numeric
            request.y = _pick("Y axis (numeric)", y_options, "viz_scatter_y")
        with right:
            request.group_by = _pick(
                "Colour by (optional)", categorical, "viz_scatter_group", optional=True
            )
        return request

    # line / bar -- the aggregating charts.
    x_options = (dates + [c for c in categorical if c not in dates]) if spec.name == "line" \
        else (categorical + [c for c in dates if c not in categorical])
    if not x_options:
        x_options = options["all"]

    left, middle, right = st.columns(3)
    with left:
        request.x = _pick(
            "X axis",
            x_options,
            f"viz_{spec.name}_x",
            help_text="A date column gives a chronological axis."
            if spec.name == "line" else "The dimension to compare across.",
        )
    with middle:
        measure = st.selectbox(
            "Measure", [COUNT_LABEL, *numeric], key=f"viz_{spec.name}_y"
        )
        request.y = None if measure == COUNT_LABEL else measure
    with right:
        if request.y is None:
            st.selectbox(
                "Aggregation", ["count"], key=f"viz_{spec.name}_agg_disabled",
                disabled=True, help="Row counts need no aggregation.",
            )
            request.aggregation = "count"
        else:
            available = [a for a in UI_AGGREGATIONS if a in supported_aggregations()]
            request.aggregation = st.selectbox(
                "Aggregation", available, index=available.index("sum"),
                key=f"viz_{spec.name}_agg",
            )

    extra_left, extra_middle, extra_right = st.columns(3)
    with extra_left:
        request.group_by = _pick(
            "Group by (optional)", categorical, f"viz_{spec.name}_group", optional=True,
            help_text="Draws one series per category.",
        )
    with extra_middle:
        is_time_axis = request.x in dates
        request.frequency = st.selectbox(
            "Time grouping",
            supported_frequencies(),
            index=supported_frequencies().index("monthly"),
            key=f"viz_{spec.name}_freq",
            disabled=not is_time_axis,
            help="Only applies when the x axis is a date column."
            if not is_time_axis else None,
        )
    with extra_right:
        if spec.name == "bar":
            request.top_n = int(
                st.number_input(
                    "Show top N categories",
                    min_value=1,
                    max_value=100,
                    value=15,
                    step=1,
                    key="viz_bar_top_n",
                )
            )
        else:
            st.empty()

    return request


def _render_chart(df: pd.DataFrame, request: ChartRequest) -> None:
    try:
        data = generate_chart_data(df, request.chart_type, **request.to_kwargs())
    except ToolError as exc:
        st.warning(str(exc))
        render_chart(empty_figure("Adjust the selection above"))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chart preparation failed")
        st.error(f"Could not prepare the chart: {exc}")
        return

    render_kwargs = {"bins": request.bins} if request.chart_type == "histogram" else {}
    try:
        figure = build_chart(data, **render_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chart rendering failed")
        st.error(f"Could not render the chart: {exc}")
        return

    render_chart(figure)

    for note in data.notes:
        st.caption(note)

    with st.expander(f"Chart data ({data.point_count:,} rows)"):
        if data.is_empty:
            st.info("No rows to show.")
        else:
            st.dataframe(data.dataframe, width="stretch")
            st.download_button(
                "Download as CSV",
                data.dataframe.to_csv(index=False).encode("utf-8"),
                file_name=f"{request.chart_type}_data.csv",
                mime="text/csv",
                key="viz_download",
            )

    with st.expander("Chart specification"):
        st.caption(
            "This is the deterministic request behind the chart. In a later "
            "phase an agent will emit exactly this structure."
        )
        st.json(
            {
                "chart_type": data.chart_type,
                "x": data.x,
                "y": data.y,
                "color": data.color,
                "aggregation": data.aggregation,
                "frequency": data.frequency,
                "x_kind": data.x_kind,
                "title": data.title,
                "point_count": data.point_count,
            }
        )


def render_visualize(settings: Settings | None = None) -> None:
    """Render the Visualize page."""
    settings = settings or get_settings()

    st.title("Visualize")
    st.caption("Build a chart by choosing a type and the columns to put on it.")

    if not has_dataset():
        render_no_dataset_notice("Visualize")
        return

    df = get_dataframe()
    result = get_load_result()
    st.caption(f"**{result.source_name}** · {result.rows:,} rows × {result.columns} columns")

    labels = {spec.label: name for name, spec in CHART_SPECS.items()}
    chosen_label = st.radio(
        "Chart type",
        list(labels),
        horizontal=True,
        key="viz_chart_type",
    )
    spec = resolve_chart_type(labels[chosen_label])
    st.caption(spec.description)

    with st.container(border=True):
        request = _render_controls(df, spec)

    _render_chart(df, request)
