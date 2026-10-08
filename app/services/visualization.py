"""The Plotly rendering engine.

One place that knows how Aurevia's charts look. Every figure goes through
:func:`apply_theme`, so fonts, colours, margins, hover behaviour and number
formatting are identical across the app, and :func:`build_chart` dispatches on
an explicit chart type -- there is no "pick something sensible" fallback.

Takes a :class:`~app.tools.charts.ChartData` (already validated, aggregated and
ordered by the tools layer) and returns a ``plotly.graph_objects.Figure``.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.tools.charts import ChartData, generate_chart_data, resolve_chart_type
from app.tools.exceptions import InvalidParameterError

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

#: Categorical palette: distinguishable, and readable on light and dark grounds.
PALETTE: tuple[str, ...] = (
    "#2E6BE6",  # blue
    "#E8743B",  # orange
    "#19A979",  # green
    "#945ECF",  # purple
    "#C7303C",  # red
    "#13A4B4",  # teal
    "#D4A62A",  # gold
    "#6F7D8C",  # slate
    "#E36BAE",  # pink
    "#8B6B4A",  # brown
)

#: Diverging scale for correlations: negative -> neutral -> positive.
DIVERGING_SCALE = (
    (0.0, "#C7303C"),
    (0.5, "#F2F2F2"),
    (1.0, "#2E6BE6"),
)

#: Sequential scale for magnitude-only heatmaps.
SEQUENTIAL_SCALE = ((0.0, "#EAF0FB"), (1.0, "#1B4FB0"))

FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", '
    "Arial, sans-serif"
)

DEFAULT_HEIGHT = 420
GRID_COLOR = "rgba(128, 128, 128, 0.18)"
AXIS_LINE_COLOR = "rgba(128, 128, 128, 0.45)"

#: Streamlit chrome we hide from the Plotly mode bar.
MODEBAR_REMOVE = ("lasso2d", "select2d", "autoScale2d", "toggleSpikelines")

#: ``st.plotly_chart`` config -- one definition, used by every page.
PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": list(MODEBAR_REMOVE),
    "scrollZoom": False,
    "responsive": True,
}

#: Tick format that keeps large numbers readable (1.2k, 3.4M).
COMPACT_TICK_FORMAT = "~s"


def apply_theme(
    fig: go.Figure,
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    height: int = DEFAULT_HEIGHT,
    x_kind: str = "category",
    show_legend: bool | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply Aurevia's consistent layout to any figure."""
    fig.update_layout(
        template="plotly_white",
        colorway=list(PALETTE),
        title={
            "text": title,
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 17},
        },
        font={"family": FONT_FAMILY, "size": 13},
        height=height,
        margin={"l": 70, "r": 30, "t": 60 if title else 30, "b": 60},
        hovermode="x unified" if x_kind == "time" else "closest",
        hoverlabel={"font": {"family": FONT_FAMILY, "size": 12}, "namelength": -1},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=show_legend if show_legend is not None else None,
        legend={
            "title": {"text": legend_title or ""},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )

    fig.update_xaxes(
        title_text=x_label,
        showgrid=x_kind == "numeric",
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=AXIS_LINE_COLOR,
        showline=True,
        ticks="outside",
        tickcolor=AXIS_LINE_COLOR,
        automargin=True,
    )
    fig.update_yaxes(
        title_text=y_label,
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=AXIS_LINE_COLOR,
        showline=False,
        automargin=True,
    )

    if x_kind == "time":
        fig.update_xaxes(showspikes=False)
    return fig


def _compact_y_axis(fig: go.Figure, values: pd.Series | Sequence[Any]) -> go.Figure:
    """Switch to SI tick labels once the numbers get long."""
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if not numeric.empty and numeric.abs().max() >= 10_000:
        fig.update_yaxes(tickformat=COMPACT_TICK_FORMAT)
    return fig


def _rotate_crowded_ticks(fig: go.Figure, category_count: int) -> go.Figure:
    if category_count > 6:
        fig.update_xaxes(tickangle=-35)
    return fig


def empty_figure(message: str = "No data to display") -> go.Figure:
    """A themed placeholder, so an empty result still renders a chart frame."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"family": FONT_FAMILY, "size": 14, "color": "#6F7D8C"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, height=260)


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #

def _hover_data(data: ChartData) -> dict[str, Any]:
    """Only include hover columns that actually survived preparation."""
    return {
        column: True
        for column in data.hover_columns
        if column in data.dataframe.columns
    }


def line_chart(data: ChartData) -> go.Figure:
    """A chronologically (or numerically) ordered line chart."""
    frame = data.dataframe
    if frame.empty:
        return empty_figure("No points to plot")

    fig = px.line(
        frame,
        x=data.x,
        y=data.y,
        color=data.color,
        markers=len(frame) <= 60,
        hover_data=_hover_data(data),
        category_orders=({data.x: data.category_order} if data.category_order else None),
    )
    fig.update_traces(line={"width": 2.2}, marker={"size": 6})
    apply_theme(
        fig,
        title=data.title,
        x_label=data.x_label,
        y_label=data.y_label,
        x_kind=data.x_kind,
        legend_title=data.color,
    )
    if data.x_kind == "time":
        fig.update_xaxes(showgrid=False, tickformat="%b %Y" if _is_coarse(data) else None)
    return _compact_y_axis(fig, frame[data.y])


def _is_coarse(data: ChartData) -> bool:
    return data.frequency in ("monthly", "quarterly", "yearly")


def bar_chart(data: ChartData) -> go.Figure:
    """A bar chart with explicit, measure-based category ordering."""
    frame = data.dataframe
    if frame.empty:
        return empty_figure("No categories to plot")

    fig = px.bar(
        frame,
        x=data.x,
        y=data.y,
        color=data.color,
        barmode="group" if data.color else "relative",
        hover_data=_hover_data(data),
        category_orders=({data.x: data.category_order} if data.category_order else None),
    )
    fig.update_traces(marker={"line": {"width": 0}})
    apply_theme(
        fig,
        title=data.title,
        x_label=data.x_label,
        y_label=data.y_label,
        x_kind="category",
        legend_title=data.color,
    )
    _rotate_crowded_ticks(fig, frame[data.x].nunique() if data.x in frame else 0)
    return _compact_y_axis(fig, frame[data.y])


def scatter_plot(data: ChartData) -> go.Figure:
    """A scatter plot with opacity scaled to the point count."""
    frame = data.dataframe
    if frame.empty:
        return empty_figure("No points to plot")

    fig = px.scatter(
        frame,
        x=data.x,
        y=data.y,
        color=data.color,
        opacity=0.75 if len(frame) < 500 else 0.45,
        hover_data=_hover_data(data),
    )
    fig.update_traces(marker={"size": 7 if len(frame) < 500 else 5})
    apply_theme(
        fig,
        title=data.title,
        x_label=data.x_label,
        y_label=data.y_label,
        x_kind="numeric",
        legend_title=data.color,
    )
    fig.update_xaxes(tickformat=COMPACT_TICK_FORMAT if _is_large(frame[data.x]) else None)
    return _compact_y_axis(fig, frame[data.y])


def _is_large(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return bool(not numeric.empty and numeric.abs().max() >= 10_000)


def histogram(data: ChartData, *, bins: int = 30) -> go.Figure:
    """A distribution histogram."""
    frame = data.dataframe
    if frame.empty:
        return empty_figure("No values to plot")

    fig = px.histogram(
        frame,
        x=data.x,
        color=data.color,
        nbins=bins,
        barmode="overlay" if data.color else "relative",
        opacity=0.72 if data.color else 1.0,
    )
    apply_theme(
        fig,
        title=data.title,
        x_label=data.x_label,
        y_label=data.y_label or "Number of rows",
        x_kind="numeric",
        legend_title=data.color,
    )
    fig.update_xaxes(
        tickformat=COMPACT_TICK_FORMAT if _is_large(frame[data.x]) else None
    )
    return fig


def box_plot(data: ChartData) -> go.Figure:
    """A box plot, optionally split by a category ordered by median."""
    frame = data.dataframe
    if frame.empty:
        return empty_figure("No values to plot")

    fig = px.box(
        frame,
        x=data.x,
        y=data.y,
        color=data.x if data.x else None,
        points="outliers",
        category_orders=({data.x: data.category_order} if data.category_order else None),
    )
    apply_theme(
        fig,
        title=data.title,
        x_label=data.x_label,
        y_label=data.y_label,
        x_kind="category",
        # The colour axis repeats the x axis, so its legend adds nothing.
        show_legend=False,
    )
    if data.x and data.x in frame:
        _rotate_crowded_ticks(fig, frame[data.x].nunique())
    return _compact_y_axis(fig, frame[data.y])


def correlation_heatmap(data: ChartData) -> go.Figure:
    """A correlation matrix with a fixed -1..1 diverging scale."""
    matrix = data.dataframe
    if matrix.empty or matrix.shape[0] < 2:
        return empty_figure("At least two numeric columns are needed")

    labels = [str(c) for c in matrix.columns]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=labels,
            y=labels,
            # A fixed domain means colour always means the same thing.
            zmin=-1,
            zmax=1,
            colorscale=list(DIVERGING_SCALE),
            colorbar={"title": "r", "thickness": 14},
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
            text=matrix.round(2).to_numpy(),
            texttemplate="%{text}" if len(labels) <= 12 else None,
            textfont={"size": 11},
        )
    )
    apply_theme(
        fig,
        title=data.title,
        x_kind="category",
        height=max(DEFAULT_HEIGHT, 90 + 34 * len(labels)),
        show_legend=False,
    )
    fig.update_xaxes(tickangle=-35, showline=False, ticks="")
    # Keep the matrix diagonal running top-left to bottom-right.
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    return fig


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

#: Explicit chart-type -> renderer map. No defaults, no guessing.
RENDERERS = {
    "line": line_chart,
    "bar": bar_chart,
    "scatter": scatter_plot,
    "histogram": histogram,
    "box": box_plot,
    "heatmap": correlation_heatmap,
}


def build_chart(data: ChartData, **kwargs: Any) -> go.Figure:
    """Render a prepared :class:`ChartData` with its matching renderer."""
    if not isinstance(data, ChartData):
        raise InvalidParameterError(
            f"build_chart() needs a ChartData, got {type(data).__name__}."
        )
    spec = resolve_chart_type(data.chart_type)
    renderer = RENDERERS[spec.name]
    if data.is_empty:
        return empty_figure(data.notes[0] if data.notes else "No data to display")
    return renderer(data, **kwargs)


def chart_from_dataframe(
    df: pd.DataFrame,
    chart_type: str,
    **options: Any,
) -> tuple[go.Figure, ChartData]:
    """Prepare and render in one call; returns the figure and its data."""
    render_options = {"bins": options["bins"]} if (
        chart_type in ("histogram", "hist") and "bins" in options
    ) else {}
    data = generate_chart_data(df, chart_type, **options)
    return build_chart(data, **render_options), data


__all__ = [
    "DEFAULT_HEIGHT",
    "DIVERGING_SCALE",
    "PALETTE",
    "PLOTLY_CONFIG",
    "RENDERERS",
    "SEQUENTIAL_SCALE",
    "apply_theme",
    "bar_chart",
    "box_plot",
    "build_chart",
    "chart_from_dataframe",
    "correlation_heatmap",
    "empty_figure",
    "histogram",
    "line_chart",
    "scatter_plot",
]
