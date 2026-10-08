"""Tests for chart-data preparation and the Plotly rendering engine."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.visualization import (
    PLOTLY_CONFIG,
    bar_chart,
    box_plot,
    build_chart,
    chart_from_dataframe,
    correlation_heatmap,
    empty_figure,
    histogram,
    line_chart,
    scatter_plot,
)
from app.tools.charts import (
    CHART_SPECS,
    ChartData,
    generate_chart_data,
    humanize,
    lower_label,
    resolve_chart_type,
    suggest_chart_options,
    supported_chart_types,
)
from app.tools.exceptions import (
    ColumnNotFoundError,
    InvalidColumnTypeError,
    InvalidDataError,
    InvalidParameterError,
    UnsupportedOperationError,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "East", "South", "North"],
            "channel": ["web", "web", "shop", "shop", "web", "shop"],
            "signup": pd.to_datetime(
                [
                    "2023-01-10", "2023-01-25", "2023-02-14",
                    "2023-03-02", "2023-03-19", "2023-04-07",
                ]
            ),
            "revenue": [100.0, 250.0, 50.0, 400.0, 175.0, 80.0],
            "data_usage_gb": [5.0, 12.0, 3.0, 20.0, 9.0, 4.0],
            "label": ["a", "b", "c", "d", "e", "f"],
        }
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_all_required_chart_types_are_supported():
    assert set(supported_chart_types()) == {
        "line", "bar", "scatter", "histogram", "box", "heatmap"
    }
    assert set(CHART_SPECS) == set(supported_chart_types())


def test_chart_type_aliases_resolve():
    assert resolve_chart_type("Bar Chart").name == "bar"
    assert resolve_chart_type("scatterplot").name == "scatter"
    assert resolve_chart_type("corr").name == "heatmap"
    assert resolve_chart_type("hist").name == "histogram"


def test_unknown_chart_type_raises(frame: pd.DataFrame):
    with pytest.raises(UnsupportedOperationError, match="Unsupported chart type"):
        generate_chart_data(frame, "pie", x="region")


def test_non_string_chart_type_raises(frame: pd.DataFrame):
    with pytest.raises(UnsupportedOperationError):
        generate_chart_data(frame, 7, x="region")


def test_suggest_chart_options_groups_columns(frame: pd.DataFrame):
    options = suggest_chart_options(frame)

    assert "revenue" in options["numeric"]
    assert "region" in options["categorical"]
    assert "signup" in options["datetime"]
    assert options["all"] == list(frame.columns)
    assert "sum" in options["aggregations"]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def test_humanize_formats_column_names():
    assert humanize("monthly_charge") == "Monthly charge"
    assert humanize("data_usage_gb") == "Data usage (GB)"
    assert humanize("customer_id") == "Customer ID"
    assert humanize("ARPU") == "ARPU"


def test_lower_label_preserves_units_and_acronyms():
    assert lower_label("data_usage_gb") == "data usage (GB)"
    assert lower_label("monthly_charge") == "monthly charge"
    assert lower_label("ARPU") == "ARPU"


# --------------------------------------------------------------------------- #
# Bar
# --------------------------------------------------------------------------- #

def test_bar_aggregates_by_category(frame: pd.DataFrame):
    data = generate_chart_data(frame, "bar", x="region", y="revenue", aggregation="sum")

    assert data.chart_type == "bar"
    assert data.x == "region" and data.y == "value"
    assert data.x_kind == "category"
    assert data.aggregation == "sum"
    assert data.point_count == 3

    lookup = dict(zip(data.dataframe["region"], data.dataframe["value"]))
    assert lookup["North"] == pytest.approx(230.0)


def test_bar_categories_are_ordered_by_the_measure(frame: pd.DataFrame):
    data = generate_chart_data(frame, "bar", x="region", y="revenue", aggregation="sum")

    # Sums are South=425, East=400, North=230 -- descending, not alphabetical.
    assert list(data.dataframe["region"]) == ["South", "East", "North"]
    assert data.category_order == ["South", "East", "North"]


def test_bar_labels_name_the_aggregation(frame: pd.DataFrame):
    data = generate_chart_data(frame, "bar", x="region", y="revenue", aggregation="mean")

    assert data.title == "Average revenue by region"
    assert data.x_label == "Region"
    assert data.y_label == "Average revenue"


def test_bar_without_a_measure_counts_rows(frame: pd.DataFrame):
    data = generate_chart_data(frame, "bar", x="region")

    assert data.aggregation == "count"
    assert data.y_label == "Number of rows"
    assert dict(zip(data.dataframe["region"], data.dataframe["value"]))["North"] == 3


def test_bar_top_n_limits_and_notes(frame: pd.DataFrame):
    data = generate_chart_data(frame, "bar", x="region", y="revenue", top_n=2)

    assert data.point_count == 2
    assert any("top 2 of 3" in note for note in data.notes)


def test_bar_grouping_adds_a_colour_series(frame: pd.DataFrame):
    data = generate_chart_data(
        frame, "bar", x="region", y="revenue", group_by="channel", aggregation="sum"
    )

    assert data.color == "channel"
    assert "channel" in data.dataframe.columns
    assert data.point_count == 4


def test_bar_requires_an_x_column(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="needs an x-axis column"):
        generate_chart_data(frame, "bar", y="revenue")


def test_bar_with_a_text_measure_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        generate_chart_data(frame, "bar", x="region", y="label", aggregation="sum")


# --------------------------------------------------------------------------- #
# Line
# --------------------------------------------------------------------------- #

def test_line_on_a_date_column_is_a_time_axis(frame: pd.DataFrame):
    data = generate_chart_data(
        frame, "line", x="signup", y="revenue", aggregation="sum", frequency="monthly"
    )

    assert data.x_kind == "time"
    assert data.frequency == "monthly"
    assert data.x_label == "Month"
    assert data.point_count == 4


def test_line_is_chronologically_ordered(frame: pd.DataFrame):
    data = generate_chart_data(frame, "line", x="signup", y="revenue", frequency="monthly")
    assert data.dataframe["signup"].is_monotonic_increasing


def test_line_carries_hover_columns(frame: pd.DataFrame):
    data = generate_chart_data(frame, "line", x="signup", y="revenue")

    assert "row_count" in data.dataframe.columns
    assert "pct_change" in data.dataframe.columns
    assert set(data.hover_columns) == {"row_count", "pct_change"}


def test_line_grouped_by_a_category(frame: pd.DataFrame):
    data = generate_chart_data(
        frame, "line", x="signup", y="revenue", group_by="channel", aggregation="sum"
    )

    assert data.color == "channel"
    assert set(data.dataframe["channel"]) == {"web", "shop"}
    # Each group's series is internally ordered.
    for _, chunk in data.dataframe.groupby("channel"):
        assert chunk["signup"].is_monotonic_increasing


def test_line_on_a_numeric_axis(frame: pd.DataFrame):
    data = generate_chart_data(
        frame, "line", x="data_usage_gb", y="revenue", aggregation="mean"
    )

    assert data.x_kind == "numeric"
    assert data.dataframe["data_usage_gb"].is_monotonic_increasing


def test_line_on_a_categorical_axis(frame: pd.DataFrame):
    data = generate_chart_data(frame, "line", x="region", y="revenue")
    assert data.x_kind == "category"
    assert data.category_order is not None


def test_line_requires_an_x_column(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="needs an x-axis column"):
        generate_chart_data(frame, "line", y="revenue")


def test_line_excludes_rows_with_unusable_dates():
    frame = pd.DataFrame(
        {"d": ["2023-01-01", None, "2023-02-01"], "v": [1.0, 2.0, 3.0]}
    )
    data = generate_chart_data(frame, "line", x="d", y="v")
    assert any("no date" in note for note in data.notes)


# --------------------------------------------------------------------------- #
# Scatter
# --------------------------------------------------------------------------- #

def test_scatter_keeps_individual_rows(frame: pd.DataFrame):
    data = generate_chart_data(frame, "scatter", x="revenue", y="data_usage_gb")

    assert data.point_count == len(frame)
    assert data.x == "revenue" and data.y == "data_usage_gb"
    assert data.x_kind == "numeric"
    assert data.aggregation is None


def test_scatter_title_and_labels(frame: pd.DataFrame):
    data = generate_chart_data(frame, "scatter", x="revenue", y="data_usage_gb")

    assert data.title == "Data usage (GB) vs revenue"
    assert data.x_label == "Revenue"
    assert data.y_label == "Data usage (GB)"


def test_scatter_needs_both_axes(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="both an x and a y"):
        generate_chart_data(frame, "scatter", x="revenue")


def test_scatter_axes_must_be_numeric(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        generate_chart_data(frame, "scatter", x="region", y="revenue")


def test_scatter_excludes_incomplete_rows():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": [2.0, 4.0, None]})
    data = generate_chart_data(frame, "scatter", x="a", y="b")

    assert data.point_count == 1
    assert any("missing values" in note for note in data.notes)


def test_scatter_samples_large_inputs():
    big = pd.DataFrame({"a": range(6_000), "b": range(6_000)})
    data = generate_chart_data(big, "scatter", x="a", y="b")

    assert data.point_count == 5_000
    assert any("Sampled" in note for note in data.notes)


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #

def test_histogram_takes_one_numeric_column(frame: pd.DataFrame):
    data = generate_chart_data(frame, "histogram", x="revenue", bins=12)

    assert data.x == "revenue"
    assert data.y is None
    assert data.y_label == "Number of rows"
    assert data.point_count == len(frame)
    assert any("12 bins" in note for note in data.notes)


def test_histogram_accepts_the_column_as_y(frame: pd.DataFrame):
    data = generate_chart_data(frame, "histogram", y="revenue")
    assert data.x == "revenue"


def test_histogram_requires_a_column(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="needs one numeric column"):
        generate_chart_data(frame, "histogram")


def test_histogram_column_must_be_numeric(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        generate_chart_data(frame, "histogram", x="region")


def test_histogram_bins_must_be_a_positive_integer(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="positive integer"):
        generate_chart_data(frame, "histogram", x="revenue", bins=0)


# --------------------------------------------------------------------------- #
# Box
# --------------------------------------------------------------------------- #

def test_box_without_a_category(frame: pd.DataFrame):
    data = generate_chart_data(frame, "box", y="revenue")

    assert data.y == "revenue"
    assert data.x is None
    assert data.title == "Spread of revenue"


def test_box_split_by_a_category(frame: pd.DataFrame):
    data = generate_chart_data(frame, "box", x="region", y="revenue")

    assert data.x == "region"
    assert data.title == "Revenue by region"
    # Categories are ordered by descending median.
    assert data.category_order == ["East", "South", "North"]


def test_box_accepts_the_category_as_group_by(frame: pd.DataFrame):
    data = generate_chart_data(frame, "box", y="revenue", group_by="region")
    assert data.x == "region"


def test_box_requires_a_numeric_y(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="needs a numeric y column"):
        generate_chart_data(frame, "box", x="region")


def test_box_y_must_be_numeric(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        generate_chart_data(frame, "box", y="label")


# --------------------------------------------------------------------------- #
# Heatmap
# --------------------------------------------------------------------------- #

def test_heatmap_is_a_correlation_matrix(frame: pd.DataFrame):
    data = generate_chart_data(frame, "heatmap")

    assert data.chart_type == "heatmap"
    assert data.dataframe.shape == (2, 2)
    assert data.category_order == ["revenue", "data_usage_gb"]
    assert "pearson" in data.title


def test_heatmap_rejects_grouping(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="does not support grouping"):
        generate_chart_data(frame, "heatmap", group_by="region")


def test_heatmap_with_too_few_numeric_columns():
    data = generate_chart_data(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), "heatmap")

    assert data.is_empty
    assert any("two numeric columns" in note for note in data.notes)


# --------------------------------------------------------------------------- #
# Shared behaviour
# --------------------------------------------------------------------------- #

def test_a_custom_title_overrides_the_generated_one(frame: pd.DataFrame):
    data = generate_chart_data(
        frame, "bar", x="region", y="revenue", title="Revenue by territory"
    )
    assert data.title == "Revenue by territory"


def test_chart_data_never_mutates_the_input(frame: pd.DataFrame):
    before = frame.copy()
    for chart_type, kwargs in [
        ("bar", {"x": "region", "y": "revenue"}),
        ("line", {"x": "signup", "y": "revenue"}),
        ("scatter", {"x": "revenue", "y": "data_usage_gb"}),
        ("histogram", {"x": "revenue"}),
        ("box", {"x": "region", "y": "revenue"}),
        ("heatmap", {}),
    ]:
        generate_chart_data(frame, chart_type, **kwargs)
    pd.testing.assert_frame_equal(frame, before)


def test_missing_columns_raise_for_every_chart_type(frame: pd.DataFrame):
    for chart_type, kwargs in [
        ("bar", {"x": "nope", "y": "revenue"}),
        ("line", {"x": "nope", "y": "revenue"}),
        ("scatter", {"x": "nope", "y": "revenue"}),
        ("histogram", {"x": "nope"}),
        ("box", {"y": "nope"}),
    ]:
        with pytest.raises(ColumnNotFoundError):
            generate_chart_data(frame, chart_type, **kwargs)


def test_empty_dataframe_yields_empty_chart_data():
    empty = pd.DataFrame(
        {"region": pd.Series(dtype="object"), "revenue": pd.Series(dtype="float64")}
    )
    data = generate_chart_data(empty, "bar", x="region", y="revenue")

    assert data.is_empty
    assert data.point_count == 0
    assert data.notes


def test_non_dataframe_input_raises():
    with pytest.raises(InvalidDataError):
        generate_chart_data([1, 2, 3], "histogram", x="a")


def test_chart_data_is_serialisable(frame: pd.DataFrame):
    import json

    payload = generate_chart_data(frame, "bar", x="region", y="revenue").to_dict()
    json.dumps(payload, default=str)

    assert payload["chart_type"] == "bar"
    assert payload["point_count"] == 3
    assert len(payload["data"]) == 3


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("chart_type", "kwargs"),
    [
        ("line", {"x": "signup", "y": "revenue", "aggregation": "sum"}),
        ("bar", {"x": "region", "y": "revenue", "aggregation": "mean"}),
        ("scatter", {"x": "revenue", "y": "data_usage_gb"}),
        ("histogram", {"x": "revenue"}),
        ("box", {"x": "region", "y": "revenue"}),
        ("heatmap", {}),
    ],
)
def test_every_chart_type_renders(frame: pd.DataFrame, chart_type, kwargs):
    figure, data = chart_from_dataframe(frame, chart_type, **kwargs)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) >= 1
    assert figure.layout.title.text == data.title


def test_axis_titles_are_set(frame: pd.DataFrame):
    figure, data = chart_from_dataframe(
        frame, "bar", x="region", y="revenue", aggregation="sum"
    )
    assert figure.layout.xaxis.title.text == data.x_label
    assert figure.layout.yaxis.title.text == data.y_label


def test_grouped_charts_produce_one_trace_per_series(frame: pd.DataFrame):
    figure, _ = chart_from_dataframe(
        frame, "bar", x="region", y="revenue", group_by="channel", aggregation="sum"
    )
    assert len(figure.data) == 2


def test_time_axis_uses_unified_hover(frame: pd.DataFrame):
    figure, _ = chart_from_dataframe(frame, "line", x="signup", y="revenue")
    assert figure.layout.hovermode == "x unified"


def test_categorical_charts_use_closest_hover(frame: pd.DataFrame):
    figure, _ = chart_from_dataframe(frame, "bar", x="region", y="revenue")
    assert figure.layout.hovermode == "closest"


def test_bar_respects_the_prepared_category_order(frame: pd.DataFrame):
    figure, data = chart_from_dataframe(
        frame, "bar", x="region", y="revenue", aggregation="sum"
    )
    assert list(figure.data[0].x) == data.category_order


def test_heatmap_uses_a_fixed_colour_domain(frame: pd.DataFrame):
    figure, _ = chart_from_dataframe(frame, "heatmap")
    trace = figure.data[0]

    assert trace.zmin == -1 and trace.zmax == 1
    assert figure.layout.yaxis.autorange == "reversed"


def test_histogram_bin_count_reaches_plotly(frame: pd.DataFrame):
    figure, _ = chart_from_dataframe(frame, "histogram", x="revenue", bins=7)
    assert figure.data[0].nbinsx == 7


def test_empty_chart_data_renders_a_placeholder():
    empty = ChartData(chart_type="bar", notes=["nothing here"])
    figure = build_chart(empty)

    assert isinstance(figure, go.Figure)
    assert figure.layout.annotations[0].text == "nothing here"


def test_empty_figure_is_themed():
    figure = empty_figure("no data")
    assert figure.layout.annotations[0].text == "no data"
    assert figure.layout.xaxis.visible is False


def test_build_chart_rejects_non_chart_data():
    with pytest.raises(InvalidParameterError, match="needs a ChartData"):
        build_chart({"chart_type": "bar"})


def test_renderers_handle_an_empty_frame_directly():
    for renderer in (line_chart, bar_chart, scatter_plot, histogram, box_plot,
                     correlation_heatmap):
        figure = renderer(ChartData(chart_type="bar", x="x", y="y"))
        assert isinstance(figure, go.Figure)
        assert figure.layout.annotations  # the placeholder message


def test_plotly_config_hides_the_logo():
    assert PLOTLY_CONFIG["displaylogo"] is False
    assert "lasso2d" in PLOTLY_CONFIG["modeBarButtonsToRemove"]
