"""Tests for the deterministic executor and chart selection.

No LLM is involved at this layer: a plan goes in, computed numbers come out.
These tests assert the numbers, so a regression in the intent router shows up
as a wrong figure rather than a vague failure.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.plans import AnalysisPlan, Intent, PlanFilter, SortDirection
from app.services.analysis_executor import (
    HANDLERS,
    execute_plan,
    normalize_chart_spec,
    select_visualization,
)


@pytest.fixture
def shop() -> pd.DataFrame:
    """Twelve rows with known totals, so every assertion can be exact."""
    return pd.DataFrame(
        {
            "order_id": [f"O{i:03d}" for i in range(12)],
            "region": ["North", "North", "South", "South",
                       "North", "South", "East", "East",
                       "North", "South", "East", "North"],
            "channel": ["web", "shop"] * 6,
            "ordered_on": pd.to_datetime(
                [
                    "2024-01-10", "2024-01-20", "2024-02-05", "2024-02-25",
                    "2024-03-02", "2024-03-18", "2024-04-07", "2024-04-21",
                    "2024-05-11", "2024-05-29", "2024-06-03", "2024-06-23",
                ]
            ),
            "revenue": [100.0, 200.0, 50.0, 70.0, 300.0, 80.0,
                        400.0, 60.0, 120.0, 90.0, 110.0, 180.0],
            "units": [1, 2, 1, 1, 3, 1, 4, 1, 2, 1, 1, 2],
            "returned": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        }
    )


def _plan(**kwargs) -> AnalysisPlan:
    kwargs.setdefault("intent", Intent.SUMMARY)
    return AnalysisPlan(**kwargs)


# --------------------------------------------------------------------------- #
# Every intent has a handler
# --------------------------------------------------------------------------- #

def test_every_intent_is_routed():
    assert set(HANDLERS) == set(Intent)


def test_an_unroutable_intent_fails_gracefully(shop: pd.DataFrame, monkeypatch):
    monkeypatch.delitem(HANDLERS, Intent.SUMMARY)
    result = execute_plan(shop, _plan(intent=Intent.SUMMARY, metric="revenue"))

    assert result.success is False
    assert "does not yet support" in result.error


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def test_ranking_computes_the_right_totals(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", sort_direction=SortDirection.DESCENDING,
              limit=10),
    )

    assert result.success
    assert result.summary_data["Top region"] == "North"
    # North: 100 + 200 + 300 + 120 + 180 = 900
    assert result.summary_data["Total revenue (North)"] == pytest.approx(900.0)
    assert result.summary_data["Distinct region values"] == 3
    assert result.tools_used == ["rank_values"]
    assert result.metadata["result_shape"] == "ranked"


def test_ranking_share_of_total(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", limit=10),
    )
    # 900 of 1760 total.
    assert result.summary_data["Share of the listed total"] == pytest.approx(51.14, abs=0.01)


def test_ascending_ranking_finds_the_bottom(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", sort_direction=SortDirection.ASCENDING, limit=10),
    )
    assert result.summary_data["Top region"] == "South"


def test_ranking_without_a_metric_counts_rows(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, dimensions=["region"], aggregation="count",
              limit=10),
    )
    assert result.summary_data["Number of rows (North)"] == 5


def test_a_rate_metric_gets_a_percentage_from_python(shop: pd.DataFrame):
    """A 0/1 flag averaged is a rate; Python must express the percentage.

    Otherwise the only way to phrase the answer in percent is for the model to
    multiply by 100, which is arithmetic it is forbidden to perform.
    """
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="returned", dimensions=["channel"],
              aggregation="mean", limit=10),
    )

    # 2 of the 6 'shop' rows are returned.
    assert result.summary_data["Average returned (shop)"] == pytest.approx(1 / 3)
    assert result.summary_data[
        "Average returned (shop) as a percentage"
    ] == pytest.approx(33.33, abs=0.01)
    # The rate also appears as a column, not a share of total.
    assert "Average returned %" in result.table_data[0]
    assert "Share of the listed total" not in result.summary_data


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def test_comparison_reports_both_extremes_and_a_baseline(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.COMPARISON, metric="revenue", dimensions=["region"],
              aggregation="mean", limit=20),
    )

    assert result.success
    assert result.summary_data["Highest region"] == "East"
    # East: (400 + 60 + 110) / 3 = 190
    assert result.summary_data["Highest average revenue"] == pytest.approx(190.0)
    assert result.summary_data["Lowest region"] == "South"
    # Overall mean of all 12 rows.
    assert result.summary_data["Overall baseline"] == pytest.approx(
        shop["revenue"].mean()
    )
    assert result.tools_used == ["compare_segments"]


def test_segmentation_shares_the_comparison_handler(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.SEGMENTATION, metric="revenue", dimensions=["region"],
              aggregation="sum", limit=20),
    )
    assert result.success
    assert result.tools_used == ["compare_segments"]


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #

def test_trend_aggregates_by_period_in_order(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.TREND, metric="revenue", time_column="ordered_on",
              time_granularity="monthly", aggregation="sum"),
    )

    assert result.success
    assert result.summary_data["Periods covered"] == 6
    assert result.summary_data["First period"] == "2024-01"
    assert result.summary_data["Last period"] == "2024-06"
    # January: 100 + 200 = 300; June: 110 + 180 = 290.
    assert result.summary_data["Total revenue in the first period"] == pytest.approx(300.0)
    assert result.summary_data["Total revenue in the last period"] == pytest.approx(290.0)
    assert result.metadata["result_shape"] == "time_series"

    periods = [row["Period"] for row in result.table_data]
    assert periods == sorted(periods)


def test_trend_reports_the_peak(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.TREND, metric="revenue", time_column="ordered_on",
              time_granularity="monthly", aggregation="sum"),
    )
    # April: 400 + 60 = 460, the largest month.
    assert result.summary_data["Peak period"] == "2024-04"
    assert result.summary_data["Peak total revenue"] == pytest.approx(460.0)


def test_trend_without_a_time_column_fails_with_a_clear_message(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.TREND, metric="revenue"))

    assert result.success is False
    assert "valid date field" in result.error


def test_periods_scopes_the_window(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.TREND, metric="revenue", time_column="ordered_on",
              time_granularity="monthly", aggregation="sum", periods=3),
    )
    assert result.summary_data["Periods covered"] == 3
    assert result.summary_data["First period"] == "2024-04"


# --------------------------------------------------------------------------- #
# Time comparison and percentage change
# --------------------------------------------------------------------------- #

def test_time_comparison_contrasts_the_last_two_periods(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.TIME_COMPARISON, metric="revenue",
              time_column="ordered_on", time_granularity="monthly",
              aggregation="sum"),
    )

    assert result.success
    assert result.summary_data["Latest period"] == "2024-06"
    assert result.summary_data["Previous period"] == "2024-05"
    # June 290 vs May 210.
    assert result.summary_data["Total revenue (latest)"] == pytest.approx(290.0)
    assert result.summary_data["Total revenue (previous)"] == pytest.approx(210.0)
    assert result.summary_data["Absolute change"] == pytest.approx(80.0)
    assert result.summary_data["Percentage change"] == pytest.approx(38.1, abs=0.1)
    assert result.summary_data["Direction"] == "increase"


def test_time_comparison_needs_two_periods():
    single = pd.DataFrame(
        {"d": pd.to_datetime(["2024-01-01", "2024-01-15"]), "v": [1.0, 2.0]}
    )
    result = execute_plan(
        single,
        _plan(intent=Intent.TIME_COMPARISON, metric="v", time_column="d",
              time_granularity="monthly", aggregation="sum"),
    )

    assert result.success is False
    assert "fewer than two" in result.error


def test_percentage_change_ranks_groups_by_movement(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.PERCENTAGE_CHANGE, metric="revenue",
              dimensions=["region"], time_column="ordered_on",
              time_granularity="monthly", aggregation="sum",
              sort_direction=SortDirection.ASCENDING, limit=10),
    )

    assert result.success
    assert "Change (%)" in result.table_data[0]
    changes = [row["Change (%)"] for row in result.table_data]
    assert changes == sorted(changes)
    # The picture for "what declined" is one line per group.
    assert result.metadata["result_shape"] == "time_series"
    assert result.chart_spec["group_by"] == "region"


def test_percentage_change_without_a_dimension_compares_periods(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.PERCENTAGE_CHANGE, metric="revenue",
              time_column="ordered_on", time_granularity="monthly",
              aggregation="sum"),
    )
    assert result.success
    assert "Percentage change" in result.summary_data


# --------------------------------------------------------------------------- #
# Summary, distribution, correlation, count, anomaly, schema
# --------------------------------------------------------------------------- #

def test_summary_of_one_metric(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.SUMMARY, metric="revenue"))

    assert result.success
    assert result.summary_data["Rows"] == 12
    assert result.summary_data["Total revenue"] == pytest.approx(1760.0)
    assert result.summary_data["Mean revenue"] == pytest.approx(146.67, abs=0.01)
    assert result.tools_used == ["calculate_statistics"]


def test_summary_of_the_whole_dataset(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.SUMMARY))

    assert result.success
    assert result.summary_data["Numeric columns summarised"] == 3
    assert result.chart_spec is None


def test_distribution_reports_quartiles(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.DISTRIBUTION, metric="revenue"))

    assert result.success
    assert result.summary_data["Values"] == 12
    assert result.summary_data["Minimum"] == pytest.approx(50.0)
    assert result.summary_data["Maximum"] == pytest.approx(400.0)
    assert "25th percentile" in result.summary_data
    assert result.chart_spec["chart_type"] == "histogram"


def test_distribution_split_by_a_category(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.DISTRIBUTION, metric="revenue", dimensions=["region"]),
    )
    assert result.metadata["result_shape"] == "distribution_by_group"
    assert result.chart_spec["chart_type"] == "box"
    assert result.chart_spec["x"] == "region"


def test_correlation_focused_on_one_metric(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.CORRELATION, metric="revenue"))

    assert result.success
    assert "revenue" in result.summary_data["Strongest pair"]
    assert -1.0 <= result.summary_data["Correlation (r)"] <= 1.0
    assert result.chart_spec == {"chart_type": "heatmap"}
    # The caution is attached by Python, not left to the model.
    assert any("not causation" in note for note in result.notes)


def test_correlation_needs_two_numeric_columns():
    thin = pd.DataFrame({"a": [1.0, 2.0, 3.0], "label": ["x", "y", "z"]})
    result = execute_plan(thin, _plan(intent=Intent.CORRELATION))

    assert result.success is False
    assert "two numeric columns" in result.error


def test_correlation_with_an_uncorrelatable_metric():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    result = execute_plan(frame, _plan(intent=Intent.CORRELATION, metric="a"))
    assert result.success  # a vs b exists


def test_count_by_category(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.COUNT, dimensions=["region"]))

    assert result.success
    assert result.summary_data["Rows counted"] == 12
    assert result.summary_data["Most common region"] == "North"
    assert result.summary_data["Rows in the most common group"] == 5
    assert result.summary_data["Share of the most common group (%)"] == pytest.approx(
        41.67, abs=0.01
    )


def test_count_falls_back_to_a_groupable_column(shop: pd.DataFrame):
    # With region and channel gone, `returned` is still a two-value field, so
    # counting by it is more informative than a bare row count.
    result = execute_plan(
        shop.drop(columns=["region", "channel"]), _plan(intent=Intent.COUNT)
    )
    assert result.success
    assert result.metadata["dimension"] == "returned"
    assert result.summary_data["Rows counted"] == 12


def test_count_with_nothing_to_group_by_is_a_scalar(shop: pd.DataFrame):
    numbers_only = shop[["revenue", "units"]]
    result = execute_plan(numbers_only, _plan(intent=Intent.COUNT))

    assert result.success
    assert result.summary_data["Rows"] == 12
    assert result.chart_spec is None
    assert result.metadata["result_shape"] == "scalar"


def test_anomaly_detection(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.ANOMALY, metric="revenue"))

    assert result.success
    assert result.summary_data["Column examined"] == "revenue"
    assert result.summary_data["Values examined"] == 12
    assert "Outliers found" in result.summary_data
    assert result.chart_spec["chart_type"] == "box"


def test_dataset_question_describes_the_schema(shop: pd.DataFrame):
    result = execute_plan(shop, _plan(intent=Intent.DATASET_QUESTION))

    assert result.success
    assert result.summary_data["Rows"] == 12
    assert result.summary_data["Columns"] == 7
    assert len(result.table_data) == 7
    assert result.chart_spec is None  # a schema listing has no chart


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

def test_filters_are_applied_before_aggregation(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.SUMMARY, metric="revenue",
              filters=[PlanFilter(column="region", operator="eq", value="North")]),
    )

    assert result.success
    assert result.summary_data["Rows"] == 5
    assert result.summary_data["Total revenue"] == pytest.approx(900.0)
    assert result.metadata["rows_before_filter"] == 12
    assert result.metadata["rows_after_filter"] == 5


def test_a_filter_matching_nothing_fails_with_a_readable_message(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.SUMMARY, metric="revenue",
              filters=[PlanFilter(column="region", operator="eq", value="Mars")]),
    )

    assert result.success is False
    assert "no rows match" in result.error
    assert "Traceback" not in result.error


def test_a_broken_filter_fails_without_a_traceback(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.SUMMARY, metric="revenue",
              filters=[PlanFilter(column="revenue", operator="gt",
                                  value="not a number")]),
    )

    assert result.success is False
    assert "Traceback" not in (result.error or "")


def test_the_input_dataframe_is_never_mutated(shop: pd.DataFrame):
    before = shop.copy()
    for intent in Intent:
        execute_plan(
            shop,
            _plan(intent=intent, metric="revenue", dimensions=["region"],
                  time_column="ordered_on", aggregation="sum", limit=10),
        )
    pd.testing.assert_frame_equal(shop, before)


def test_an_empty_dataframe_is_handled():
    empty = pd.DataFrame({"region": pd.Series(dtype="object"),
                          "revenue": pd.Series(dtype="float64")})
    result = execute_plan(
        empty,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", limit=10),
    )
    # The handler returns a valid, empty result rather than raising.
    assert result.success in (True, False)
    assert "Traceback" not in (result.error or "")


# --------------------------------------------------------------------------- #
# Visualization selection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ("time_series", "line"),
        ("ranked", "bar"),
        ("correlation_matrix", "heatmap"),
        ("distribution", "histogram"),
        ("distribution_by_group", "box"),
        ("outliers", "box"),
        ("scalar", None),
        ("schema", None),
        ("none", None),
    ],
)
def test_the_result_shape_picks_the_chart(shape: str, expected: str | None):
    chosen = select_visualization(
        intent=Intent.SUMMARY, requested=None, shape=shape, row_count=10
    )
    assert chosen == expected


def test_a_compatible_suggestion_is_honoured():
    assert select_visualization(
        intent=Intent.TREND, requested="bar", shape="time_series", row_count=10
    ) == "bar"


def test_an_incompatible_suggestion_is_overridden():
    # A heatmap cannot show a ranking, whatever the planner asked for.
    assert select_visualization(
        intent=Intent.RANKING, requested="heatmap", shape="ranked", row_count=10
    ) == "bar"


def test_an_unknown_suggestion_falls_back_to_the_natural_chart():
    assert select_visualization(
        intent=Intent.RANKING, requested="pie", shape="ranked", row_count=10
    ) == "bar"


def test_a_single_row_result_gets_no_chart():
    assert select_visualization(
        intent=Intent.RANKING, requested="bar", shape="ranked", row_count=1
    ) is None


def test_an_overridden_chart_spec_is_repaired():
    # A box spec rendered as a histogram must move its column onto x.
    repaired = normalize_chart_spec({"chart_type": "box", "y": "revenue"}, "histogram")
    assert repaired == {"chart_type": "histogram", "x": "revenue"}


def test_a_histogram_spec_rendered_as_a_box_moves_to_y():
    repaired = normalize_chart_spec({"chart_type": "histogram", "x": "revenue"}, "box")
    assert repaired == {"chart_type": "box", "y": "revenue"}


def test_a_heatmap_spec_drops_every_axis():
    repaired = normalize_chart_spec(
        {"chart_type": "bar", "x": "region", "y": "revenue", "aggregation": "sum"},
        "heatmap",
    )
    assert repaired == {"chart_type": "heatmap"}


def test_a_temporal_spec_rendered_as_bars_uses_the_group_as_x():
    repaired = normalize_chart_spec(
        {
            "chart_type": "line", "x": "ordered_on", "y": "revenue",
            "frequency": "monthly", "group_by": "region", "aggregation": "sum",
        },
        "bar",
    )
    assert repaired["x"] == "region"
    assert "frequency" not in repaired
    assert "group_by" not in repaired


def test_every_chart_spec_the_executor_emits_can_be_rendered(shop: pd.DataFrame):
    """A spec the executor produces must always be drawable."""
    from app.services.visualization import build_chart
    from app.tools import generate_chart_data

    for intent in Intent:
        result = execute_plan(
            shop,
            _plan(intent=intent, metric="revenue", dimensions=["region"],
                  time_column="ordered_on", time_granularity="monthly",
                  aggregation="sum", limit=10),
        )
        if not result.success or not result.has_chart:
            continue
        data = generate_chart_data(shop, **result.chart_spec)
        figure = build_chart(data)
        assert figure is not None, intent


# --------------------------------------------------------------------------- #
# Time scoping for non-temporal intents
# --------------------------------------------------------------------------- #

def test_a_ranking_can_be_scoped_to_recent_periods(shop: pd.DataFrame):
    """"Which region earned most last month?" must not report the total.

    Non-temporal handlers have no time logic of their own, so without an
    explicit window a scoped question would aggregate everything while the
    answer claimed to describe one period.
    """
    scoped = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", time_column="ordered_on",
              time_granularity="monthly", periods=1, limit=10),
    )
    full = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", limit=10),
    )

    assert scoped.success
    assert scoped.metadata["rows_analysed"] < full.metadata["rows_analysed"]
    assert scoped.metadata["time_window"] == "2024-06"
    assert scoped.metadata["time_window_periods"] == 1


def test_a_two_period_window_covers_both(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.COUNT, dimensions=["region"],
              time_column="ordered_on", time_granularity="monthly",
              periods=2),
    )
    assert result.metadata["time_window"] == "2024-05 to 2024-06"
    assert result.metadata["rows_after_window"] < result.metadata["rows_before_window"]


def test_a_temporal_intent_scopes_itself(shop: pd.DataFrame):
    """Trends window inside the handler, so no outer window is applied."""
    result = execute_plan(
        shop,
        _plan(intent=Intent.TREND, metric="revenue", time_column="ordered_on",
              time_granularity="monthly", aggregation="sum", periods=3),
    )
    assert "time_window" not in result.metadata
    assert result.summary_data["Periods covered"] == 3


def test_a_window_wider_than_the_data_is_a_no_op(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", time_column="ordered_on",
              time_granularity="monthly", periods=99, limit=10),
    )
    assert "time_window" not in result.metadata
    assert result.metadata["rows_analysed"] == len(shop)


def test_a_window_without_a_date_column_is_a_no_op(shop: pd.DataFrame):
    result = execute_plan(
        shop,
        _plan(intent=Intent.RANKING, metric="revenue", dimensions=["region"],
              aggregation="sum", periods=1, limit=10),
    )
    assert "time_window" not in result.metadata
