"""Tests for the aggregation registry, grouping, statistics and correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tools.aggregations import (
    AGGREGATIONS,
    Aggregation,
    add_percentage_change,
    register_aggregation,
    resolve_aggregation,
    supported_aggregations,
)
from app.tools.analysis import (
    calculate_correlation,
    calculate_statistics,
    group_and_aggregate,
)
from app.tools.exceptions import (
    ColumnNotFoundError,
    InvalidColumnTypeError,
    InvalidDataError,
    InvalidParameterError,
    UnsupportedOperationError,
)


@pytest.fixture
def sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "North", "South", "South", "South", "East"],
            "channel": ["web", "shop", "web", "shop", "web", "web"],
            "revenue": [100.0, 200.0, 50.0, 70.0, 80.0, 400.0],
            "units": [1, 3, 2, 2, 4, 8],
            "label": ["a", "b", "c", "d", "e", "f"],
        }
    )


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

def test_all_required_aggregations_are_registered():
    required = {"sum", "mean", "median", "count", "min", "max", "std", "percentage_change"}
    assert required <= set(AGGREGATIONS)


def test_aliases_resolve():
    assert resolve_aggregation("average").name == "mean"
    assert resolve_aggregation("TOTAL").name == "sum"
    assert resolve_aggregation("standard deviation").name == "std"
    assert resolve_aggregation("pct_change").name == "percentage_change"


def test_unknown_aggregation_raises():
    with pytest.raises(UnsupportedOperationError, match="Unsupported aggregation"):
        resolve_aggregation("geometric_mean")


def test_non_string_aggregation_raises():
    with pytest.raises(UnsupportedOperationError):
        resolve_aggregation(42)


def test_supported_aggregations_can_be_filtered():
    numeric = supported_aggregations(numeric_only=True)
    flexible = supported_aggregations(numeric_only=False)

    assert "sum" in numeric and "count" not in numeric
    assert "count" in flexible


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sum", 20.0),
        ("mean", 5.0),
        ("median", 5.0),
        ("count", 4),
        ("min", 2.0),
        ("max", 8.0),
    ],
)
def test_builtin_aggregation_values(name: str, expected: float):
    series = pd.Series([2.0, 4.0, 6.0, 8.0])
    assert resolve_aggregation(name).apply(series) == pytest.approx(expected)


def test_std_is_the_sample_standard_deviation():
    series = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert resolve_aggregation("std").apply(series) == pytest.approx(2.13809, abs=1e-4)


def test_std_of_a_single_value_is_zero_not_nan():
    assert resolve_aggregation("std").apply(pd.Series([5.0])) == 0.0


def test_count_ignores_missing_values():
    assert resolve_aggregation("count").apply(pd.Series([1.0, None, 3.0])) == 2


def test_aggregations_ignore_missing_values():
    series = pd.Series([10.0, None, 30.0])
    assert resolve_aggregation("mean").apply(series) == pytest.approx(20.0)
    assert resolve_aggregation("sum").apply(series) == pytest.approx(40.0)


def test_aggregation_of_an_all_missing_series_is_nan():
    series = pd.Series([None, None], dtype="float64")
    assert np.isnan(resolve_aggregation("mean").apply(series))


def test_percentage_change_is_first_to_last():
    assert resolve_aggregation("percentage_change").apply(
        pd.Series([100.0, 150.0, 200.0])
    ) == pytest.approx(100.0)


def test_percentage_change_handles_a_zero_start():
    assert np.isnan(
        resolve_aggregation("percentage_change").apply(pd.Series([0.0, 50.0]))
    )


def test_percentage_change_needs_two_values():
    assert np.isnan(resolve_aggregation("percentage_change").apply(pd.Series([5.0])))


def test_min_max_work_on_dates():
    dates = pd.Series(pd.to_datetime(["2023-05-01", "2023-01-01", "2023-09-01"]))
    assert resolve_aggregation("min").apply(dates) == pd.Timestamp("2023-01-01")
    assert resolve_aggregation("max").apply(dates) == pd.Timestamp("2023-09-01")


def test_registry_is_extensible():
    variance = Aggregation(
        "variance_test", "Variance", lambda s: float(s.var(ddof=1))
    )
    try:
        register_aggregation(variance)
        assert resolve_aggregation("variance_test").apply(
            pd.Series([1.0, 2.0, 3.0])
        ) == pytest.approx(1.0)
        assert "variance_test" in supported_aggregations()
    finally:
        AGGREGATIONS.pop("variance_test", None)


def test_registering_a_duplicate_name_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_aggregation(Aggregation("sum", "Sum", lambda s: 0))


def test_add_percentage_change_is_period_over_period():
    change = add_percentage_change(pd.Series([100.0, 150.0, 75.0]))

    assert np.isnan(change.iloc[0])
    assert change.iloc[1] == pytest.approx(50.0)
    assert change.iloc[2] == pytest.approx(-50.0)


def test_add_percentage_change_survives_a_zero_denominator():
    change = add_percentage_change(pd.Series([0.0, 10.0]))
    assert np.isnan(change.iloc[1])


# --------------------------------------------------------------------------- #
# group_and_aggregate
# --------------------------------------------------------------------------- #

def test_group_by_single_column(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"revenue": "sum"})

    assert list(result.columns) == ["region", "revenue_sum"]
    assert len(result) == 3
    lookup = dict(zip(result["region"], result["revenue_sum"]))
    assert lookup["North"] == pytest.approx(300.0)
    assert lookup["South"] == pytest.approx(200.0)


def test_results_are_sorted_by_the_first_measure_descending(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"revenue": "sum"})
    assert list(result["region"]) == ["East", "North", "South"]


def test_ascending_sort(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"revenue": "sum"}, ascending=True)
    assert list(result["region"]) == ["South", "North", "East"]


def test_group_by_multiple_columns(sales: pd.DataFrame):
    result = group_and_aggregate(sales, ["region", "channel"], {"revenue": "sum"})

    assert list(result.columns) == ["region", "channel", "revenue_sum"]
    assert len(result) == 5


def test_multiple_aggregations_on_one_column(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"revenue": ["sum", "mean"]})

    assert list(result.columns) == ["region", "revenue_sum", "revenue_mean"]
    north = result[result["region"] == "North"].iloc[0]
    assert north["revenue_mean"] == pytest.approx(150.0)


def test_aggregations_across_several_columns(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"revenue": "sum", "units": "mean"})
    assert list(result.columns) == ["region", "revenue_sum", "units_mean"]


def test_counting_rows(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", "count")

    assert list(result.columns) == ["region", "row_count"]
    assert dict(zip(result["region"], result["row_count"]))["South"] == 3


def test_aggregation_spec_as_a_list_of_pairs(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", [("revenue", "sum")])
    assert list(result.columns) == ["region", "revenue_sum"]


def test_top_n_limits_the_result(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"revenue": "sum"}, top_n=2)
    assert len(result) == 2
    assert list(result["region"]) == ["East", "North"]


def test_sort_by_a_named_measure(sales: pd.DataFrame):
    result = group_and_aggregate(
        sales, "region", {"revenue": "sum", "units": "sum"}, sort_by="units_sum"
    )
    assert list(result["units_sum"]) == [8, 8, 4]


def test_sorting_by_an_unknown_column_raises(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="Cannot sort by"):
        group_and_aggregate(sales, "region", {"revenue": "sum"}, sort_by="nope")


def test_grouping_does_not_mutate_the_input(sales: pd.DataFrame):
    before = sales.copy()
    group_and_aggregate(sales, "region", {"revenue": "sum"})
    pd.testing.assert_frame_equal(sales, before)


def test_grouping_an_empty_dataframe_returns_the_right_columns():
    empty = pd.DataFrame({"region": pd.Series(dtype="object"),
                          "revenue": pd.Series(dtype="float64")})

    result = group_and_aggregate(empty, "region", {"revenue": "sum"})

    assert result.empty
    assert list(result.columns) == ["region", "revenue_sum"]


def test_grouping_by_a_missing_column_raises(sales: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        group_and_aggregate(sales, "territory", {"revenue": "sum"})


def test_aggregating_a_missing_column_raises(sales: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        group_and_aggregate(sales, "region", {"profit": "sum"})


def test_summing_a_text_column_raises(sales: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError, match="must be numeric"):
        group_and_aggregate(sales, "region", {"label": "sum"})


def test_counting_a_text_column_is_allowed(sales: pd.DataFrame):
    result = group_and_aggregate(sales, "region", {"label": "count"})
    assert list(result.columns) == ["region", "label_count"]


def test_duplicate_grouping_columns_raise(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="Duplicate grouping"):
        group_and_aggregate(sales, ["region", "region"], {"revenue": "sum"})


def test_missing_aggregation_spec_raises(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="'aggregations' is required"):
        group_and_aggregate(sales, "region", None)


def test_bare_non_count_aggregation_raises(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="only supported for 'count'"):
        group_and_aggregate(sales, "region", "sum")


def test_group_keys_with_missing_values_are_dropped_by_default():
    frame = pd.DataFrame({"g": ["a", None, "a"], "v": [1.0, 2.0, 3.0]})

    kept = group_and_aggregate(frame, "g", {"v": "sum"})
    assert len(kept) == 1

    with_nulls = group_and_aggregate(frame, "g", {"v": "sum"}, dropna=False)
    assert len(with_nulls) == 2


# --------------------------------------------------------------------------- #
# calculate_statistics
# --------------------------------------------------------------------------- #

def test_statistics_defaults_to_every_numeric_column(sales: pd.DataFrame):
    result = calculate_statistics(sales)

    assert set(result.columns) == {"revenue", "units"}
    assert "label" not in result.values
    assert result.row_count == 6


def test_statistics_values(sales: pd.DataFrame):
    result = calculate_statistics(sales, "revenue", ["mean", "min", "max", "sum"])
    values = result.values["revenue"]

    assert values["mean"] == pytest.approx(150.0)
    assert values["min"] == pytest.approx(50.0)
    assert values["max"] == pytest.approx(400.0)
    assert values["sum"] == pytest.approx(900.0)


def test_statistics_to_frame(sales: pd.DataFrame):
    frame = calculate_statistics(sales, ["revenue"], ["mean", "std"]).to_frame()

    assert list(frame.columns) == ["column", "mean", "std"]
    assert frame.iloc[0]["column"] == "revenue"


def test_statistics_skips_non_numeric_columns_with_a_reason(sales: pd.DataFrame):
    result = calculate_statistics(sales, ["revenue", "label"])

    assert result.columns == ["revenue"]
    assert "label" in result.skipped
    assert "numeric" in result.skipped["label"]


def test_statistics_on_an_all_missing_column():
    frame = pd.DataFrame({"a": pd.Series([None, None], dtype="float64")})
    result = calculate_statistics(frame, ["a"])

    assert result.is_empty
    assert "a" in result.skipped


def test_statistics_on_an_empty_dataframe():
    result = calculate_statistics(pd.DataFrame())

    assert result.is_empty
    assert result.row_count == 0
    assert result.to_frame().empty


def test_statistics_with_an_unknown_statistic_raises(sales: pd.DataFrame):
    with pytest.raises(UnsupportedOperationError):
        calculate_statistics(sales, ["revenue"], ["mode"])


def test_statistics_with_no_statistics_raises(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="at least one statistic"):
        calculate_statistics(sales, ["revenue"], [])


def test_statistics_on_a_missing_column_raises(sales: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        calculate_statistics(sales, ["margin"])


def test_statistics_coerces_numeric_text():
    frame = pd.DataFrame({"amount": ["10", "20", "30"]})
    result = calculate_statistics(frame, ["amount"], ["mean"])
    assert result.values["amount"]["mean"] == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# calculate_correlation
# --------------------------------------------------------------------------- #

def test_perfect_positive_correlation():
    base = np.arange(50, dtype=float)
    frame = pd.DataFrame({"x": base, "y": base * 3 + 7})

    result = calculate_correlation(frame)

    assert result.strongest["correlation"] == pytest.approx(1.0, abs=1e-9)
    assert result.strongest["direction"] == "positive"
    assert result.strongest["strength"] == "very strong"


def test_negative_correlation_is_labelled():
    base = np.arange(50, dtype=float)
    frame = pd.DataFrame({"x": base, "y": -2 * base})

    pair = calculate_correlation(frame).strongest

    assert pair["correlation"] == pytest.approx(-1.0, abs=1e-9)
    assert pair["direction"] == "negative"


def test_correlation_matrix_shape(sales: pd.DataFrame):
    result = calculate_correlation(sales)

    assert result.matrix.shape == (2, 2)
    assert set(result.columns) == {"revenue", "units"}
    assert result.matrix.loc["revenue", "revenue"] == pytest.approx(1.0)


def test_threshold_filters_the_reported_pairs():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.1, 2.9, 4.2, 4.8],
            "c": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
    )
    strict = calculate_correlation(frame, threshold=0.95)

    assert all(abs(p["correlation"]) >= 0.95 for p in strict.pairs)
    assert {"a", "b"} == set(
        [strict.pairs[0]["left"], strict.pairs[0]["right"]]
    )


def test_pairs_are_sorted_by_absolute_strength():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "c": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        }
    )
    pairs = calculate_correlation(frame).pairs
    magnitudes = [abs(p["correlation"]) for p in pairs]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_correlation_ignores_constant_columns():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0], "c": [3.0, 2.0, 1.0]})

    result = calculate_correlation(frame)

    assert "b" not in result.columns
    assert any("constant" in note for note in result.notes)


def test_correlation_needs_two_numeric_columns():
    result = calculate_correlation(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))

    assert result.is_empty
    assert result.pairs == []
    assert any("two numeric columns" in note for note in result.notes)


def test_correlation_on_an_empty_dataframe():
    result = calculate_correlation(pd.DataFrame())

    assert result.is_empty
    assert result.strongest is None


def test_spearman_and_kendall_are_supported():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})

    for method in ("spearman", "kendall"):
        result = calculate_correlation(frame, method=method)
        assert result.method == method
        assert result.matrix.loc["x", "y"] == pytest.approx(1.0)


def test_unknown_correlation_method_raises(sales: pd.DataFrame):
    with pytest.raises(UnsupportedOperationError, match="correlation method"):
        calculate_correlation(sales, method="cosine")


def test_correlation_threshold_must_be_a_fraction(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="between 0 and 1"):
        calculate_correlation(sales, threshold=1.5)


def test_correlation_threshold_must_be_numeric(sales: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be a number"):
        calculate_correlation(sales, threshold="high")


def test_correlating_a_text_column_raises(sales: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        calculate_correlation(sales, ["revenue", "label"])


def test_correlation_warns_when_there_are_too_few_rows():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
    result = calculate_correlation(frame, min_periods=3)
    assert any("complete rows" in note for note in result.notes)


def test_correlation_is_serialisable(sales: pd.DataFrame):
    payload = calculate_correlation(sales).to_dict()
    assert payload["method"] == "pearson"
    assert isinstance(payload["matrix"], dict)


def test_non_dataframe_input_raises():
    with pytest.raises(InvalidDataError):
        calculate_correlation([1, 2, 3])
