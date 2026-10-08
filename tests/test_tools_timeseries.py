"""Tests for time-series aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tools.exceptions import (
    ColumnNotFoundError,
    InvalidColumnTypeError,
    InvalidDataError,
    UnsupportedOperationError,
)
from app.tools.timeseries import (
    FREQUENCIES,
    calculate_time_trend,
    resolve_frequency,
    supported_frequencies,
)


@pytest.fixture
def daily() -> pd.DataFrame:
    """Two full years of daily rows, one unit of revenue each."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    return pd.DataFrame({"date": dates, "revenue": 1.0, "units": 2})


@pytest.fixture
def sparse() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2023-03-15", "2023-01-10", "2023-01-20",
                "2023-02-05", "2023-03-01", "2023-01-31",
            ],
            "amount": [30.0, 10.0, 5.0, 20.0, 40.0, 15.0],
        }
    )


# --------------------------------------------------------------------------- #
# Frequencies
# --------------------------------------------------------------------------- #

def test_all_required_frequencies_are_supported():
    assert supported_frequencies() == ["daily", "weekly", "monthly", "quarterly", "yearly"]
    assert set(FREQUENCIES) == set(supported_frequencies())


def test_frequency_aliases_resolve():
    assert resolve_frequency("month").name == "monthly"
    assert resolve_frequency("Q").name == "quarterly"
    assert resolve_frequency("ANNUAL").name == "yearly"


def test_unknown_frequency_raises():
    with pytest.raises(UnsupportedOperationError, match="Unsupported frequency"):
        calculate_time_trend(
            pd.DataFrame({"d": ["2023-01-01"], "v": [1.0]}), "d", "v", frequency="hourly"
        )


# --------------------------------------------------------------------------- #
# Aggregation by frequency
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("frequency", "expected_periods"),
    [("daily", 731), ("monthly", 24), ("quarterly", 8), ("yearly", 2)],
)
def test_period_counts_per_frequency(daily: pd.DataFrame, frequency, expected_periods):
    result = calculate_time_trend(daily, "date", "revenue", frequency=frequency)
    assert len(result.dataframe) == expected_periods


def test_weekly_aggregation_covers_every_row(daily: pd.DataFrame):
    result = calculate_time_trend(daily, "date", "revenue", frequency="weekly")

    assert 104 <= len(result.dataframe) <= 106
    assert result.dataframe["value"].sum() == pytest.approx(len(daily))


def test_monthly_sums_are_correct(daily: pd.DataFrame):
    result = calculate_time_trend(daily, "date", "revenue", frequency="monthly")
    january = result.dataframe.iloc[0]

    assert january["period_label"] == "2023-01"
    assert january["value"] == pytest.approx(31.0)
    assert january["row_count"] == 31


def test_yearly_aggregation_totals(daily: pd.DataFrame):
    result = calculate_time_trend(daily, "date", "revenue", frequency="yearly")

    assert list(result.dataframe["period_label"]) == ["2023", "2024"]
    assert list(result.dataframe["value"]) == [365.0, 366.0]


def test_quarterly_labels(daily: pd.DataFrame):
    result = calculate_time_trend(daily, "date", "revenue", frequency="quarterly")
    assert result.dataframe["period_label"].iloc[0] == "2023-Q1"
    assert result.dataframe["period_label"].iloc[-1] == "2024-Q4"


def test_aggregation_can_be_changed(daily: pd.DataFrame):
    mean = calculate_time_trend(daily, "date", "revenue", aggregation="mean")
    assert mean.dataframe["value"].iloc[0] == pytest.approx(1.0)

    maximum = calculate_time_trend(daily, "date", "units", aggregation="max")
    assert maximum.dataframe["value"].iloc[0] == 2


def test_no_value_column_counts_rows(sparse: pd.DataFrame):
    result = calculate_time_trend(sparse, "date", frequency="monthly")

    assert result.value_column is None
    assert result.aggregation == "count"
    assert list(result.dataframe["value"]) == [3, 1, 2]


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #

def test_output_is_chronological_regardless_of_input_order(sparse: pd.DataFrame):
    result = calculate_time_trend(sparse, "date", "amount", frequency="monthly")

    assert list(result.dataframe["period_label"]) == ["2023-01", "2023-02", "2023-03"]
    assert result.dataframe["period"].is_monotonic_increasing
    assert result.first_period == "2023-01"
    assert result.last_period == "2023-03"


def test_daily_output_is_sorted_even_from_shuffled_input(daily: pd.DataFrame):
    shuffled = daily.sample(frac=1.0, random_state=7)
    result = calculate_time_trend(shuffled, "date", "revenue", frequency="daily")
    assert result.dataframe["period"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Percentage change
# --------------------------------------------------------------------------- #

def test_period_over_period_percentage_change(sparse: pd.DataFrame):
    result = calculate_time_trend(sparse, "date", "amount", frequency="monthly")
    change = result.dataframe["pct_change"]

    assert np.isnan(change.iloc[0])
    # 30 -> 20 is -33.3%, 20 -> 70 is +250%.
    assert change.iloc[1] == pytest.approx(-33.3333, abs=1e-3)
    assert change.iloc[2] == pytest.approx(250.0)


def test_total_change_from_first_to_last(sparse: pd.DataFrame):
    result = calculate_time_trend(sparse, "date", "amount", frequency="monthly")
    assert result.total_change_pct == pytest.approx(133.3333, abs=1e-3)


def test_total_change_needs_two_periods():
    frame = pd.DataFrame({"d": ["2023-01-01"], "v": [10.0]})
    assert calculate_time_trend(frame, "d", "v").total_change_pct is None


# --------------------------------------------------------------------------- #
# Gap filling
# --------------------------------------------------------------------------- #

def test_gaps_are_not_filled_by_default():
    frame = pd.DataFrame(
        {"d": ["2023-01-15", "2023-04-15"], "v": [10.0, 20.0]}
    )
    result = calculate_time_trend(frame, "d", "v", frequency="monthly")
    assert len(result.dataframe) == 2


def test_fill_gaps_inserts_the_missing_periods():
    frame = pd.DataFrame({"d": ["2023-01-15", "2023-04-15"], "v": [10.0, 20.0]})
    result = calculate_time_trend(
        frame, "d", "v", frequency="monthly", fill_gaps=True, aggregation="sum"
    )

    assert list(result.dataframe["period_label"]) == [
        "2023-01", "2023-02", "2023-03", "2023-04"
    ]
    # Sums of nothing are zero, and the inserted periods hold no rows.
    assert list(result.dataframe["value"]) == [10.0, 0.0, 0.0, 20.0]
    assert list(result.dataframe["row_count"]) == [1, 0, 0, 1]


# --------------------------------------------------------------------------- #
# Missing and invalid dates
# --------------------------------------------------------------------------- #

def test_missing_dates_are_excluded_and_counted():
    frame = pd.DataFrame(
        {"d": ["2023-01-01", None, "2023-02-01", None], "v": [1.0, 2.0, 3.0, 4.0]}
    )
    result = calculate_time_trend(frame, "d", "v", frequency="monthly")

    assert result.rows_missing_date == 2
    assert result.rows_used == 2
    assert any("no date" in note for note in result.notes)


def test_unparseable_dates_are_excluded_and_counted():
    frame = pd.DataFrame(
        {
            "d": ["2023-01-01", "not a date", "2023-02-01", "2023-03-01"],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = calculate_time_trend(frame, "d", "v", frequency="monthly")

    assert result.rows_unparseable_date == 1
    assert result.rows_used == 3
    assert any("unrecognisable" in note for note in result.notes)


def test_all_dates_missing_returns_an_empty_trend():
    frame = pd.DataFrame({"d": [None, None], "v": [1.0, 2.0]})

    with pytest.raises(InvalidColumnTypeError, match="entirely empty"):
        calculate_time_trend(frame, "d", "v")


def test_a_column_with_no_recognisable_dates_raises():
    frame = pd.DataFrame({"d": ["x", "y", "z"], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(InvalidColumnTypeError, match="no recognisable dates"):
        calculate_time_trend(frame, "d", "v")


def test_a_numeric_date_column_raises():
    frame = pd.DataFrame({"year": [2020, 2021, 2022], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(InvalidColumnTypeError, match="numeric"):
        calculate_time_trend(frame, "year", "v")


def test_empty_trend_frame_has_the_right_columns():
    # A real datetime column that happens to hold only NaT: the dtype is
    # already correct, so validation passes but no period survives.
    frame = pd.DataFrame(
        {
            "d": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
            "v": [1.0, 2.0],
        }
    )
    result = calculate_time_trend(frame, "d", "v")

    assert result.is_empty
    assert result.rows_missing_date == 2
    assert any("no rows with a usable date" in n.lower() for n in result.notes)
    assert list(result.dataframe.columns) == [
        "period", "period_label", "value", "row_count", "pct_change"
    ]
    assert result.periods == []
    assert result.values == []
    assert result.first_period is None


# --------------------------------------------------------------------------- #
# Validation and immutability
# --------------------------------------------------------------------------- #

def test_missing_date_column_raises(sparse: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        calculate_time_trend(sparse, "timestamp", "amount")


def test_missing_value_column_raises(sparse: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        calculate_time_trend(sparse, "date", "total")


def test_summing_a_text_value_column_raises():
    frame = pd.DataFrame({"d": ["2023-01-01", "2023-02-01"], "label": ["a", "b"]})
    with pytest.raises(InvalidColumnTypeError, match="must be numeric"):
        calculate_time_trend(frame, "d", "label", aggregation="sum")


def test_counting_a_text_value_column_is_allowed():
    frame = pd.DataFrame({"d": ["2023-01-01", "2023-01-02"], "label": ["a", "b"]})
    result = calculate_time_trend(frame, "d", "label", aggregation="count")
    assert result.dataframe["value"].iloc[0] == 2


def test_non_dataframe_input_raises():
    with pytest.raises(InvalidDataError):
        calculate_time_trend("nope", "d", "v")


def test_input_is_not_mutated(sparse: pd.DataFrame):
    before = sparse.copy()
    calculate_time_trend(sparse, "date", "amount", frequency="monthly")
    pd.testing.assert_frame_equal(sparse, before)


def test_native_datetime_column_works(daily: pd.DataFrame):
    result = calculate_time_trend(daily, "date", "revenue", frequency="monthly")
    assert result.date_column == "date"
    assert not result.is_empty


def test_result_is_serialisable(sparse: pd.DataFrame):
    payload = calculate_time_trend(sparse, "date", "amount").to_dict()

    assert payload["frequency"] == "monthly"
    assert payload["period_count"] == 3
    assert "period" not in payload["points"][0]  # the raw timestamp is dropped
    assert payload["points"][0]["period_label"] == "2023-01"
