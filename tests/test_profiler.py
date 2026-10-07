"""Tests for schema detection, type inference, and the dataset profiler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.profile import FieldType, WarningSeverity
from app.services.data_loader import load_dataframe
from app.services.profiler import (
    descriptive_statistics,
    infer_field_type,
    profile_column,
    profile_dataframe,
)


@pytest.fixture
def messy_profile(messy_frame: pd.DataFrame):
    return profile_dataframe(messy_frame, name="messy", source="fixture")


# --------------------------------------------------------------------------- #
# Schema detection
# --------------------------------------------------------------------------- #

def test_schema_shape_and_column_names(messy_frame: pd.DataFrame, messy_profile):
    assert messy_profile.row_count == len(messy_frame)
    assert messy_profile.column_count == messy_frame.shape[1]
    assert messy_profile.column_names == list(messy_frame.columns)
    assert messy_profile.memory_usage_bytes > 0


def test_profile_groups_columns_by_role(messy_profile):
    assert "monthly_charge" in messy_profile.numeric_columns
    assert "support_calls" in messy_profile.numeric_columns
    assert "region" in messy_profile.categorical_columns
    assert "is_active" in messy_profile.boolean_columns
    assert "signup_date" in messy_profile.date_columns
    assert "customer_id" in messy_profile.id_columns
    assert "plan" in messy_profile.low_cardinality_columns


def test_column_lookup_and_type_map(messy_profile):
    assert messy_profile.column("missing_column") is None
    types = messy_profile.type_map()
    assert types["monthly_charge"] == "numeric"
    assert types["signup_date"] == "datetime"


def test_profile_is_serialisable(messy_profile):
    payload = messy_profile.to_dict()

    assert payload["row_count"] == 12
    assert len(payload["columns"]) == messy_profile.column_count
    assert isinstance(payload["summary"]["numeric_columns"], list)
    assert all(isinstance(w["severity"], str) for w in payload["warnings"])


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, 2.5, 3.75, 4.0, 5.25], FieldType.NUMERIC),
        ([10, 25, 31, 47, 52, 68, 71], FieldType.INTEGER),
        (["red", "blue", "red", "green", "blue"], FieldType.CATEGORICAL),
        ([True, False, True, True, False], FieldType.BOOLEAN),
        (["yes", "no", "yes", "no", "yes"], FieldType.BOOLEAN),
        ([0, 1, 1, 0, 1, 0], FieldType.BOOLEAN),
        (["2023-01-01", "2023-06-15", "2024-02-29", "2024-11-30"], FieldType.DATETIME),
        ([None, None, None, None], FieldType.EMPTY),
    ],
)
def test_infer_field_type(values, expected):
    assert infer_field_type(pd.Series(values), "value") is expected


def test_identifier_detected_from_name_and_uniqueness():
    series = pd.Series(range(1000, 1100))
    assert infer_field_type(series, "customer_id") is FieldType.IDENTIFIER
    # Same values, a measure-like name -> an ordinary integer column.
    assert infer_field_type(series, "monthly_total") is FieldType.INTEGER


def test_unique_strings_are_identifiers_not_text():
    series = pd.Series([f"SKU-{i:05d}" for i in range(200)])
    assert infer_field_type(series, "sku") is FieldType.IDENTIFIER


def test_free_text_is_not_categorical():
    series = pd.Series(
        [f"the customer called about issue number {i} on the line" for i in range(60)]
        + ["duplicate note"] * 5
    )
    assert infer_field_type(series, "comment") is FieldType.TEXT


def test_numeric_column_is_not_mistaken_for_a_date():
    years = pd.Series([2019, 2020, 2021, 2022, 2023] * 4)
    assert infer_field_type(years, "report_year") is not FieldType.DATETIME


def test_short_codes_are_not_dates():
    codes = pd.Series(["A1", "B2", "C3", "A1", "B2"] * 4)
    assert infer_field_type(codes, "code_label") is FieldType.CATEGORICAL


def test_native_datetime_dtype_is_detected():
    series = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    assert infer_field_type(series, "observed") is FieldType.DATETIME


def test_slash_dates_are_detected():
    series = pd.Series(["01/02/2023", "15/03/2023", "28/04/2023", "09/05/2023"])
    assert infer_field_type(series, "txn_date") is FieldType.DATETIME


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #

def test_missing_value_profiling_per_column(messy_profile):
    notes = messy_profile.column("notes")
    assert notes.missing_count == 9
    assert notes.missing_pct == pytest.approx(75.0)
    assert notes.count == 3

    charge = messy_profile.column("monthly_charge")
    assert charge.missing_count == 0
    assert charge.missing_pct == 0.0


def test_missing_cells_aggregate(messy_profile):
    # 9 missing notes + 12 missing in the fully empty column.
    assert messy_profile.missing_cells == 21
    assert messy_profile.total_cells == 12 * 9
    assert messy_profile.missing_cells_pct == pytest.approx(21 / 108 * 100, abs=1e-3)
    assert set(messy_profile.columns_with_missing) == {"notes", "empty_col"}


def test_fully_empty_column_is_flagged(messy_profile):
    empty = messy_profile.column("empty_col")
    assert empty.inferred_type is FieldType.EMPTY
    assert empty.count == 0
    assert empty.missing_pct == 100.0

    codes = {w.code for w in messy_profile.warnings}
    assert "empty_columns" in codes


def test_empty_dataframe_profiles_without_error():
    profile = profile_dataframe(pd.DataFrame(), name="nothing")

    assert profile.row_count == 0
    assert profile.column_count == 0
    assert profile.missing_cells_pct == 0.0
    assert {w.code for w in profile.warnings} == {"empty_dataset"}


def test_dataframe_with_columns_but_no_rows():
    profile = profile_dataframe(pd.DataFrame(columns=["a", "b"]), name="headers")

    assert profile.row_count == 0
    assert profile.column_count == 2
    assert profile.duplicate_row_count == 0


def test_profile_dataframe_rejects_none():
    with pytest.raises(ValueError):
        profile_dataframe(None)


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #

def test_duplicate_row_detection():
    df = pd.DataFrame({"a": [1, 2, 2, 3, 3, 3], "b": ["x", "y", "y", "z", "z", "z"]})

    profile = profile_dataframe(df, name="dupes")

    assert profile.duplicate_row_count == 3  # one extra 'y' + two extra 'z'
    assert profile.duplicate_row_pct == pytest.approx(50.0)

    warning = next(w for w in profile.warnings if w.code == "duplicate_rows")
    assert warning.severity is WarningSeverity.CRITICAL


def test_no_duplicates_reported_when_rows_are_unique(messy_profile):
    assert messy_profile.duplicate_row_count == 0
    assert not [w for w in messy_profile.warnings if w.code == "duplicate_rows"]


def test_constant_column_is_flagged(messy_profile):
    plan = messy_profile.column("plan")
    assert plan.is_constant is True
    assert plan.unique_count == 1
    assert "constant_columns" in {w.code for w in messy_profile.warnings}


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

def test_date_range_is_reported(messy_profile):
    signup = messy_profile.column("signup_date")

    assert signup.inferred_type is FieldType.DATETIME
    assert signup.min_date == "2023-01-15"
    assert signup.max_date == "2023-12-31"
    assert signup.date_range_days == 350


def test_missing_date_column_warns():
    profile = profile_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert "no_date_column" in {w.code for w in profile.warnings}


def test_native_datetime_column_range():
    df = pd.DataFrame({"observed": pd.date_range("2024-03-01", periods=10, freq="D")})

    column = profile_dataframe(df).column("observed")

    assert column.min_date == "2024-03-01"
    assert column.max_date == "2024-03-10"
    assert column.date_range_days == 9


# --------------------------------------------------------------------------- #
# Outliers and statistics
# --------------------------------------------------------------------------- #

def test_outlier_detection_flags_the_extreme_value(messy_profile):
    charge = messy_profile.column("monthly_charge")

    assert charge.outlier_count == 1
    assert charge.outlier_upper_bound is not None
    assert charge.max == 9000.0
    assert charge.outlier_pct == pytest.approx(100 / 12, abs=1e-3)
    assert "outliers_detected" in {w.code for w in messy_profile.warnings}


def test_no_outliers_in_a_clean_uniform_column():
    df = pd.DataFrame({"value": list(range(1, 101))})

    column = profile_dataframe(df).column("value")

    assert column.outlier_count == 0


def test_outlier_detection_falls_back_to_sigma_on_zero_iqr():
    # 97 identical values -> IQR is 0, so the IQR fence cannot work.
    df = pd.DataFrame({"value": [5.0] * 97 + [500.0, 501.0, 502.0]})

    column = profile_dataframe(df).column("value")

    assert column.outlier_count >= 3


def test_descriptive_statistics_for_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "label": list("wxyz")})
    profile = profile_dataframe(df)

    column = profile.column("a")
    assert column.mean == pytest.approx(2.5)
    assert column.median == pytest.approx(2.5)
    assert column.min == 1.0 and column.max == 4.0
    assert column.q1 == pytest.approx(1.75)
    assert column.q3 == pytest.approx(3.25)

    stats = descriptive_statistics(df, profile)
    assert "mean" in stats.columns
    assert list(stats.index) == ["a"]


def test_descriptive_statistics_empty_without_numeric_columns():
    df = pd.DataFrame({"label": list("abc")})
    assert descriptive_statistics(df, profile_dataframe(df)).empty


def test_zeros_and_negatives_are_counted():
    df = pd.DataFrame({"monthly_charge": [-10.0, 0.0, 0.0, 5.0, 20.0]})

    profile = profile_dataframe(df)
    column = profile.column("monthly_charge")

    assert column.zeros == 2
    assert column.negatives == 1
    assert "unexpected_negative_values" in {w.code for w in profile.warnings}


def test_infinite_values_do_not_break_numeric_stats():
    df = pd.DataFrame({"value": [1.0, 2.0, np.inf, -np.inf, 4.0]})

    column = profile_dataframe(df).column("value")

    assert column.max == 4.0
    assert column.min == 1.0


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #

def test_strong_correlation_is_detected():
    base = np.arange(100, dtype=float)
    df = pd.DataFrame({"x": base, "y": base * 2 + 1, "noise": np.tile([1.0, 2.0], 50)})

    profile = profile_dataframe(df)
    pair = next(
        p for p in profile.correlations if {p["left"], p["right"]} == {"x", "y"}
    )

    assert pair["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert pair["direction"] == "positive"
    assert "correlated_columns" in {w.code for w in profile.warnings}


def test_no_correlations_with_a_single_numeric_column():
    assert profile_dataframe(pd.DataFrame({"x": [1, 2, 3]})).correlations == []


# --------------------------------------------------------------------------- #
# Column-level edge cases
# --------------------------------------------------------------------------- #

def test_profile_column_survives_unhashable_values():
    series = pd.Series([[1, 2], [3, 4], [1, 2]], name="payload")

    column = profile_column(series)

    assert column.count == 3
    assert column.unique_count == 2


def test_low_cardinality_detection():
    df = pd.DataFrame({"tier": ["a", "b", "c"] * 40, "uid": [f"u{i}" for i in range(120)]})

    profile = profile_dataframe(df)

    assert "tier" in profile.low_cardinality_columns
    assert "uid" not in profile.low_cardinality_columns


def test_top_values_are_recorded():
    df = pd.DataFrame({"region": ["N"] * 6 + ["S"] * 3 + ["E"]})

    column = profile_dataframe(df).column("region")

    assert list(column.top_values.items())[0] == ("N", 6)
    assert column.top_values["S"] == 3


def test_warnings_are_sorted_most_severe_first():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "blank": [None] * 4,
            "b": [2.0, 2.0, 2.0, 2.0],
        }
    )

    severities = [w.severity for w in profile_dataframe(df).warnings]

    order = {WarningSeverity.CRITICAL: 0, WarningSeverity.WARNING: 1, WarningSeverity.INFO: 2}
    assert severities == sorted(severities, key=lambda s: order[s])


# --------------------------------------------------------------------------- #
# The bundled sample dataset
# --------------------------------------------------------------------------- #

def test_sample_dataset_profiles_as_expected(sample_dataset_path: Path, settings):
    result = load_dataframe(sample_dataset_path, settings=settings)
    profile = profile_dataframe(
        result.dataframe, name=result.source_name, source="csv"
    )

    assert profile.row_count == 5_000
    assert profile.column_count == 15

    assert profile.id_columns == ["customer_id"]
    assert profile.date_columns == ["signup_date"]
    assert "churn" in profile.boolean_columns
    for column in ("monthly_charge", "data_usage_gb", "revenue", "tenure_months"):
        assert column in profile.numeric_columns
    for column in ("region", "city", "contract_type", "product_type", "network_type"):
        assert column in profile.categorical_columns

    # The generator plants missing values, negative charges and outliers.
    assert profile.missing_cells > 0
    assert profile.column("satisfaction_score").missing_count > 0
    assert profile.column("monthly_charge").negatives > 0
    assert profile.column("data_usage_gb").outlier_count > 0

    signup = profile.column("signup_date")
    assert signup.min_date == "2021-01-01"
    assert signup.max_date == "2024-12-31"

    codes = {w.code for w in profile.warnings}
    assert "outliers_detected" in codes
    assert "unexpected_negative_values" in codes
