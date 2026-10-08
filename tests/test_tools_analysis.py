"""Tests for schema, column summaries, segments, ranking and outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.profile import FieldType
from app.tools.analysis import (
    compare_segments,
    detect_outliers,
    get_column_summary,
    get_dataset_schema,
    rank_values,
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
            "customer_id": [f"C{i:03d}" for i in range(10)],
            "region": ["North"] * 4 + ["South"] * 3 + ["East"] * 2 + ["West"],
            "signup": pd.to_datetime(
                [
                    "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01",
                    "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01",
                    "2023-09-01", "2023-10-01",
                ]
            ),
            "revenue": [100.0, 120.0, 90.0, 110.0, 300.0, 280.0, 310.0, 50.0, 60.0, 5000.0],
            "active": [True, False, True, True, True, False, True, True, False, True],
            "notes": [None, "a", None, None, "b", None, None, None, None, None],
        }
    )


# --------------------------------------------------------------------------- #
# get_dataset_schema
# --------------------------------------------------------------------------- #

def test_schema_reports_shape_and_columns(frame: pd.DataFrame):
    schema = get_dataset_schema(frame, name="customers")

    assert schema.name == "customers"
    assert schema.row_count == 10
    assert schema.column_count == 6
    assert schema.column_names == list(frame.columns)


def test_schema_assigns_field_roles(frame: pd.DataFrame):
    schema = get_dataset_schema(frame)

    assert "revenue" in schema.numeric_columns
    assert "region" in schema.categorical_columns
    assert "active" in schema.boolean_columns
    assert "signup" in schema.date_columns
    assert "customer_id" in schema.identifier_columns


def test_schema_reports_nullability(frame: pd.DataFrame):
    schema = get_dataset_schema(frame)

    assert schema.column("notes").nullable is True
    assert schema.column("notes").missing_count == 8
    assert schema.column("revenue").nullable is False
    assert schema.column("revenue").missing_pct == 0.0


def test_schema_includes_sample_values(frame: pd.DataFrame):
    schema = get_dataset_schema(frame, sample_size=2)
    samples = schema.column("region").sample_values

    assert len(samples) == 2
    assert samples[0] == "North"


def test_schema_samples_are_json_safe(frame: pd.DataFrame):
    import json

    payload = get_dataset_schema(frame).to_dict()
    json.dumps(payload)  # must not raise on numpy / Timestamp values

    assert set(payload["roles"]) == {
        "numeric", "categorical", "boolean", "date", "identifier"
    }


def test_schema_lookup_of_a_missing_column_returns_none(frame: pd.DataFrame):
    assert get_dataset_schema(frame).column("nope") is None


def test_schema_of_an_empty_dataframe():
    schema = get_dataset_schema(pd.DataFrame())

    assert schema.row_count == 0
    assert schema.column_count == 0
    assert schema.columns == []


def test_schema_of_a_frame_with_headers_only():
    schema = get_dataset_schema(pd.DataFrame(columns=["a", "b"]))

    assert schema.row_count == 0
    assert schema.column_count == 2
    assert schema.column("a").missing_pct == 0.0


def test_schema_rejects_non_dataframes():
    with pytest.raises(InvalidDataError):
        get_dataset_schema("not a dataframe")


def test_schema_sample_size_must_be_positive(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="positive integer"):
        get_dataset_schema(frame, sample_size=0)


# --------------------------------------------------------------------------- #
# get_column_summary
# --------------------------------------------------------------------------- #

def test_column_summary_for_a_numeric_column(frame: pd.DataFrame):
    summary = get_column_summary(frame, "revenue")

    assert summary.name == "revenue"
    assert summary.inferred_type in (FieldType.NUMERIC, FieldType.INTEGER)
    assert summary.count == 10
    assert summary.min == 50.0
    assert summary.max == 5000.0
    assert summary.outlier_count >= 1


def test_column_summary_for_a_categorical_column(frame: pd.DataFrame):
    summary = get_column_summary(frame, "region")

    assert summary.inferred_type is FieldType.CATEGORICAL
    assert summary.unique_count == 4
    assert summary.top_values["North"] == 4


def test_column_summary_for_a_date_column(frame: pd.DataFrame):
    summary = get_column_summary(frame, "signup")

    assert summary.inferred_type is FieldType.DATETIME
    assert summary.min_date == "2023-01-01"
    assert summary.max_date == "2023-10-01"
    assert summary.date_range_days == 273


def test_column_summary_reports_missing_values(frame: pd.DataFrame):
    summary = get_column_summary(frame, "notes")
    assert summary.missing_count == 8
    assert summary.missing_pct == pytest.approx(80.0)


def test_column_summary_of_a_missing_column_raises(frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        get_column_summary(frame, "profit")


def test_column_summary_requires_a_column_name(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="column name is required"):
        get_column_summary(frame, "")


def test_column_summary_rejects_non_string_names(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be strings"):
        get_column_summary(frame, 3)


# --------------------------------------------------------------------------- #
# compare_segments
# --------------------------------------------------------------------------- #

def test_compare_segments_against_the_overall_baseline(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", aggregation="mean")

    assert result.segment_column == "region"
    assert result.aggregation == "mean"
    assert result.baseline == "overall"
    assert result.baseline_value == pytest.approx(frame["revenue"].mean())
    assert len(result.dataframe) == 4


def test_segments_are_sorted_by_value_descending(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", aggregation="sum")
    values = result.dataframe["value"].tolist()
    assert values == sorted(values, reverse=True)
    assert result.best["segment"] == "West"


def test_segment_difference_is_relative_to_the_baseline(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", aggregation="mean")
    row = result.dataframe[result.dataframe["segment"] == "North"].iloc[0]

    assert row["value"] == pytest.approx(105.0)
    assert row["difference"] == pytest.approx(105.0 - result.baseline_value)


def test_segment_shares_sum_to_one_hundred(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", aggregation="sum")
    assert result.dataframe["share_pct"].sum() == pytest.approx(100.0)


def test_compare_segments_against_a_named_baseline(frame: pd.DataFrame):
    result = compare_segments(
        frame, "region", "revenue", aggregation="mean", baseline="North"
    )

    assert result.baseline == "North"
    assert result.baseline_value == pytest.approx(105.0)
    north = result.dataframe[result.dataframe["segment"] == "North"].iloc[0]
    assert north["difference"] == pytest.approx(0.0)


def test_unknown_baseline_segment_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="Baseline segment"):
        compare_segments(frame, "region", "revenue", baseline="Atlantis")


def test_compare_segments_can_be_restricted(frame: pd.DataFrame):
    result = compare_segments(
        frame, "region", "revenue", segments=["North", "South"]
    )
    assert set(result.dataframe["segment"]) == {"North", "South"}


def test_restricting_to_unknown_segments_returns_empty(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", segments=["Atlantis"])

    assert result.is_empty
    assert result.best is None
    assert any("No rows matched" in note for note in result.notes)


def test_compare_segments_without_a_metric_counts_rows(frame: pd.DataFrame):
    result = compare_segments(frame, "region")

    assert result.aggregation == "count"
    north = result.dataframe[result.dataframe["segment"] == "North"].iloc[0]
    assert north["value"] == 4


def test_compare_segments_row_counts(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", aggregation="mean")
    row = result.dataframe[result.dataframe["segment"] == "South"].iloc[0]
    assert row["row_count"] == 3


def test_compare_segments_top_n(frame: pd.DataFrame):
    result = compare_segments(frame, "region", "revenue", aggregation="sum", top_n=2)
    assert len(result.dataframe) == 2


def test_compare_segments_on_an_empty_dataframe():
    empty = pd.DataFrame({"region": pd.Series(dtype="object"),
                          "revenue": pd.Series(dtype="float64")})
    result = compare_segments(empty, "region", "revenue")

    assert result.is_empty
    assert list(result.dataframe.columns) == [
        "segment", "row_count", "value", "share_pct", "difference", "difference_pct"
    ]


def test_compare_segments_with_a_text_metric_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        compare_segments(frame, "region", "customer_id", aggregation="mean")


def test_compare_segments_with_a_missing_column_raises(frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        compare_segments(frame, "territory", "revenue")


def test_compare_segments_rejects_a_scalar_segment_list(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be a list"):
        compare_segments(frame, "region", "revenue", segments="North")


def test_compare_segments_is_serialisable(frame: pd.DataFrame):
    payload = compare_segments(frame, "region", "revenue").to_dict()
    assert payload["segment_column"] == "region"
    assert len(payload["segments"]) == 4


# --------------------------------------------------------------------------- #
# rank_values
# --------------------------------------------------------------------------- #

def test_rank_groups_by_an_aggregated_metric(frame: pd.DataFrame):
    result = rank_values(frame, "region", "revenue", aggregation="sum", top_n=3)

    assert list(result.dataframe["rank"]) == [1, 2, 3]
    assert result.dataframe.iloc[0]["region"] == "West"
    assert result.dataframe.iloc[0]["value"] == pytest.approx(5000.0)
    assert result.total_candidates == 4


def test_rank_ascending_finds_the_bottom(frame: pd.DataFrame):
    result = rank_values(frame, "region", "revenue", aggregation="sum", ascending=True)
    assert result.dataframe.iloc[0]["region"] == "East"


def test_rank_without_a_metric_counts_rows(frame: pd.DataFrame):
    result = rank_values(frame, "region")

    assert result.aggregation == "count"
    assert result.dataframe.iloc[0]["region"] == "North"
    assert result.dataframe.iloc[0]["value"] == 4


def test_rank_includes_row_counts(frame: pd.DataFrame):
    result = rank_values(frame, "region", "revenue", aggregation="mean")
    assert "row_count" in result.dataframe.columns


def test_top_n_limits_and_notes_the_remainder(frame: pd.DataFrame):
    result = rank_values(frame, "region", "revenue", top_n=2)

    assert len(result.dataframe) == 2
    assert any("top 2 of 4" in note for note in result.notes)


def test_rank_rows_with_aggregation_none(frame: pd.DataFrame):
    result = rank_values(frame, "revenue", aggregation=None, top_n=3)

    assert result.aggregation is None
    assert list(result.dataframe["value"]) == [5000.0, 310.0, 300.0]
    assert result.total_candidates == 10


def test_row_level_ranking_rejects_a_metric_column(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="row-level"):
        rank_values(frame, "revenue", "region", aggregation=None)


def test_row_level_ranking_of_a_text_column_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError):
        rank_values(frame, "region", aggregation=None)


def test_ranking_an_empty_dataframe():
    empty = pd.DataFrame({"region": pd.Series(dtype="object"),
                          "revenue": pd.Series(dtype="float64")})
    result = rank_values(empty, "region", "revenue")

    assert result.is_empty
    assert result.top is None
    assert any("empty" in note for note in result.notes)


def test_ranking_a_missing_column_raises(frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        rank_values(frame, "territory")


def test_top_n_must_be_a_positive_integer(frame: pd.DataFrame):
    for bad in (0, -1, 2.5, "ten"):
        with pytest.raises(InvalidParameterError, match="positive integer"):
            rank_values(frame, "region", top_n=bad)


def test_ranking_is_serialisable(frame: pd.DataFrame):
    payload = rank_values(frame, "region", "revenue").to_dict()
    assert payload["column"] == "region"
    assert payload["entries"][0]["rank"] == 1


# --------------------------------------------------------------------------- #
# detect_outliers
# --------------------------------------------------------------------------- #

def test_iqr_outlier_detection(frame: pd.DataFrame):
    result = detect_outliers(frame, "revenue")

    assert result.method == "iqr"
    assert result.threshold == 1.5
    assert result.count >= 1
    assert 5000.0 in result.rows["revenue"].tolist()
    assert result.upper_bound is not None and result.upper_bound < 5000.0


def test_outlier_percentage(frame: pd.DataFrame):
    result = detect_outliers(frame, "revenue")
    assert result.pct == pytest.approx(100.0 * result.count / 10)
    assert result.values_considered == 10


def test_zscore_outlier_detection():
    values = [10.0] * 30 + [500.0]
    result = detect_outliers(pd.DataFrame({"v": values}), "v", method="zscore")

    assert result.method == "zscore"
    assert result.count == 1


def test_a_larger_threshold_finds_fewer_outliers(frame: pd.DataFrame):
    strict = detect_outliers(frame, "revenue", threshold=5.0)
    loose = detect_outliers(frame, "revenue", threshold=1.5)
    assert strict.count <= loose.count


def test_outlier_rows_are_ordered_most_extreme_first():
    # A spread base so the IQR is non-zero, plus two extremes.
    frame = pd.DataFrame({"v": [float(v) for v in range(1, 21)] + [500.0, 1000.0]})
    result = detect_outliers(frame, "v")

    assert result.count == 2
    assert result.rows.iloc[0]["v"] == 1000.0


def test_max_rows_caps_the_returned_rows():
    frame = pd.DataFrame(
        {"v": [float(v) for v in range(1, 101)] + list(np.arange(500.0, 520.0))}
    )
    result = detect_outliers(frame, "v", max_rows=5)

    assert result.count == 20
    assert len(result.rows) == 5
    assert any("most extreme" in note for note in result.notes)


def test_no_outliers_in_a_uniform_column():
    result = detect_outliers(pd.DataFrame({"v": list(range(1, 101))}), "v")

    assert result.is_empty
    assert result.count == 0
    assert result.rows.empty


def test_constant_column_reports_a_note_not_an_error():
    result = detect_outliers(pd.DataFrame({"v": [5.0] * 10}), "v")

    assert result.is_empty
    assert any("interquartile range is zero" in note for note in result.notes)


def test_constant_column_with_zscore_reports_a_note():
    result = detect_outliers(pd.DataFrame({"v": [5.0] * 10}), "v", method="zscore")
    assert any("no variation" in note for note in result.notes)


def test_outliers_in_an_all_missing_column():
    frame = pd.DataFrame({"v": pd.Series([None, None], dtype="float64")})
    result = detect_outliers(frame, "v")

    assert result.is_empty
    assert any("no finite numeric values" in note for note in result.notes)


def test_outliers_on_an_empty_dataframe():
    empty = pd.DataFrame({"v": pd.Series(dtype="float64")})
    result = detect_outliers(empty, "v")
    assert result.is_empty


def test_outliers_ignore_infinities():
    frame = pd.DataFrame({"v": [1.0, 2.0, np.inf, 3.0, -np.inf, 4.0]})
    result = detect_outliers(frame, "v")
    assert result.values_considered == 4


def test_outliers_on_a_text_column_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError, match="must be numeric"):
        detect_outliers(frame, "region")


def test_outliers_on_a_date_column_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidColumnTypeError, match="date column"):
        detect_outliers(frame, "signup")


def test_outliers_on_a_missing_column_raises(frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        detect_outliers(frame, "profit")


def test_unknown_outlier_method_raises(frame: pd.DataFrame):
    with pytest.raises(UnsupportedOperationError, match="outlier method"):
        detect_outliers(frame, "revenue", method="isolation_forest")


def test_outlier_threshold_must_be_positive(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be positive"):
        detect_outliers(frame, "revenue", threshold=0)


def test_outlier_threshold_must_be_numeric(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be a number"):
        detect_outliers(frame, "revenue", threshold="high")


def test_outlier_detection_does_not_mutate_the_input(frame: pd.DataFrame):
    before = frame.copy()
    result = detect_outliers(frame, "revenue")
    result.rows.loc[result.rows.index[0], "revenue"] = -1.0
    pd.testing.assert_frame_equal(frame, before)


def test_outliers_are_serialisable(frame: pd.DataFrame):
    payload = detect_outliers(frame, "revenue").to_dict()
    assert payload["column"] == "revenue"
    assert payload["method"] == "iqr"
    assert isinstance(payload["indices"], list)
