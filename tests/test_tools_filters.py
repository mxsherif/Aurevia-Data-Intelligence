"""Tests for row filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from app.tools.exceptions import (
    ColumnNotFoundError,
    InvalidDataError,
    InvalidParameterError,
    UnsupportedOperationError,
)
from app.tools.filters import (
    FilterCondition,
    build_mask,
    describe_conditions,
    filter_rows,
    normalize_operator,
    supported_operators,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "East", None, "West"],
            "revenue": [100.0, 250.0, 50.0, 400.0, 175.0, None],
            "signup": [
                "2023-01-10", "2023-03-15", "2023-06-01",
                "2023-09-20", "2023-12-31", "2024-02-14",
            ],
            "active": [True, False, True, True, False, True],
        }
    )


# --------------------------------------------------------------------------- #
# Equality / inequality
# --------------------------------------------------------------------------- #

def test_equality(frame: pd.DataFrame):
    result = filter_rows(frame, {"column": "region", "operator": "eq", "value": "North"})
    assert len(result) == 2
    assert set(result["region"]) == {"North"}


def test_equality_ignores_surrounding_whitespace(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("region", "==", "  North  "))) == 2


def test_inequality_keeps_nulls(frame: pd.DataFrame):
    # A missing region is not equal to "North", so it survives a != filter.
    result = filter_rows(frame, ("region", "!=", "North"))
    assert len(result) == 4
    assert result["region"].isna().sum() == 1


def test_numeric_equality_coerces_string_input(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("revenue", "eq", "250"))) == 1


def test_boolean_equality_accepts_text(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("active", "eq", "yes"))) == 4
    assert len(filter_rows(frame, ("active", "eq", False))) == 2


# --------------------------------------------------------------------------- #
# Ordered comparisons
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("operator", "expected"),
    [("gt", 2), ("gte", 3), ("lt", 2), ("lte", 3)],
)
def test_ordered_comparisons(frame: pd.DataFrame, operator: str, expected: int):
    assert len(filter_rows(frame, ("revenue", operator, 175))) == expected


def test_comparison_excludes_missing_values(frame: pd.DataFrame):
    result = filter_rows(frame, ("revenue", ">", 0))
    assert len(result) == 5
    assert result["revenue"].notna().all()


def test_comparison_on_date_column(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("signup", ">=", "2023-09-01"))) == 3


# --------------------------------------------------------------------------- #
# Ranges
# --------------------------------------------------------------------------- #

def test_numeric_range_is_inclusive(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("revenue", "between", [100, 250]))) == 3


def test_numeric_range_accepts_mapping(frame: pd.DataFrame):
    condition = {"column": "revenue", "operator": "range", "value": {"min": 0, "max": 100}}
    assert len(filter_rows(frame, condition)) == 2


def test_reversed_range_bounds_are_swapped(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("revenue", "between", [250, 100]))) == 3


def test_not_between_excludes_the_range_and_nulls(frame: pd.DataFrame):
    result = filter_rows(frame, ("revenue", "not_between", [100, 250]))
    assert sorted(result["revenue"]) == [50.0, 400.0]


def test_date_range(frame: pd.DataFrame):
    result = filter_rows(
        frame, ("signup", "date_between", ["2023-03-01", "2023-09-30"])
    )
    assert len(result) == 3


def test_date_range_end_includes_the_whole_day(frame: pd.DataFrame):
    # 2023-12-31 as an end bound must include that day's rows.
    result = filter_rows(
        frame, ("signup", "date_between", ["2023-12-01", "2023-12-31"])
    )
    assert len(result) == 1


def test_range_needs_two_bounds(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="two bounds"):
        filter_rows(frame, ("revenue", "between", 100))


def test_non_numeric_range_bounds_raise(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be numeric"):
        filter_rows(frame, ("revenue", "between", ["low", "high"]))


# --------------------------------------------------------------------------- #
# Membership
# --------------------------------------------------------------------------- #

def test_categorical_inclusion(frame: pd.DataFrame):
    result = filter_rows(frame, ("region", "in", ["North", "West"]))
    assert len(result) == 3


def test_categorical_exclusion(frame: pd.DataFrame):
    result = filter_rows(frame, ("region", "not_in", ["North"]))
    assert "North" not in set(result["region"].dropna())


def test_inclusion_accepts_a_scalar(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("region", "in", "South"))) == 1


def test_inclusion_on_numeric_column(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("revenue", "in", [50, 400]))) == 2


def test_empty_inclusion_list_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="at least one value"):
        filter_rows(frame, ("region", "in", []))


# --------------------------------------------------------------------------- #
# Null handling
# --------------------------------------------------------------------------- #

def test_is_null(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("region", "is_null", None))) == 1
    assert len(filter_rows(frame, {"column": "revenue", "operator": "is_null"})) == 1


def test_not_null(frame: pd.DataFrame):
    assert len(filter_rows(frame, ("revenue", "not_null", None))) == 5


def test_null_operators_need_no_value(frame: pd.DataFrame):
    # The absence of 'value' must not be read as a missing argument.
    assert len(filter_rows(frame, FilterCondition("region", "is_null"))) == 1


# --------------------------------------------------------------------------- #
# Combining conditions
# --------------------------------------------------------------------------- #

def test_multiple_conditions_default_to_and(frame: pd.DataFrame):
    result = filter_rows(
        frame,
        [("region", "in", ["North", "East"]), ("revenue", ">=", 100)],
    )
    assert len(result) == 2


def test_or_logic(frame: pd.DataFrame):
    result = filter_rows(
        frame,
        [("region", "eq", "South"), ("revenue", ">", 300)],
        logic="or",
    )
    assert len(result) == 2


def test_invalid_logic_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="'logic' must be"):
        filter_rows(frame, ("revenue", ">", 0), logic="xor")


def test_no_conditions_returns_a_copy(frame: pd.DataFrame):
    for conditions in (None, []):
        result = filter_rows(frame, conditions)
        pd.testing.assert_frame_equal(result, frame)
        assert result is not frame


# --------------------------------------------------------------------------- #
# Immutability and empty results
# --------------------------------------------------------------------------- #

def test_input_dataframe_is_never_mutated(frame: pd.DataFrame):
    before = frame.copy()
    result = filter_rows(frame, ("region", "eq", "North"))

    result.loc[result.index[0], "revenue"] = -999.0

    pd.testing.assert_frame_equal(frame, before)


def test_result_is_a_new_object(frame: pd.DataFrame):
    result = filter_rows(frame, ("revenue", ">", 0))
    assert result is not frame
    assert result._is_copy is None  # not a view onto the original


def test_no_matches_returns_empty_frame_with_same_columns(frame: pd.DataFrame):
    result = filter_rows(frame, ("region", "eq", "Atlantis"))

    assert result.empty
    assert list(result.columns) == list(frame.columns)


def test_filtering_an_empty_dataframe():
    empty = pd.DataFrame({"a": pd.Series(dtype="float64")})
    result = filter_rows(empty, ("a", ">", 1))
    assert result.empty
    assert list(result.columns) == ["a"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_missing_column_raises_with_suggestion(frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'revenue'"):
        filter_rows(frame, ("revenu", ">", 1))


def test_missing_column_lists_alternatives(frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError, match="Available columns"):
        filter_rows(frame, ("zzz", ">", 1))


def test_unknown_operator_raises(frame: pd.DataFrame):
    with pytest.raises(UnsupportedOperationError, match="Unsupported filter operator"):
        filter_rows(frame, ("revenue", "approximately", 1))


def test_missing_value_for_value_operator_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="requires a value"):
        filter_rows(frame, {"column": "revenue", "operator": "gt"})


def test_malformed_condition_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="must be a dict"):
        filter_rows(frame, ["revenue"])


def test_condition_missing_operator_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="missing"):
        filter_rows(frame, {"column": "revenue"})


def test_non_dataframe_input_raises():
    with pytest.raises(InvalidDataError, match="Expected a pandas DataFrame"):
        filter_rows({"region": ["North"]}, ("region", "eq", "North"))


def test_comparing_numeric_column_with_text_raises(frame: pd.DataFrame):
    with pytest.raises(InvalidParameterError, match="cannot be compared"):
        filter_rows(frame, ("revenue", ">", "a lot"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def test_operator_aliases_resolve():
    assert normalize_operator(">=") == "gte"
    assert normalize_operator("Not Equals") == "ne"
    assert normalize_operator("isnull") == "is_null"


def test_supported_operators_covers_the_documented_set():
    expected = {
        "eq", "ne", "gt", "gte", "lt", "lte",
        "between", "date_between", "in", "not_in", "is_null", "not_null",
    }
    assert expected <= set(supported_operators())


def test_build_mask_returns_an_aligned_boolean_series(frame: pd.DataFrame):
    mask = build_mask(frame, ("region", "eq", "North"))

    assert mask.dtype == bool
    assert list(mask.index) == list(frame.index)
    assert mask.sum() == 2


def test_describe_conditions_is_human_readable(frame: pd.DataFrame):
    described = describe_conditions(
        [
            ("revenue", ">", 100),
            ("region", "in", ["North", "South"]),
            ("revenue", "is_null", None),
            ("signup", "date_between", ["2023-01-01", "2023-12-31"]),
        ]
    )
    assert described[0] == "revenue > 100"
    assert described[1].startswith("region in [North, South")
    assert described[2] == "revenue is null"
    assert "between" in described[3]
