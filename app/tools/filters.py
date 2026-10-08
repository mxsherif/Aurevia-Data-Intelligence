"""Row filtering.

Filters are expressed as a list of conditions, each a column / operator / value
triple. Conditions can be plain dicts -- which is what an LLM agent will emit in
Phase 3 -- or :class:`FilterCondition` instances.

:func:`filter_rows` always returns a **new** dataframe; the input is never
mutated and never aliased.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
from pandas.api import types as pdt

from app.tools.exceptions import (
    InvalidParameterError,
    UnsupportedOperationError,
)
from app.tools.validation import (
    require_column,
    require_dataframe,
    require_datetime_column,
    require_numeric_column,
)

#: Operators that ignore the `value` field entirely.
NO_VALUE_OPERATORS = frozenset({"is_null", "not_null"})

#: Operators that need a two-element [low, high] value.
RANGE_OPERATORS = frozenset({"between", "not_between", "date_between"})

#: Operators that need an iterable of values.
MEMBERSHIP_OPERATORS = frozenset({"in", "not_in"})

OPERATOR_ALIASES: dict[str, str] = {
    "=": "eq",
    "==": "eq",
    "equals": "eq",
    "equal": "eq",
    "!=": "ne",
    "<>": "ne",
    "not_equals": "ne",
    "not_equal": "ne",
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
    "greater_than": "gt",
    "greater_than_or_equal": "gte",
    "less_than": "lt",
    "less_than_or_equal": "lte",
    "range": "between",
    "in_range": "between",
    "date_range": "date_between",
    "isin": "in",
    "one_of": "in",
    "not_isin": "not_in",
    "isnull": "is_null",
    "is_na": "is_null",
    "missing": "is_null",
    "notnull": "not_null",
    "is_not_null": "not_null",
    "not_na": "not_null",
}


@dataclass(frozen=True)
class FilterCondition:
    """A single ``column <operator> value`` test."""

    column: str
    operator: str
    value: Any = None

    @classmethod
    def from_any(cls, raw: Any) -> FilterCondition:
        """Build a condition from a dict, a tuple, or another condition."""
        if isinstance(raw, FilterCondition):
            return raw
        if isinstance(raw, Mapping):
            missing = {"column", "operator"} - set(raw)
            if missing:
                raise InvalidParameterError(
                    f"Filter condition is missing {sorted(missing)}: {dict(raw)!r}"
                )
            return cls(
                column=raw["column"],
                operator=raw["operator"],
                value=raw.get("value"),
            )
        if isinstance(raw, (tuple, list)) and len(raw) in (2, 3):
            column, operator, *rest = raw
            return cls(column=column, operator=operator, value=rest[0] if rest else None)
        raise InvalidParameterError(
            "A filter condition must be a dict with 'column' and 'operator', "
            f"a (column, operator, value) tuple, or a FilterCondition -- got {raw!r}."
        )

    def describe(self) -> str:
        """Human-readable rendering, used in UI captions and tool metadata."""
        operator = normalize_operator(self.operator)
        if operator in NO_VALUE_OPERATORS:
            return f"{self.column} {operator.replace('_', ' ')}"
        if operator in RANGE_OPERATORS:
            low, high = _range_bounds(self.value, operator)
            return f"{self.column} between {low} and {high}"
        if operator in MEMBERSHIP_OPERATORS:
            values = list(_membership_values(self.value, operator))
            shown = ", ".join(map(str, values[:5]))
            more = f", +{len(values) - 5} more" if len(values) > 5 else ""
            verb = "in" if operator == "in" else "not in"
            return f"{self.column} {verb} [{shown}{more}]"
        symbols = {"eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
        return f"{self.column} {symbols.get(operator, operator)} {self.value!r}"

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "operator": self.operator, "value": self.value}


def normalize_operator(operator: Any) -> str:
    """Resolve an operator name or symbol to its canonical form."""
    if not isinstance(operator, str):
        raise UnsupportedOperationError("filter operator", operator, supported_operators())
    key = operator.strip().lower().replace(" ", "_").replace("-", "_")
    key = OPERATOR_ALIASES.get(key, key)
    if key not in _OPERATORS:
        raise UnsupportedOperationError("filter operator", operator, supported_operators())
    return key


def supported_operators() -> list[str]:
    return sorted(_OPERATORS)


# --------------------------------------------------------------------------- #
# Value coercion
# --------------------------------------------------------------------------- #

def _range_bounds(value: Any, operator: str) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        if {"min", "max"} <= set(value):
            return value["min"], value["max"]
        if {"start", "end"} <= set(value):
            return value["start"], value["end"]
        raise InvalidParameterError(
            f"The '{operator}' operator needs a mapping with min/max or start/end, "
            f"got {dict(value)!r}."
        )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    raise InvalidParameterError(
        f"The '{operator}' operator needs two bounds as [low, high], got {value!r}."
    )


def _membership_values(value: Any, operator: str) -> list[Any]:
    if value is None:
        raise InvalidParameterError(f"The '{operator}' operator needs a list of values.")
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [value]
    values = list(value)
    if not values:
        raise InvalidParameterError(
            f"The '{operator}' operator needs at least one value, got an empty list."
        )
    return values


def _comparable(df: pd.DataFrame, column: str, value: Any) -> tuple[pd.Series, Any]:
    """Align a column and a scalar so ``>`` and friends compare like with like.

    Numeric columns compare numerically, date columns compare as timestamps,
    and everything else compares as trimmed strings.
    """
    series = df[column]

    if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
        coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(coerced):
            raise InvalidParameterError(
                f"Column '{column}' is numeric, so it cannot be compared with {value!r}."
            )
        return series, coerced

    if pdt.is_datetime64_any_dtype(series):
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            raise InvalidParameterError(
                f"Column '{column}' is a date column, so it cannot be compared "
                f"with {value!r}."
            )
        return series, timestamp

    if pdt.is_bool_dtype(series):
        return series, _as_bool(value, column)

    return series.astype("string").str.strip(), str(value).strip()


def _as_bool(value: Any, column: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    raise InvalidParameterError(
        f"Column '{column}' is boolean, so it cannot be compared with {value!r}."
    )


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #

MaskBuilder = Callable[[pd.DataFrame, FilterCondition], "pd.Series[bool]"]


def _op_eq(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    series, value = _comparable(df, condition.column, condition.value)
    return (series == value).fillna(False)


def _op_ne(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    series, value = _comparable(df, condition.column, condition.value)
    # A missing value is not "equal", so it survives a != filter.
    return (~(series == value)).fillna(True)


def _ordered(operator: str) -> MaskBuilder:
    comparisons = {
        "gt": lambda s, v: s > v,
        "gte": lambda s, v: s >= v,
        "lt": lambda s, v: s < v,
        "lte": lambda s, v: s <= v,
    }
    compare = comparisons[operator]

    def build(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
        series, value = _comparable(df, condition.column, condition.value)
        return compare(series, value).fillna(False)

    return build


def _op_between(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    low, high = _range_bounds(condition.value, "between")
    series = require_numeric_column(df, condition.column)
    low_value, high_value = _numeric_bounds(low, high, condition.column)
    return series.between(low_value, high_value, inclusive="both").fillna(False)


def _op_not_between(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    return ~_op_between(df, condition) & df[condition.column].notna()


def _numeric_bounds(low: Any, high: Any, column: str) -> tuple[float, float]:
    bounds = pd.to_numeric(pd.Series([low, high]), errors="coerce")
    if bounds.isna().any():
        raise InvalidParameterError(
            f"Range bounds for '{column}' must be numeric, got [{low!r}, {high!r}]."
        )
    low_value, high_value = float(bounds.iloc[0]), float(bounds.iloc[1])
    if low_value > high_value:
        low_value, high_value = high_value, low_value
    return low_value, high_value


def _op_date_between(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    start, end = _range_bounds(condition.value, "date_between")
    series = require_datetime_column(df, condition.column)
    bounds = pd.to_datetime(pd.Series([start, end]), errors="coerce")
    if bounds.isna().any():
        raise InvalidParameterError(
            f"Date bounds for '{condition.column}' could not be parsed: "
            f"[{start!r}, {end!r}]."
        )
    low, high = bounds.iloc[0], bounds.iloc[1]
    if low > high:
        low, high = high, low
    # An end date with no time component should include that whole day.
    if high.normalize() == high:
        high = high + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return series.between(low, high, inclusive="both").fillna(False)


def _op_in(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    values = _membership_values(condition.value, "in")
    series = df[condition.column]
    if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
        wanted = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        return series.isin(wanted.tolist())
    if pdt.is_datetime64_any_dtype(series):
        wanted = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
        return series.isin(wanted.tolist())
    normalized = {str(v).strip() for v in values}
    return series.astype("string").str.strip().isin(normalized).fillna(False)


def _op_not_in(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    return ~_op_in(df, condition)


def _op_is_null(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    return df[condition.column].isna()


def _op_not_null(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    return df[condition.column].notna()


_OPERATORS: dict[str, MaskBuilder] = {
    "eq": _op_eq,
    "ne": _op_ne,
    "gt": _ordered("gt"),
    "gte": _ordered("gte"),
    "lt": _ordered("lt"),
    "lte": _ordered("lte"),
    "between": _op_between,
    "not_between": _op_not_between,
    "date_between": _op_date_between,
    "in": _op_in,
    "not_in": _op_not_in,
    "is_null": _op_is_null,
    "not_null": _op_not_null,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_mask(df: pd.DataFrame, condition: Any) -> pd.Series:
    """Boolean mask for a single condition (exposed for testing and reuse)."""
    require_dataframe(df)
    resolved = FilterCondition.from_any(condition)
    column = require_column(df, resolved.column)
    operator = normalize_operator(resolved.operator)

    if operator not in NO_VALUE_OPERATORS and resolved.value is None:
        raise InvalidParameterError(
            f"The '{operator}' operator on '{column}' requires a value."
        )

    mask = _OPERATORS[operator](df, FilterCondition(column, operator, resolved.value))
    return mask.astype(bool).reindex(df.index, fill_value=False)


def filter_rows(
    df: pd.DataFrame,
    conditions: Any = None,
    *,
    logic: str = "and",
) -> pd.DataFrame:
    """Return a **new** dataframe containing the rows that match `conditions`.

    `conditions` may be a single condition or a list of them, each expressed as
    a dict (``{"column": ..., "operator": ..., "value": ...}``), a
    ``(column, operator, value)`` tuple, or a :class:`FilterCondition`.
    `logic` combines them with ``"and"`` (default) or ``"or"``.

    An empty or missing condition list returns an unaliased copy of `df`, and a
    filter that matches nothing returns an empty frame with the same columns --
    never an error.
    """
    require_dataframe(df)

    if logic not in ("and", "or"):
        raise InvalidParameterError(f"'logic' must be 'and' or 'or', got {logic!r}.")

    if conditions is None:
        return df.copy()
    if isinstance(conditions, (Mapping, FilterCondition)):
        conditions = [conditions]
    elif isinstance(conditions, tuple) and len(conditions) in (2, 3) and isinstance(
        conditions[0], str
    ):
        conditions = [conditions]
    elif not isinstance(conditions, Iterable):
        raise InvalidParameterError(
            f"'conditions' must be a condition or a list of conditions, got {conditions!r}."
        )

    resolved = list(conditions)
    if not resolved:
        return df.copy()

    masks = [build_mask(df, condition) for condition in resolved]
    combined = masks[0]
    for mask in masks[1:]:
        combined = (combined & mask) if logic == "and" else (combined | mask)

    return df.loc[combined].copy()


def describe_conditions(conditions: Any) -> list[str]:
    """Human-readable descriptions for a condition list."""
    if conditions is None:
        return []
    if isinstance(conditions, (Mapping, FilterCondition)):
        conditions = [conditions]
    return [FilterCondition.from_any(c).describe() for c in conditions]
