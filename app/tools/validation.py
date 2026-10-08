"""Shared input validation for the analysis tools.

Centralising these checks keeps every tool's error messages identical and makes
"handle missing columns gracefully" a single, tested code path.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd
from pandas.api import types as pdt

from app.models.profile import FieldType
from app.services.profiler import infer_field_type
from app.tools.exceptions import (
    ColumnNotFoundError,
    InvalidColumnTypeError,
    InvalidDataError,
    InvalidParameterError,
)

#: Share of values that must parse as dates before we treat a column as one.
DATE_PARSE_RATIO = 0.5


def require_dataframe(df: Any, *, allow_empty: bool = True) -> pd.DataFrame:
    """Validate that `df` is a usable dataframe and return it unchanged."""
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if not allow_empty and df.empty:
        raise InvalidDataError("The dataset is empty.")
    return df


def require_column(df: pd.DataFrame, column: Any) -> str:
    """Validate that `column` names a real column and return it as a string."""
    require_dataframe(df)
    if column is None or (isinstance(column, str) and not column.strip()):
        raise InvalidParameterError("A column name is required, but none was given.")
    if not isinstance(column, str):
        raise InvalidParameterError(
            f"Column names must be strings, got {type(column).__name__}: {column!r}."
        )
    if column not in df.columns:
        raise ColumnNotFoundError(column, df.columns)
    return column


def require_columns(df: pd.DataFrame, columns: Iterable[Any]) -> list[str]:
    """Validate several column names at once, preserving their order."""
    require_dataframe(df)
    if isinstance(columns, str):
        columns = [columns]
    resolved = [require_column(df, column) for column in columns]
    if not resolved:
        raise InvalidParameterError("At least one column is required.")
    return resolved


def require_numeric_column(df: pd.DataFrame, column: Any) -> pd.Series:
    """Return `column` as a numeric series, or explain why it cannot be one.

    Object columns holding numbers as text (a common CSV artefact) are coerced
    rather than rejected; a column that is genuinely non-numeric raises.
    """
    name = require_column(df, column)
    series = df[name]

    if pdt.is_bool_dtype(series):
        return series.astype("float64")
    if pdt.is_numeric_dtype(series):
        return series
    if pdt.is_datetime64_any_dtype(series):
        raise InvalidColumnTypeError(name, "numeric", f"a date column ({series.dtype})")

    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    if non_null and coerced.notna().sum() / non_null >= 0.5:
        return coerced

    raise InvalidColumnTypeError(name, "numeric", f"of type {series.dtype}")


def require_datetime_column(df: pd.DataFrame, column: Any) -> pd.Series:
    """Return `column` as a datetime series, coercing unparseable values to NaT."""
    name = require_column(df, column)
    series = df[name]

    if pdt.is_datetime64_any_dtype(series):
        return series
    if pdt.is_numeric_dtype(series) and infer_field_type(series, name) is not FieldType.DATETIME:
        # Bare numbers are too ambiguous to read as dates (see the profiler).
        raise InvalidColumnTypeError(
            name, "a date column", f"numeric ({series.dtype})"
        )

    converted = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null = int(series.notna().sum())
    if non_null == 0:
        raise InvalidColumnTypeError(name, "a date column", "entirely empty")
    if converted.notna().sum() / non_null < DATE_PARSE_RATIO:
        raise InvalidColumnTypeError(
            name,
            "a date column",
            f"of type {series.dtype} with no recognisable dates",
        )
    return converted


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns the tools are willing to treat as numeric measures."""
    require_dataframe(df)
    return [
        str(c)
        for c in df.columns
        if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame, *, max_unique: int | None = None) -> list[str]:
    """Columns suitable for grouping: categorical, boolean, or low-cardinality."""
    require_dataframe(df)
    out: list[str] = []
    for column in df.columns:
        series = df[column]
        field_type = infer_field_type(series, str(column))
        if field_type in (FieldType.CATEGORICAL, FieldType.BOOLEAN):
            if max_unique is None or series.nunique(dropna=True) <= max_unique:
                out.append(str(column))
    return out


def datetime_columns(df: pd.DataFrame) -> list[str]:
    """Columns the profiler recognises as dates."""
    require_dataframe(df)
    return [
        str(c)
        for c in df.columns
        if infer_field_type(df[c], str(c)) is FieldType.DATETIME
    ]


def require_positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    """Validate a positive integer parameter such as ``top_n`` or ``bins``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParameterError(
            f"'{name}' must be a positive integer, got {value!r}."
        )
    as_int = int(value)
    if as_int != value or as_int < 1:
        raise InvalidParameterError(
            f"'{name}' must be a positive integer, got {value!r}."
        )
    if maximum is not None and as_int > maximum:
        raise InvalidParameterError(
            f"'{name}' must not exceed {maximum}, got {as_int}."
        )
    return as_int


def as_sequence(value: Any, *, name: str) -> list[Any]:
    """Normalise a scalar-or-iterable parameter into a list."""
    if value is None:
        raise InvalidParameterError(f"'{name}' is required.")
    if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, set, frozenset)):
        return [value]
    return list(value)
