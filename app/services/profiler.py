"""Dataset profiler.

Turns a dataframe into a :class:`DatasetProfile`: shape and memory, per-column
semantic types and statistics, missing values, duplicates, outliers,
correlations, and a list of data-quality warnings.

The profiler is deliberately defensive -- a single pathological column must not
prevent the rest of the dataset from being profiled.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from app.models.profile import (
    ColumnProfile,
    DataQualityWarning,
    DatasetProfile,
    FieldType,
    WarningSeverity,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunable thresholds
# --------------------------------------------------------------------------- #

#: A column with <= this many distinct values is "low cardinality".
LOW_CARDINALITY_MAX = 20
#: ...or whose distinct ratio is below this (used for larger datasets).
LOW_CARDINALITY_RATIO = 0.05
#: Object columns at or below this distinct ratio are categorical, not free text.
CATEGORICAL_RATIO = 0.5
#: Share of distinct values above which a column is considered an identifier.
ID_UNIQUE_RATIO = 0.95
#: Multiplier for the IQR outlier fence.
IQR_MULTIPLIER = 1.5
#: Missing-value share that triggers a warning / a critical warning.
MISSING_WARN_PCT = 5.0
MISSING_CRITICAL_PCT = 40.0
#: |r| above which a numeric pair is reported as correlated.
CORRELATION_THRESHOLD = 0.75
#: Rows sampled when attempting to parse an object column as dates.
DATE_SNIFF_SAMPLE = 200
#: Share of the sample that must parse as a date for the column to be a date.
DATE_SNIFF_RATIO = 0.9
#: Most frequent values kept per categorical column.
TOP_VALUES = 10
#: Max numeric columns fed to the correlation matrix (it is O(n^2)).
MAX_CORRELATION_COLUMNS = 40

_BOOLEAN_TOKENS = (
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"t", "f"},
    {"0", "1"},
    {"on", "off"},
)

_ID_NAME_PATTERN = re.compile(r"(^|[_\s])(id|uuid|guid|key|code|number|no|msisdn)$", re.I)
_DATE_NAME_PATTERN = re.compile(r"(date|time|timestamp|_at$|_on$|month|year|day)", re.I)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _safe_float(value: Any) -> float | None:
    """Coerce to a plain float, mapping NaN/inf/non-numeric to ``None``."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else out


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def _safe_nunique(series: pd.Series) -> int:
    """``nunique`` that survives unhashable cell values (lists, dicts, sets)."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def _safe_unique_set(series: pd.Series) -> set | None:
    """Distinct values, or ``None`` when they cannot be hashed."""
    try:
        return set(series.unique())
    except TypeError:
        return None


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

def _looks_boolean(series: pd.Series) -> bool:
    """True for 0/1, yes/no, true/false style columns."""
    values = series.dropna()
    if values.empty:
        return False

    if pdt.is_bool_dtype(values):
        return True

    distinct = _safe_unique_set(values)
    if distinct is None or len(distinct) > 2:
        return False

    if pdt.is_numeric_dtype(values):
        return distinct.issubset({0, 1}) and len(distinct) == 2

    tokens = {str(v).strip().lower() for v in distinct}
    return any(tokens and tokens.issubset(valid) for valid in _BOOLEAN_TOKENS)


def _looks_datetime(series: pd.Series, column_name: str = "") -> bool:
    """Try to recognise date-like object columns without being trigger-happy."""
    values = series.dropna()
    if values.empty:
        return False
    if pdt.is_datetime64_any_dtype(values):
        return True
    if pdt.is_numeric_dtype(values) or pdt.is_bool_dtype(values):
        # Bare numbers are years/epochs at best -- too ambiguous to claim.
        return False
    if not pdt.is_object_dtype(values) and not isinstance(values.dtype, pd.StringDtype):
        return False

    try:
        sample = values.head(DATE_SNIFF_SAMPLE).astype(str)
    except (TypeError, ValueError):
        return False
    # Require some date-ish punctuation so "A1", "12" and free text are excluded.
    looks_shaped = sample.str.contains(r"[-/:]|\d{4}", regex=True, na=False)
    if looks_shaped.mean() < DATE_SNIFF_RATIO:
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    ratio = float(parsed.notna().mean())
    if ratio >= DATE_SNIFF_RATIO:
        return True
    # A date-sounding name lowers the bar slightly, but never to zero.
    return bool(_DATE_NAME_PATTERN.search(column_name)) and ratio >= 0.6


def infer_field_type(series: pd.Series, column_name: str | None = None) -> FieldType:
    """Infer the semantic :class:`FieldType` of a single column."""
    name = column_name if column_name is not None else str(series.name or "")
    values = series.dropna()

    if values.empty:
        return FieldType.EMPTY

    if _looks_boolean(values):
        return FieldType.BOOLEAN

    if _looks_datetime(values, name):
        return FieldType.DATETIME

    if pdt.is_integer_dtype(values):
        if _is_identifier(values, name, integer_like=True):
            return FieldType.IDENTIFIER
        return FieldType.INTEGER

    if pdt.is_numeric_dtype(values):
        return FieldType.NUMERIC

    if isinstance(values.dtype, pd.CategoricalDtype):
        return FieldType.CATEGORICAL

    if pdt.is_object_dtype(values) or isinstance(values.dtype, pd.StringDtype):
        if _is_identifier(values, name, integer_like=False):
            return FieldType.IDENTIFIER
        distinct = _safe_nunique(values)
        if distinct / len(values) <= CATEGORICAL_RATIO or distinct <= LOW_CARDINALITY_MAX:
            return FieldType.CATEGORICAL
        return FieldType.TEXT

    return FieldType.UNKNOWN


def _is_identifier(values: pd.Series, name: str, *, integer_like: bool) -> bool:
    """Near-unique values plus an id-ish name (or textual near-uniqueness)."""
    if len(values) < 2:
        return False
    unique_ratio = _safe_nunique(values) / len(values)
    if unique_ratio < ID_UNIQUE_RATIO:
        return False
    if _ID_NAME_PATTERN.search(name):
        return True
    # A near-unique *numeric* column is often a real measure (e.g. revenue), so
    # only treat strings as implicit identifiers.
    return not integer_like


# --------------------------------------------------------------------------- #
# Per-column profiling
# --------------------------------------------------------------------------- #

def _profile_numeric(series: pd.Series, profile: ColumnProfile) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return

    profile.min = _safe_float(values.min())
    profile.max = _safe_float(values.max())
    profile.mean = _safe_float(values.mean())
    profile.median = _safe_float(values.median())
    profile.std = _safe_float(values.std()) if len(values) > 1 else 0.0
    profile.q1 = _safe_float(values.quantile(0.25))
    profile.q3 = _safe_float(values.quantile(0.75))
    profile.skew = _safe_float(values.skew()) if len(values) > 2 else None
    profile.zeros = int((values == 0).sum())
    profile.negatives = int((values < 0).sum())

    if profile.q1 is None or profile.q3 is None:
        return
    iqr = profile.q3 - profile.q1
    if iqr <= 0:
        # Degenerate spread: fall back to a 3-sigma rule when possible.
        if not profile.std or profile.mean is None:
            return
        lower = profile.mean - 3 * profile.std
        upper = profile.mean + 3 * profile.std
    else:
        lower = profile.q1 - IQR_MULTIPLIER * iqr
        upper = profile.q3 + IQR_MULTIPLIER * iqr

    mask = (values < lower) | (values > upper)
    profile.outlier_lower_bound = _safe_float(lower)
    profile.outlier_upper_bound = _safe_float(upper)
    profile.outlier_count = int(mask.sum())
    profile.outlier_pct = _pct(profile.outlier_count, len(values))


def _profile_datetime(series: pd.Series, profile: ColumnProfile) -> None:
    if pdt.is_datetime64_any_dtype(series):
        values = series.dropna()
    else:
        values = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
    if values.empty:
        return

    low, high = values.min(), values.max()
    profile.min_date = str(pd.Timestamp(low).date())
    profile.max_date = str(pd.Timestamp(high).date())
    profile.date_range_days = int((pd.Timestamp(high) - pd.Timestamp(low)).days)


def _profile_categorical(series: pd.Series, profile: ColumnProfile) -> None:
    values = series.dropna()
    if values.empty:
        return
    counts = values.astype(str).value_counts().head(TOP_VALUES)
    profile.top_values = {str(k): int(v) for k, v in counts.items()}


def profile_column(series: pd.Series, name: str | None = None) -> ColumnProfile:
    """Profile one column; never raises."""
    column_name = name if name is not None else str(series.name)
    total = len(series)
    non_null = int(series.notna().sum())
    missing = total - non_null

    unique = _safe_nunique(series)

    try:
        memory = int(series.memory_usage(deep=True))
    except (TypeError, ValueError):
        memory = int(series.memory_usage(deep=False))

    inferred = infer_field_type(series, column_name)

    profile = ColumnProfile(
        name=column_name,
        dtype=str(series.dtype),
        inferred_type=inferred,
        count=non_null,
        missing_count=missing,
        missing_pct=_pct(missing, total),
        unique_count=unique,
        unique_pct=_pct(unique, non_null),
        memory_bytes=memory,
        is_constant=non_null > 0 and unique <= 1,
        is_likely_id=inferred is FieldType.IDENTIFIER,
    )

    profile.is_low_cardinality = bool(
        non_null > 0
        and inferred not in (FieldType.IDENTIFIER, FieldType.EMPTY)
        and (unique <= LOW_CARDINALITY_MAX or unique / max(non_null, 1) <= LOW_CARDINALITY_RATIO)
    )

    sample = series.dropna().head(5).tolist()
    profile.sample_values = [str(v) for v in sample]

    try:
        if inferred in (FieldType.NUMERIC, FieldType.INTEGER):
            _profile_numeric(series, profile)
        elif inferred is FieldType.DATETIME:
            _profile_datetime(series, profile)
        elif inferred in (FieldType.CATEGORICAL, FieldType.TEXT, FieldType.BOOLEAN):
            _profile_categorical(series, profile)
        elif inferred is FieldType.IDENTIFIER and pdt.is_numeric_dtype(series):
            _profile_numeric(series, profile)
    except Exception:  # noqa: BLE001 - one bad column must not kill the profile
        logger.exception("Failed to profile column %r in detail", column_name)

    return profile


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #

def compute_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    *,
    threshold: float = CORRELATION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Strongly correlated numeric pairs, strongest first."""
    usable = [c for c in numeric_columns if c in df.columns][:MAX_CORRELATION_COLUMNS]
    if len(usable) < 2:
        return []

    try:
        matrix = df[usable].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    except Exception:  # noqa: BLE001
        logger.exception("Correlation computation failed")
        return []

    pairs: list[dict[str, Any]] = []
    columns = list(matrix.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = _safe_float(matrix.loc[left, right])
            if value is None or abs(value) < threshold:
                continue
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": round(value, 4),
                    "direction": "positive" if value > 0 else "negative",
                }
            )

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs


# --------------------------------------------------------------------------- #
# Data-quality warnings
# --------------------------------------------------------------------------- #

def _warn(code: str, severity: WarningSeverity, message: str, columns=None, **details):
    return DataQualityWarning(
        code=code,
        severity=severity,
        message=message,
        columns=list(columns or []),
        details=details,
    )


def build_warnings(profile: DatasetProfile) -> list[DataQualityWarning]:
    """Derive human-readable data-quality findings from a profile."""
    warnings: list[DataQualityWarning] = []

    # -- dataset level ----------------------------------------------------- #
    if profile.row_count == 0:
        warnings.append(
            _warn("empty_dataset", WarningSeverity.CRITICAL, "The dataset has no rows.")
        )
        return warnings

    if profile.row_count < 30:
        warnings.append(
            _warn(
                "very_small_dataset",
                WarningSeverity.INFO,
                f"Only {profile.row_count} rows -- statistics and models will be unreliable.",
                row_count=profile.row_count,
            )
        )

    if profile.duplicate_row_count:
        severity = (
            WarningSeverity.CRITICAL
            if profile.duplicate_row_pct > 10
            else WarningSeverity.WARNING
        )
        warnings.append(
            _warn(
                "duplicate_rows",
                severity,
                f"{profile.duplicate_row_count:,} duplicate row(s) "
                f"({profile.duplicate_row_pct:.2f}% of the dataset).",
                count=profile.duplicate_row_count,
                pct=profile.duplicate_row_pct,
            )
        )

    if profile.missing_cells_pct >= MISSING_WARN_PCT:
        warnings.append(
            _warn(
                "high_overall_missing",
                WarningSeverity.WARNING,
                f"{profile.missing_cells_pct:.2f}% of all cells are missing.",
                pct=profile.missing_cells_pct,
            )
        )

    if profile.truncated and profile.original_row_count:
        warnings.append(
            _warn(
                "dataset_truncated",
                WarningSeverity.INFO,
                f"Only the first {profile.row_count:,} of "
                f"{profile.original_row_count:,} rows were profiled.",
                loaded=profile.row_count,
                original=profile.original_row_count,
            )
        )

    # -- column level ------------------------------------------------------ #
    empty_columns = [c.name for c in profile.columns if c.count == 0]
    if empty_columns:
        warnings.append(
            _warn(
                "empty_columns",
                WarningSeverity.CRITICAL,
                f"{len(empty_columns)} column(s) are completely empty.",
                empty_columns,
            )
        )

    critical_missing = [
        c.name
        for c in profile.columns
        if c.count > 0 and c.missing_pct >= MISSING_CRITICAL_PCT
    ]
    if critical_missing:
        warnings.append(
            _warn(
                "columns_mostly_missing",
                WarningSeverity.CRITICAL,
                f"{len(critical_missing)} column(s) are missing at least "
                f"{MISSING_CRITICAL_PCT:.0f}% of their values.",
                critical_missing,
            )
        )

    moderate_missing = [
        c.name
        for c in profile.columns
        if c.count > 0 and MISSING_WARN_PCT <= c.missing_pct < MISSING_CRITICAL_PCT
    ]
    if moderate_missing:
        warnings.append(
            _warn(
                "columns_with_missing",
                WarningSeverity.WARNING,
                f"{len(moderate_missing)} column(s) have more than "
                f"{MISSING_WARN_PCT:.0f}% missing values.",
                moderate_missing,
            )
        )

    constant = [c.name for c in profile.columns if c.is_constant]
    if constant:
        warnings.append(
            _warn(
                "constant_columns",
                WarningSeverity.WARNING,
                f"{len(constant)} column(s) hold a single repeated value and carry no signal.",
                constant,
            )
        )

    outlier_heavy = [c.name for c in profile.columns if c.outlier_pct >= 1.0]
    if outlier_heavy:
        warnings.append(
            _warn(
                "outliers_detected",
                WarningSeverity.INFO,
                f"{len(outlier_heavy)} numeric column(s) contain possible outliers "
                "beyond the 1.5xIQR fence.",
                outlier_heavy,
                counts={c: profile.column(c).outlier_count for c in outlier_heavy},
            )
        )

    negative_in_positive_field = [
        c.name
        for c in profile.columns
        if c.is_numeric
        and c.negatives > 0
        and re.search(r"(charge|revenue|price|amount|cost|usage|count|calls|tenure)", c.name, re.I)
    ]
    if negative_in_positive_field:
        warnings.append(
            _warn(
                "unexpected_negative_values",
                WarningSeverity.WARNING,
                f"{len(negative_in_positive_field)} column(s) contain negative values "
                "where only non-negative values are expected.",
                negative_in_positive_field,
            )
        )

    high_cardinality_text = [
        c.name
        for c in profile.columns
        if c.inferred_type is FieldType.TEXT and c.unique_pct > 80
    ]
    if high_cardinality_text:
        warnings.append(
            _warn(
                "high_cardinality_text",
                WarningSeverity.INFO,
                f"{len(high_cardinality_text)} free-text column(s) are nearly all distinct "
                "and may need encoding before modelling.",
                high_cardinality_text,
            )
        )

    if not profile.date_columns:
        warnings.append(
            _warn(
                "no_date_column",
                WarningSeverity.INFO,
                "No date column was detected -- time-series analysis will not be available.",
            )
        )

    if profile.correlations:
        top = profile.correlations[0]
        warnings.append(
            _warn(
                "correlated_columns",
                WarningSeverity.INFO,
                f"{len(profile.correlations)} strongly correlated numeric pair(s), "
                f"strongest: {top['left']} vs {top['right']} (r={top['correlation']}).",
                [top["left"], top["right"]],
                pairs=profile.correlations[:5],
            )
        )

    order = {
        WarningSeverity.CRITICAL: 0,
        WarningSeverity.WARNING: 1,
        WarningSeverity.INFO: 2,
    }
    warnings.sort(key=lambda w: order[w.severity])
    return warnings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def profile_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "dataset",
    source: str = "unknown",
    truncated: bool = False,
    original_row_count: int | None = None,
    notes: list[str] | None = None,
) -> DatasetProfile:
    """Build a full :class:`DatasetProfile` for `df`.

    An empty dataframe yields a valid (mostly zeroed) profile rather than an
    error, so callers can always render something.
    """
    if df is None:
        raise ValueError("profile_dataframe() requires a DataFrame, got None")

    rows, cols = df.shape

    try:
        memory = int(df.memory_usage(deep=True).sum())
    except (TypeError, ValueError):
        memory = int(df.memory_usage(deep=False).sum())

    profile = DatasetProfile(
        name=name,
        source=source,
        row_count=int(rows),
        column_count=int(cols),
        memory_usage_bytes=memory,
        truncated=truncated,
        original_row_count=original_row_count,
        notes=list(notes or []),
    )

    if cols:
        profile.columns = [profile_column(df[c], str(c)) for c in df.columns]

    profile.total_cells = int(rows * cols)
    profile.missing_cells = sum(c.missing_count for c in profile.columns)
    profile.missing_cells_pct = _pct(profile.missing_cells, profile.total_cells)

    if rows:
        try:
            duplicates = int(df.duplicated().sum())
        except TypeError:  # unhashable cells
            duplicates = int(df.astype(str).duplicated().sum())
        profile.duplicate_row_count = duplicates
        profile.duplicate_row_pct = _pct(duplicates, rows)

    profile.correlations = compute_correlations(df, profile.numeric_columns)
    profile.warnings = build_warnings(profile)

    logger.info(
        "Profiled %s: %d rows, %d columns, %d warning(s)",
        name, rows, cols, len(profile.warnings),
    )
    return profile


def descriptive_statistics(df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """A tidy ``describe()``-style table for the numeric columns."""
    numeric = profile.numeric_columns
    if not numeric:
        return pd.DataFrame()
    stats = df[numeric].apply(pd.to_numeric, errors="coerce").describe().T
    return stats.round(4)
