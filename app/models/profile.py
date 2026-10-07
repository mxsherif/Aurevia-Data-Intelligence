"""Structured representation of a profiled dataset.

These are plain dataclasses on purpose: they are produced by the profiler,
consumed by the UI, and (in a later phase) serialised into agent prompts, so
cheap construction and `to_dict()` round-tripping matter more than validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FieldType(str, Enum):
    """Semantic type inferred for a column (richer than the pandas dtype)."""

    NUMERIC = "numeric"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    EMPTY = "empty"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # nicer rendering in tables
        return self.value


class WarningSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value


@dataclass
class DataQualityWarning:
    """A single, human-readable data-quality finding."""

    code: str
    severity: WarningSeverity
    message: str
    columns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ColumnProfile:
    """Everything known about a single column."""

    name: str
    dtype: str
    inferred_type: FieldType

    count: int = 0                     # non-null values
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    unique_pct: float = 0.0
    memory_bytes: int = 0

    is_constant: bool = False
    is_low_cardinality: bool = False
    is_likely_id: bool = False

    # Numeric-only statistics (None for non-numeric columns).
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q1: float | None = None
    q3: float | None = None
    skew: float | None = None
    zeros: int = 0
    negatives: int = 0

    # Outliers (IQR rule, numeric columns only).
    outlier_count: int = 0
    outlier_pct: float = 0.0
    outlier_lower_bound: float | None = None
    outlier_upper_bound: float | None = None

    # Datetime-only.
    min_date: str | None = None
    max_date: str | None = None
    date_range_days: int | None = None

    # Categorical / text.
    top_values: dict[str, int] = field(default_factory=dict)
    sample_values: list[Any] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.inferred_type in (FieldType.NUMERIC, FieldType.INTEGER)

    @property
    def is_datetime(self) -> bool:
        return self.inferred_type is FieldType.DATETIME

    @property
    def is_categorical(self) -> bool:
        return self.inferred_type in (FieldType.CATEGORICAL, FieldType.TEXT)

    @property
    def is_boolean(self) -> bool:
        return self.inferred_type is FieldType.BOOLEAN

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["inferred_type"] = self.inferred_type.value
        return data


@dataclass
class DatasetProfile:
    """Aggregate profile of a dataframe."""

    name: str = "dataset"
    source: str = "unknown"

    row_count: int = 0
    column_count: int = 0
    memory_usage_bytes: int = 0

    duplicate_row_count: int = 0
    duplicate_row_pct: float = 0.0

    total_cells: int = 0
    missing_cells: int = 0
    missing_cells_pct: float = 0.0

    columns: list[ColumnProfile] = field(default_factory=list)
    warnings: list[DataQualityWarning] = field(default_factory=list)

    # Pairs of strongly correlated numeric columns:
    # [{"left": "...", "right": "...", "correlation": 0.93}, ...]
    correlations: list[dict[str, Any]] = field(default_factory=list)

    truncated: bool = False
    original_row_count: int | None = None
    notes: list[str] = field(default_factory=list)

    # -- convenience accessors -------------------------------------------- #

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def _names_where(self, predicate) -> list[str]:
        return [c.name for c in self.columns if predicate(c)]

    @property
    def numeric_columns(self) -> list[str]:
        return self._names_where(lambda c: c.is_numeric)

    @property
    def categorical_columns(self) -> list[str]:
        return self._names_where(lambda c: c.is_categorical)

    @property
    def boolean_columns(self) -> list[str]:
        return self._names_where(lambda c: c.is_boolean)

    @property
    def date_columns(self) -> list[str]:
        return self._names_where(lambda c: c.is_datetime)

    @property
    def id_columns(self) -> list[str]:
        return self._names_where(lambda c: c.is_likely_id)

    @property
    def low_cardinality_columns(self) -> list[str]:
        return self._names_where(lambda c: c.is_low_cardinality)

    @property
    def columns_with_missing(self) -> list[str]:
        return self._names_where(lambda c: c.missing_count > 0)

    @property
    def memory_usage_mb(self) -> float:
        return self.memory_usage_bytes / (1024 * 1024)

    def column(self, name: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def type_map(self) -> dict[str, str]:
        return {c.name: c.inferred_type.value for c in self.columns}

    def warnings_by_severity(self, severity: WarningSeverity) -> list[DataQualityWarning]:
        return [w for w in self.warnings if w.severity is severity]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_usage_bytes": self.memory_usage_bytes,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_pct": self.duplicate_row_pct,
            "total_cells": self.total_cells,
            "missing_cells": self.missing_cells,
            "missing_cells_pct": self.missing_cells_pct,
            "truncated": self.truncated,
            "original_row_count": self.original_row_count,
            "notes": list(self.notes),
            "columns": [c.to_dict() for c in self.columns],
            "warnings": [w.to_dict() for w in self.warnings],
            "correlations": list(self.correlations),
            "summary": {
                "numeric_columns": self.numeric_columns,
                "categorical_columns": self.categorical_columns,
                "boolean_columns": self.boolean_columns,
                "date_columns": self.date_columns,
                "id_columns": self.id_columns,
                "low_cardinality_columns": self.low_cardinality_columns,
            },
        }
