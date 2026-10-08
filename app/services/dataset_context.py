"""The compact dataset description sent to the LLM.

The model needs to know what it can analyse; it does not need the data. This
module builds a small, bounded summary — column names, inferred roles, ranges,
a few category values — and nothing else. **No dataframe rows are ever sent.**

Everything here is size-capped so the prompt stays roughly constant whether the
dataset has 500 rows or 500,000.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.models.profile import DatasetProfile, FieldType
from app.tools.analysis import get_dataset_schema
from app.tools.validation import require_dataframe

#: Distinct values listed per categorical column.
MAX_CATEGORY_VALUES = 8
#: Columns described in full before the rest are name-only.
MAX_DESCRIBED_COLUMNS = 40
#: Characters a single category value is truncated to.
MAX_VALUE_LENGTH = 40


@dataclass
class ColumnContext:
    """What the model is told about one column."""

    name: str
    type: str
    missing_pct: float = 0.0
    #: Numeric columns: observed range and centre.
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    #: Categorical columns: how many distinct values, and some of them.
    distinct: int | None = None
    values: list[str] = field(default_factory=list)
    #: Date columns: the covered span.
    earliest: str | None = None
    latest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.missing_pct >= 0.5:
            payload["missing_pct"] = round(self.missing_pct, 1)
        if self.minimum is not None:
            payload["range"] = [_compact(self.minimum), _compact(self.maximum)]
            if self.mean is not None:
                payload["mean"] = _compact(self.mean)
        if self.distinct is not None:
            payload["distinct"] = self.distinct
            if self.values:
                payload["examples"] = self.values
        if self.earliest:
            payload["covers"] = [self.earliest, self.latest]
        return payload


@dataclass
class DatasetContext:
    """The whole prompt-ready dataset description."""

    name: str
    row_count: int
    column_count: int
    columns: list[ColumnContext] = field(default_factory=list)
    #: Column names grouped by analytical role.
    roles: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "rows": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
            "roles": {role: names for role, names in self.roles.items() if names},
            "notes": self.notes,
        }

    def to_prompt(self) -> str:
        """Compact JSON for embedding in a system prompt."""
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    def approx_tokens(self) -> int:
        """Rough token estimate, for the UI's transparency panel."""
        return max(1, len(self.to_prompt()) // 4)


def _compact(value: float | None) -> float | None:
    """Round a figure to something readable in a prompt."""
    if value is None:
        return None
    number = float(value)
    if abs(number) >= 1000:
        return round(number, 0)
    if abs(number) >= 10:
        return round(number, 1)
    return round(number, 3)


def build_dataset_context(
    df: pd.DataFrame,
    profile: DatasetProfile | None = None,
    *,
    name: str = "dataset",
    max_category_values: int = MAX_CATEGORY_VALUES,
) -> DatasetContext:
    """Summarise `df` for the LLM, reusing `profile` when one already exists.

    The Overview page has already profiled the dataset, so passing that profile
    in avoids recomputing statistics purely to build a prompt.
    """
    require_dataframe(df)
    schema = get_dataset_schema(df, name=name)

    context = DatasetContext(
        name=name,
        row_count=schema.row_count,
        column_count=schema.column_count,
        roles={
            "numeric": schema.numeric_columns,
            "categorical": schema.categorical_columns,
            "boolean": schema.boolean_columns,
            "date": schema.date_columns,
            "identifier": schema.identifier_columns,
        },
    )

    by_name = {c.name: c for c in (profile.columns if profile else [])}

    for index, column in enumerate(schema.columns):
        if index >= MAX_DESCRIBED_COLUMNS:
            remaining = schema.column_count - MAX_DESCRIBED_COLUMNS
            context.notes.append(
                f"{remaining} further column(s) omitted from this description."
            )
            break

        entry = ColumnContext(
            name=column.name,
            type=column.inferred_type,
            missing_pct=column.missing_pct,
        )
        detail = by_name.get(column.name)
        field_type = column.inferred_type

        if field_type in (FieldType.NUMERIC.value, FieldType.INTEGER.value):
            if detail is not None and detail.min is not None:
                entry.minimum, entry.maximum, entry.mean = (
                    detail.min, detail.max, detail.mean
                )
            else:
                numeric = pd.to_numeric(df[column.name], errors="coerce").dropna()
                if not numeric.empty:
                    entry.minimum = float(numeric.min())
                    entry.maximum = float(numeric.max())
                    entry.mean = float(numeric.mean())

        elif field_type == FieldType.DATETIME.value:
            if detail is not None and detail.min_date:
                entry.earliest, entry.latest = detail.min_date, detail.max_date
            else:
                dates = pd.to_datetime(
                    df[column.name], errors="coerce", format="mixed"
                ).dropna()
                if not dates.empty:
                    entry.earliest = str(dates.min().date())
                    entry.latest = str(dates.max().date())

        elif field_type in (
            FieldType.CATEGORICAL.value,
            FieldType.BOOLEAN.value,
            FieldType.TEXT.value,
        ):
            entry.distinct = column.unique_count
            # Only list values when there are few enough to be informative;
            # a free-text column's values are noise in a prompt.
            if column.unique_count <= max_category_values * 6:
                if detail is not None and detail.top_values:
                    sampled = list(detail.top_values)[:max_category_values]
                else:
                    sampled = [
                        str(v)
                        for v in df[column.name].dropna().astype(str)
                        .value_counts().head(max_category_values).index
                    ]
                entry.values = [_truncate(v) for v in sampled]

        elif field_type == FieldType.IDENTIFIER.value:
            entry.distinct = column.unique_count

        context.columns.append(entry)

    if profile is not None:
        if profile.duplicate_row_count:
            context.notes.append(
                f"{profile.duplicate_row_count} duplicate row(s) present."
            )
        if profile.missing_cells_pct >= 1:
            context.notes.append(
                f"{profile.missing_cells_pct:.1f}% of all cells are missing."
            )
        if profile.truncated and profile.original_row_count:
            context.notes.append(
                f"Only the first {profile.row_count} of "
                f"{profile.original_row_count} rows were loaded."
            )

    return context


def _truncate(value: str) -> str:
    text = str(value)
    return text if len(text) <= MAX_VALUE_LENGTH else text[: MAX_VALUE_LENGTH - 1] + "…"


__all__ = [
    "ColumnContext",
    "DatasetContext",
    "MAX_CATEGORY_VALUES",
    "build_dataset_context",
]
