"""The analytical plan: Aurevia's contract between the LLM and Python.

The planner agent emits an :class:`AnalysisPlan` as OpenAI structured output,
so the model fills a validated schema instead of prose we have to parse. The
plan says *what analysis to run*; it never carries results. Every field is
re-checked against the real dataset by :mod:`app.services.plan_validator`
before anything executes.

Schema notes for structured output: the fields are deliberately flat and
typed as strings or lists of strings, because a free-form ``Any`` value cannot
be expressed in a strict JSON schema. Coercion to real types happens in
Python, against the actual column dtype.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    """The analytical question types Phase 3 supports.

    Deliberately a closed set: the executor has one deterministic handler per
    intent, so an unknown intent is a planning error rather than something to
    improvise around.
    """

    SUMMARY = "summary"
    RANKING = "ranking"
    COMPARISON = "comparison"
    SEGMENTATION = "segmentation"
    TREND = "trend"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    TIME_COMPARISON = "time_comparison"
    PERCENTAGE_CHANGE = "percentage_change"
    COUNT = "count"
    ANOMALY = "anomaly"
    DATASET_QUESTION = "dataset_question"

    def __str__(self) -> str:
        return self.value


#: Human-readable labels for the plan display.
INTENT_LABELS: dict[Intent, str] = {
    Intent.SUMMARY: "Summary statistics",
    Intent.RANKING: "Ranking",
    Intent.COMPARISON: "Group comparison",
    Intent.SEGMENTATION: "Segment breakdown",
    Intent.TREND: "Trend over time",
    Intent.DISTRIBUTION: "Distribution",
    Intent.CORRELATION: "Correlation",
    Intent.TIME_COMPARISON: "Period-over-period comparison",
    Intent.PERCENTAGE_CHANGE: "Percentage change",
    Intent.COUNT: "Counts",
    Intent.ANOMALY: "Outlier inquiry",
    Intent.DATASET_QUESTION: "Dataset question",
}

#: Intents whose answer is inherently about time.
TIME_INTENTS = frozenset(
    {Intent.TREND, Intent.TIME_COMPARISON, Intent.PERCENTAGE_CHANGE}
)

#: Intents that need a numeric measure to compute anything.
METRIC_INTENTS = frozenset(
    {
        Intent.SUMMARY,
        Intent.CORRELATION,
        Intent.DISTRIBUTION,
        Intent.ANOMALY,
    }
)


class SortDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"

    def __str__(self) -> str:
        return self.value


class PlanFilter(BaseModel):
    """One filter condition as the planner expresses it.

    Values arrive as strings because the model cannot know a column's dtype;
    :meth:`to_condition` hands them to the Phase 2 filter layer, which coerces
    each value against the real column and raises if that is impossible.
    """

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Dataset column to filter on.")
    operator: str = Field(
        description=(
            "One of: eq, ne, gt, gte, lt, lte, between, not_between, "
            "date_between, in, not_in, is_null, not_null."
        )
    )
    value: str | None = Field(
        default=None,
        description="Single comparison value, for the scalar operators.",
    )
    values: list[str] = Field(
        default_factory=list,
        description=(
            "Value list for 'in'/'not_in', or exactly two bounds "
            "[low, high] for 'between' and 'date_between'."
        ),
    )

    def to_condition(self) -> dict[str, Any]:
        """Convert to the dict shape :func:`app.tools.filter_rows` accepts."""
        operator = (self.operator or "").strip().lower()
        if operator in {"between", "not_between", "date_between"}:
            payload: Any = list(self.values)
        elif operator in {"in", "not_in"}:
            payload = list(self.values) if self.values else self.value
        elif operator in {"is_null", "not_null"}:
            payload = None
        else:
            payload = self.value if self.value is not None else (
                self.values[0] if self.values else None
            )
        return {"column": self.column, "operator": self.operator, "value": payload}

    def describe(self) -> str:
        """Short human-readable rendering for the plan display."""
        operator = (self.operator or "").strip().lower()
        if operator in {"is_null", "not_null"}:
            return f"{self.column} {operator.replace('_', ' ')}"
        if operator in {"between", "not_between", "date_between"} and len(self.values) == 2:
            return f"{self.column} between {self.values[0]} and {self.values[1]}"
        if operator in {"in", "not_in"}:
            shown = ", ".join(self.values[:4])
            more = f", +{len(self.values) - 4} more" if len(self.values) > 4 else ""
            verb = "is one of" if operator == "in" else "is not one of"
            return f"{self.column} {verb} [{shown}{more}]"
        symbols = {"eq": "=", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
        return f"{self.column} {symbols.get(operator, operator)} {self.value}"


class AnalysisPlan(BaseModel):
    """A validated, executable description of the analysis a question needs."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    intent: Intent = Field(description="The kind of analysis required.")

    metric: str | None = Field(
        default=None,
        description=(
            "Numeric column to measure (e.g. revenue). Null when the question "
            "is about row counts or about the dataset itself."
        ),
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Categorical columns to group or break down by.",
    )
    filters: list[PlanFilter] = Field(
        default_factory=list,
        description="Row filters to apply before aggregating.",
    )
    aggregation: str | None = Field(
        default=None,
        description=(
            "How to reduce the metric: sum, mean, median, count, min, max, "
            "std, or percentage_change."
        ),
    )

    time_column: str | None = Field(
        default=None, description="Date column for time-based analysis."
    )
    time_granularity: str | None = Field(
        default=None,
        description="daily, weekly, monthly, quarterly or yearly.",
    )
    periods: int | None = Field(
        default=None,
        description=(
            "How many of the most recent periods to consider, when the "
            "question scopes a window such as 'the last six months'."
        ),
    )

    comparison: str | None = Field(
        default=None,
        description=(
            "Comparison style, when one applies: percentage_change, "
            "absolute_change, or vs_overall."
        ),
    )
    sort_direction: SortDirection | None = Field(
        default=None,
        description="descending for largest-first, ascending for smallest-first.",
    )
    limit: int | None = Field(
        default=None, description="How many rows or categories to return."
    )
    visualization: str | None = Field(
        default=None,
        description=(
            "Suggested chart: line, bar, scatter, histogram, box, heatmap, "
            "or none when no chart would help."
        ),
    )

    steps: list[str] = Field(
        default_factory=list,
        description=(
            "Short user-facing execution summary, one line per step. This is "
            "shown to the user; it is not private reasoning."
        ),
    )

    requires_clarification: bool = Field(
        default=False,
        description=(
            "True only when the question is genuinely ambiguous and no "
            "sensible default exists."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description="One concise question to resolve the ambiguity.",
    )

    # -- convenience ------------------------------------------------------- #

    @property
    def primary_dimension(self) -> str | None:
        return self.dimensions[0] if self.dimensions else None

    @property
    def is_time_based(self) -> bool:
        return self.time_column is not None or self.intent in TIME_INTENTS

    @property
    def intent_label(self) -> str:
        return INTENT_LABELS.get(self.intent, str(self.intent))

    def filter_conditions(self) -> list[dict[str, Any]]:
        return [f.to_condition() for f in self.filters]

    def describe_steps(self) -> list[str]:
        """The plan's steps, with a generated fallback when the model gave none."""
        if self.steps:
            return list(self.steps)

        steps: list[str] = []
        targets = [t for t in (self.metric, *self.dimensions, self.time_column) if t]
        if targets:
            steps.append("Identify the " + ", ".join(targets) + " field(s)")
        for condition in self.filters:
            steps.append(f"Filter to rows where {condition.describe()}")
        grouped = f" by {', '.join(self.dimensions)}" if self.dimensions else ""
        if self.metric and self.aggregation:
            steps.append(f"Aggregate {self.aggregation} of {self.metric}{grouped}")
        elif self.dimensions:
            steps.append(f"Count rows{grouped}")
        if self.time_column:
            steps.append(
                f"Group by {self.time_granularity or 'month'} using {self.time_column}"
            )
        if self.comparison:
            steps.append(f"Calculate {self.comparison.replace('_', ' ')}")
        if self.sort_direction:
            steps.append(f"Sort {self.sort_direction} and take the top {self.limit or 10}")
        return [s for s in steps if s] or ["Run the requested analysis"]
