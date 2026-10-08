"""Analytical context: what the session remembers between questions.

Not a chat transcript. The only thing carried forward is the *analytical state*
of the last answered question — which metric, which dimension, which filters,
which period — so that "what about last quarter?" can be understood without
re-sending the conversation to the model.

Two properties matter:

- **It is compact.** A handful of field names and filters, not prose. The
  planner receives at most a few dozen tokens of context.
- **It is scoped to one dataset.** Context is stamped with the dataset it came
  from and is dropped the moment a different file is loaded. Carrying "revenue
  by region" onto an unrelated dataset would be worse than having no memory.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TimeRange(BaseModel):
    """A period the user scoped an analysis to."""

    model_config = ConfigDict(extra="forbid")

    column: str | None = None
    granularity: str | None = None
    #: How many of the most recent periods ("the last six months" -> 6).
    periods: int | None = None
    #: An explicit window, when the user named dates.
    start: str | None = None
    end: str | None = None
    #: The phrase the user used, for display.
    label: str | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.column, self.granularity, self.periods, self.start, self.end)
        )

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.start and self.end:
            return f"{self.start} to {self.end}"
        if self.periods and self.granularity:
            return f"the last {self.periods} {self.granularity} period(s)"
        if self.granularity:
            return f"by {self.granularity}"
        return self.column or ""


class AnalyticalContext(BaseModel):
    """The analytical state carried between questions in one session."""

    model_config = ConfigDict(extra="forbid")

    #: The dataset this context belongs to. A mismatch means discard.
    dataset_key: str | None = None

    current_metric: str | None = None
    dimensions: list[str] = Field(default_factory=list)
    filters: list[dict[str, Any]] = Field(default_factory=list)
    time_range: TimeRange | None = None
    comparison_period: str | None = None
    #: Category values the user named ("Cairo", "Fiber Internet"), so a later
    #: "compare that with Delta" has something to compare against.
    referenced_entities: list[str] = Field(default_factory=list)

    previous_question: str | None = None
    previous_intent: str | None = None
    previous_result_summary: dict[str, Any] | None = None
    #: How many questions have been answered against this dataset.
    turn_count: int = 0

    # -- state ------------------------------------------------------------- #

    @property
    def is_empty(self) -> bool:
        return self.previous_question is None and self.current_metric is None

    @property
    def primary_dimension(self) -> str | None:
        return self.dimensions[0] if self.dimensions else None

    def belongs_to(self, dataset_key: str | None) -> bool:
        """True when this context was built from `dataset_key`."""
        return self.dataset_key is not None and self.dataset_key == dataset_key

    def reset(self, dataset_key: str | None = None) -> AnalyticalContext:
        """A fresh context for `dataset_key`, keeping nothing."""
        return AnalyticalContext(dataset_key=dataset_key)

    # -- prompt surface ---------------------------------------------------- #

    def to_prompt(self) -> str:
        """The compact description handed to the planner, or ``""`` if empty.

        Only reached when a continuation signal was detected, so an unrelated
        question never sees it.
        """
        if self.is_empty:
            return ""

        parts: list[str] = []
        if self.previous_question:
            parts.append(f'previous question: "{self.previous_question}"')
        if self.previous_intent:
            parts.append(f"previous analysis: {self.previous_intent}")
        if self.current_metric:
            parts.append(f"metric: {self.current_metric}")
        if self.dimensions:
            parts.append("grouped by: " + ", ".join(self.dimensions))
        if self.filters:
            parts.append(
                "filters: "
                + "; ".join(
                    f"{f.get('column')} {f.get('operator')} {f.get('value')}"
                    for f in self.filters
                )
            )
        if self.time_range and not self.time_range.is_empty:
            parts.append(f"time: {self.time_range.describe()}")
        if self.referenced_entities:
            parts.append("values mentioned: " + ", ".join(self.referenced_entities[:6]))
        return "\n".join(f"- {p}" for p in parts)

    def describe(self) -> str:
        """A one-line summary for the UI."""
        if self.is_empty:
            return "No prior analysis in this session."
        bits = []
        if self.current_metric:
            bits.append(self.current_metric)
        if self.dimensions:
            bits.append("by " + ", ".join(self.dimensions))
        if self.time_range and not self.time_range.is_empty:
            bits.append(self.time_range.describe())
        return " · ".join(bits) or (self.previous_intent or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "current_metric": self.current_metric,
            "dimensions": list(self.dimensions),
            "filters": list(self.filters),
            "time_range": self.time_range.model_dump() if self.time_range else None,
            "comparison_period": self.comparison_period,
            "referenced_entities": list(self.referenced_entities),
            "previous_question": self.previous_question,
            "previous_intent": self.previous_intent,
            "turn_count": self.turn_count,
        }


class AnalysisHistoryItem(BaseModel):
    """One completed analysis, kept for the session only."""

    model_config = ConfigDict(extra="forbid")

    question: str
    intent: str
    plan_summary: list[str] = Field(default_factory=list)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    #: Whether the answer passed validation, so history shows what to trust.
    valid: bool = True
    strength: str = "strong"
    #: Which page produced it: "ask" or "investigate".
    source: str = "ask"
    title: str = ""

    @property
    def when(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    def headline(self) -> str:
        """The single most useful computed figure, for a compact history row."""
        for key, value in self.result_summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{key}: {value:,.2f}" if isinstance(value, float) else (
                    f"{key}: {value:,}"
                )
        for key, value in self.result_summary.items():
            return f"{key}: {value}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "title": self.title,
            "plan_summary": list(self.plan_summary),
            "result_summary": self.result_summary,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "valid": self.valid,
            "strength": self.strength,
            "source": self.source,
        }


__all__ = ["AnalysisHistoryItem", "AnalyticalContext", "TimeRange"]
