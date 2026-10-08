"""Conversational analytical context: when to carry it, and when not to.

"What about last quarter?" is meaningless without the previous question. "Show
the distribution of satisfaction scores" is meaningless *with* it. Getting this
wrong in the second direction is worse: inheriting `revenue by region` into an
unrelated question produces a confidently wrong answer rather than a confused
one.

So context is opt-in, not ambient. It is sent to the planner only when the
question shows a **continuation signal** — a linking phrase ("what about",
"and for"), a back-reference ("that", "those"), or a fragment too short to
stand on its own ("last quarter?"). A self-contained question is planned from
the schema alone, exactly as in Phase 3.

Even under continuation, inheritance only fills *gaps*. Anything the new
question names explicitly wins, and naming a different measure drops the old
one rather than keeping both.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.models.context import AnalyticalContext, TimeRange
from app.models.plans import AnalysisPlan, Intent
from app.models.results import AnalysisResult
from app.tools.resolution import resolve_column
from app.tools.validation import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
)

logger = logging.getLogger(__name__)

#: Phrases that link a question to the previous one.
CONTINUATION_PATTERNS: tuple[str, ...] = (
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\band (?:for|in|by|about)\b",
    r"\bnow show\b",
    r"\bnow compare\b",
    r"\balso\b",
    r"\bsame (?:for|but)\b",
    r"\binstead\b",
    r"\bonly for\b",
    r"\bjust for\b",
    r"\bexclude\b",
    r"\bnarrow (?:it|this|that)\b",
    r"\bbreak (?:that|it|this) down\b",
    r"\bdrill (?:in|down)\b",
    r"\bcompare (?:that|it|this|those|these)\b",
    r"\bversus that\b",
    r"\bvs that\b",
    r"\bwhat changed\b",
    r"\bby (?:region|city|product|segment|contract|network|month|quarter)\?$",
)

#: Back-references that only make sense against a previous result.
REFERENCE_PATTERNS: tuple[str, ...] = (
    r"\bthat\b", r"\bthose\b", r"\bthese\b", r"\bthem\b", r"\bthere\b",
    r"\bit\b", r"\bits\b", r"\bthis one\b", r"\bthe same\b", r"\bsuch\b",
    r"\bthe (?:former|latter)\b", r"\bprevious (?:one|analysis|result)\b",
)

#: Time phrases that scope a question without naming a measure.
TIME_PHRASES: dict[str, tuple[str, str | None, int | None]] = {
    # phrase -> (label, granularity, periods)
    "last quarter": ("the last quarter", "quarterly", 1),
    "latest quarter": ("the latest quarter", "quarterly", 1),
    "this quarter": ("the latest quarter", "quarterly", 1),
    "previous quarter": ("the previous quarter", "quarterly", 2),
    "last month": ("the last month", "monthly", 1),
    "latest month": ("the latest month", "monthly", 1),
    "this month": ("the latest month", "monthly", 1),
    "previous month": ("the previous month", "monthly", 2),
    "last year": ("the last year", "yearly", 1),
    "latest year": ("the latest year", "yearly", 1),
    "this year": ("the latest year", "yearly", 1),
    "previous year": ("the previous year", "yearly", 2),
    "last week": ("the last week", "weekly", 1),
    "year to date": ("the year to date", "monthly", 12),
    "last six months": ("the last six months", "monthly", 6),
    "last 6 months": ("the last six months", "monthly", 6),
    "last three months": ("the last three months", "monthly", 3),
    "last 3 months": ("the last three months", "monthly", 3),
    "last twelve months": ("the last twelve months", "monthly", 12),
    "last 12 months": ("the last twelve months", "monthly", 12),
    "over time": ("over time", "monthly", None),
}

#: A question this short with no field named is treated as a fragment.
FRAGMENT_WORD_LIMIT = 6

#: Intents that start a genuinely new line of enquiry, so stale assumptions
#: about metric and grouping are dropped even under a continuation signal.
RESETTING_INTENTS = frozenset(
    {Intent.DATASET_QUESTION, Intent.CORRELATION}
)

_WORD = re.compile(r"[a-z0-9_]+")


@dataclass
class ContextResolution:
    """What the context layer decided for one question."""

    #: True when the question was read as continuing the previous one.
    is_continuation: bool = False
    #: Which signals fired, for the UI and the logs.
    signals: list[str] = field(default_factory=list)
    #: The compact context block for the planner prompt ("" when not carried).
    context_prompt: str = ""
    #: A time scope detected in the question itself.
    time_range: TimeRange | None = None
    #: Measures and dimensions the question names explicitly.
    explicit_metric: str | None = None
    explicit_dimensions: list[str] = field(default_factory=list)
    #: Category values named in the question ("Cairo", "Fiber Internet").
    entities: list[str] = field(default_factory=list)
    #: User-facing notes about what was carried forward.
    notes: list[str] = field(default_factory=list)

    @property
    def carries_context(self) -> bool:
        return self.is_continuation and bool(self.context_prompt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_continuation": self.is_continuation,
            "signals": list(self.signals),
            "explicit_metric": self.explicit_metric,
            "explicit_dimensions": list(self.explicit_dimensions),
            "entities": list(self.entities),
            "time_range": self.time_range.model_dump() if self.time_range else None,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Signal detection
# --------------------------------------------------------------------------- #

def _matches(question: str, patterns: tuple[str, ...]) -> list[str]:
    lowered = question.lower()
    return [p for p in patterns if re.search(p, lowered)]


def detect_time_phrase(question: str) -> TimeRange | None:
    """Pull a relative time scope out of the question text."""
    lowered = question.lower()
    # Longest phrase first, so "last six months" beats "last".
    for phrase in sorted(TIME_PHRASES, key=len, reverse=True):
        if phrase in lowered:
            label, granularity, periods = TIME_PHRASES[phrase]
            return TimeRange(granularity=granularity, periods=periods, label=label)
    return None


def _named_columns(
    question: str, df: pd.DataFrame
) -> tuple[list[str], list[str], list[str]]:
    """Columns the question names, split into numeric, categorical and date."""
    columns = [str(c) for c in df.columns]
    numeric = set(numeric_columns(df))
    categorical = set(categorical_columns(df, max_unique=1000))
    dates = set(datetime_columns(df))

    words = _WORD.findall(question.lower())
    phrases = [f"{a} {b}" for a, b in zip(words, words[1:])]
    # Longer phrases first: "monthly charge" should win over "charge".
    candidates = [*phrases, *words]

    found_numeric: list[str] = []
    found_categorical: list[str] = []
    found_dates: list[str] = []

    for term in candidates:
        match = resolve_column(term, columns)
        if not match.is_trusted:
            continue
        column = match.column
        if column in numeric and column not in found_numeric:
            found_numeric.append(column)
        elif column in categorical and column not in found_categorical:
            found_categorical.append(column)
        elif column in dates and column not in found_dates:
            found_dates.append(column)

    return found_numeric, found_categorical, found_dates


def _named_entities(question: str, df: pd.DataFrame) -> list[str]:
    """Category *values* the question names, e.g. "Cairo" or "Two year".

    Only low-cardinality columns are scanned, and only exact word matches
    count, so this cannot invent a filter the user did not ask for.
    """
    lowered = question.lower()
    found: list[str] = []

    for column in categorical_columns(df, max_unique=60):
        try:
            values = df[column].dropna().astype(str).unique()
        except Exception:  # noqa: BLE001 - a hint is optional
            continue
        for value in values:
            text = str(value).strip()
            if len(text) < 3:
                continue
            if re.search(rf"\b{re.escape(text.lower())}\b", lowered):
                if text not in found:
                    found.append(text)
    return found[:5]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def resolve_context(
    question: str,
    context: AnalyticalContext | None,
    df: pd.DataFrame,
    *,
    dataset_key: str | None = None,
) -> ContextResolution:
    """Decide whether `question` continues the previous analysis.

    Returns a :class:`ContextResolution`. When it is not a continuation, the
    context prompt is empty and the planner sees the question alone.
    """
    resolution = ContextResolution()
    cleaned = (question or "").strip()
    if not cleaned:
        return resolution

    resolution.time_range = detect_time_phrase(cleaned)
    numeric, categorical, _dates = _named_columns(cleaned, df)
    resolution.explicit_metric = numeric[0] if numeric else None
    resolution.explicit_dimensions = categorical
    resolution.entities = _named_entities(cleaned, df)

    # No usable prior state means nothing to continue from.
    if context is None or context.is_empty:
        return resolution
    if dataset_key is not None and not context.belongs_to(dataset_key):
        resolution.notes.append(
            "Previous analysis was on a different dataset, so it was not "
            "carried forward."
        )
        return resolution

    signals: list[str] = []
    if _matches(cleaned, CONTINUATION_PATTERNS):
        signals.append("linking phrase")
    if _matches(cleaned, REFERENCE_PATTERNS):
        signals.append("back-reference")

    words = _WORD.findall(cleaned.lower())
    names_a_field = bool(numeric or categorical)
    short = len(words) <= FRAGMENT_WORD_LIMIT

    # Shortness alone is not a signal. "What does this dataset contain?" is
    # brief and names no field, yet it is a complete question; treating it as a
    # follow-up would answer it about the previous measure. A fragment has to
    # be corroborated by something that only makes sense in context.
    if resolution.time_range is not None and not names_a_field and len(words) <= 10:
        # "last quarter?" -- a period with nothing to apply it to.
        signals.append("time scope only")
    if short and not names_a_field and resolution.entities:
        # "Delta?" -- a category value with nothing to measure about it.
        signals.append("entity fragment")

    if not signals:
        return resolution

    resolution.is_continuation = True
    resolution.signals = signals
    resolution.context_prompt = context.to_prompt()
    logger.info("Continuation detected (%s) for %r", ", ".join(signals), cleaned)
    return resolution


def apply_context(
    plan: AnalysisPlan,
    context: AnalyticalContext | None,
    resolution: ContextResolution,
    df: pd.DataFrame,
) -> tuple[AnalysisPlan, list[str]]:
    """Fill the plan's gaps from context, returning the plan and what changed.

    Only gaps are filled. Anything the planner specified, and anything the new
    question named explicitly, is left alone.
    """
    notes: list[str] = []
    if context is None or not resolution.is_continuation:
        return plan, notes

    updates: dict[str, Any] = {}

    # -- metric --------------------------------------------------------- #
    if plan.intent in RESETTING_INTENTS:
        notes.append(
            f"Started a fresh {plan.intent_label.lower()}; the previous "
            "measure was not carried over."
        )
    elif plan.metric is None and resolution.explicit_metric is None:
        if context.current_metric and context.current_metric in df.columns:
            updates["metric"] = context.current_metric
            notes.append(f"Continued with the measure `{context.current_metric}`.")
    elif (
        resolution.explicit_metric
        and context.current_metric
        and resolution.explicit_metric != context.current_metric
    ):
        notes.append(
            f"Switched the measure to `{resolution.explicit_metric}` as asked."
        )

    # -- dimensions ----------------------------------------------------- #
    if plan.intent not in RESETTING_INTENTS:
        if not plan.dimensions and not resolution.explicit_dimensions:
            carried = [d for d in context.dimensions if d in df.columns]
            if carried:
                updates["dimensions"] = carried
                notes.append("Kept the breakdown by " + ", ".join(carried) + ".")
        elif resolution.explicit_dimensions and context.dimensions:
            if set(resolution.explicit_dimensions) != set(context.dimensions):
                notes.append(
                    "Changed the breakdown to "
                    + ", ".join(resolution.explicit_dimensions)
                    + " as asked."
                )

    # -- time ----------------------------------------------------------- #
    scope = resolution.time_range
    if scope is not None:
        if plan.time_column is None and context.time_range:
            column = context.time_range.column
            if column and column in df.columns:
                updates["time_column"] = column
        if plan.time_granularity is None and scope.granularity:
            updates["time_granularity"] = scope.granularity
        if plan.periods is None and scope.periods:
            updates["periods"] = scope.periods
        notes.append(f"Scoped the analysis to {scope.describe()}.")
    elif plan.is_time_based and plan.time_column is None and context.time_range:
        column = context.time_range.column
        if column and column in df.columns:
            updates["time_column"] = column
            notes.append(f"Reused the date field `{column}`.")

    # -- filters -------------------------------------------------------- #
    # Carried only when the new question adds none of its own, and never onto
    # a column the new question is already filtering or grouping by.
    if not plan.filters and context.filters and plan.intent not in RESETTING_INTENTS:
        occupied = {*(plan.dimensions or []), *(updates.get("dimensions") or [])}
        inherited = [
            f for f in context.filters
            if f.get("column") in df.columns and f.get("column") not in occupied
        ]
        if inherited and not resolution.entities:
            from app.models.plans import PlanFilter

            updates["filters"] = [
                PlanFilter(
                    column=str(f["column"]),
                    operator=str(f.get("operator", "eq")),
                    value=None if f.get("value") is None else str(f.get("value")),
                    values=[str(v) for v in f.get("values", [])],
                )
                for f in inherited
            ]
            notes.append(
                "Kept the filter "
                + "; ".join(f"{f['column']} {f.get('operator')} {f.get('value')}"
                            for f in inherited)
                + "."
            )

    if not updates:
        return plan, notes
    return plan.model_copy(update=updates), notes


def update_context(
    context: AnalyticalContext | None,
    question: str,
    plan: AnalysisPlan,
    result: AnalysisResult | None,
    *,
    dataset_key: str | None = None,
    resolution: ContextResolution | None = None,
) -> AnalyticalContext:
    """The context after a question has been answered.

    Records the analytical state that a follow-up might need, and nothing else
    -- no prose, no transcript.
    """
    base = context if context is not None and context.belongs_to(dataset_key) else (
        AnalyticalContext(dataset_key=dataset_key)
    )

    time_range: TimeRange | None = None
    if plan.time_column:
        label = None
        if resolution is not None and resolution.time_range is not None:
            label = resolution.time_range.label
        time_range = TimeRange(
            column=plan.time_column,
            granularity=plan.time_granularity,
            periods=plan.periods,
            label=label,
        )
    elif base.time_range and plan.is_time_based:
        time_range = base.time_range

    entities = list(resolution.entities) if resolution else []
    if result is not None and result.success:
        # The leading category of a ranking is the thing a follow-up is most
        # likely to reference ("compare that with Delta").
        for key, value in result.summary_data.items():
            if isinstance(value, str) and key.lower().startswith(("top", "highest")):
                if value not in entities:
                    entities.append(value)

    summary: dict[str, Any] | None = None
    if result is not None and result.success:
        summary = {
            "title": result.title,
            **{
                key: value
                for index, (key, value) in enumerate(result.summary_data.items())
                if index < 4
            },
        }

    return AnalyticalContext(
        dataset_key=dataset_key,
        current_metric=plan.metric or (
            base.current_metric if plan.intent not in RESETTING_INTENTS else None
        ),
        dimensions=list(plan.dimensions),
        filters=[f.to_condition() for f in plan.filters],
        time_range=time_range,
        comparison_period=plan.comparison,
        referenced_entities=entities[:6],
        previous_question=question,
        previous_intent=str(plan.intent),
        previous_result_summary=summary,
        turn_count=base.turn_count + 1,
    )


__all__ = [
    "CONTINUATION_PATTERNS",
    "FRAGMENT_WORD_LIMIT",
    "REFERENCE_PATTERNS",
    "RESETTING_INTENTS",
    "TIME_PHRASES",
    "ContextResolution",
    "apply_context",
    "detect_time_phrase",
    "resolve_context",
    "update_context",
]
