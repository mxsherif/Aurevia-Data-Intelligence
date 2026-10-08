"""Validation of planner output against the real dataset.

The planner is an LLM. It is good at reading intent and bad at being certain,
so nothing it produces is executed as given. Every plan passes through here
first, where each field is checked against the actual dataframe and either
**repaired**, **rejected**, or left alone.

Three outcomes:

- ``ok`` — the plan is executable. ``adjustments`` records anything repaired,
  so the UI can tell the user that "monthly bill" became `monthly_charge`.
- ``needs_clarification`` — the question is genuinely ambiguous, or a reference
  could not be resolved safely. The user gets a question, not a wrong answer.
- ``rejected`` — the dataset cannot answer this (no date column for a trend,
  no numeric column to correlate).

The guiding rule: **never silently analyse the wrong column.** A low-confidence
column match is a clarification, not a guess.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from app.models.plans import (
    AnalysisPlan,
    Intent,
    METRIC_INTENTS,
    PlanFilter,
    SortDirection,
    TIME_INTENTS,
)
from app.tools.aggregations import resolve_aggregation, supported_aggregations
from app.tools.charts import supported_chart_types
from app.tools.exceptions import ToolError
from app.tools.filters import build_mask, normalize_operator, supported_operators
from app.tools.resolution import ColumnMatch, resolve_column
from app.tools.timeseries import supported_frequencies
from app.tools.validation import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
    require_dataframe,
)

logger = logging.getLogger(__name__)

#: Hard cap on the planner's `limit`, whatever it asks for.
MAX_LIMIT = 100
DEFAULT_LIMIT = 10

#: Values the planner may use to mean "no chart".
NO_CHART_TOKENS = frozenset({"none", "null", "no", "table", "nothing", ""})

#: A grouping column this close to one-row-per-record summarises nothing.
ID_LIKE_RATIO = 0.5
ID_LIKE_MIN_DISTINCT = 10

#: Intents whose answer reads better with surrounding context than with a
#: single row.
CONTEXT_INTENTS = frozenset(
    {
        Intent.RANKING,
        Intent.COMPARISON,
        Intent.SEGMENTATION,
        Intent.COUNT,
        Intent.PERCENTAGE_CHANGE,
    }
)

#: Comparison styles the executor understands.
SUPPORTED_COMPARISONS = frozenset(
    {"percentage_change", "absolute_change", "vs_overall"}
)


class Verdict(str, Enum):
    OK = "ok"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"

    def __str__(self) -> str:
        return self.value


@dataclass
class PlanValidation:
    """The result of checking one plan."""

    verdict: Verdict
    plan: AnalysisPlan | None = None
    #: User-facing note for each repair we made.
    adjustments: list[str] = field(default_factory=list)
    #: Why the plan was rejected, or what we need clarified.
    message: str | None = None
    #: A concise question when `verdict` is NEEDS_CLARIFICATION.
    clarification_question: str | None = None
    #: Technical detail for the log, never shown in the UI.
    diagnostics: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK and self.plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "adjustments": list(self.adjustments),
            "message": self.message,
            "clarification_question": self.clarification_question,
        }


class _Validator:
    """Holds the per-request state so the checks stay readable."""

    def __init__(
        self,
        df: pd.DataFrame,
        plan: AnalysisPlan,
        question: str = "",
    ) -> None:
        self.df = df
        self.plan = plan
        self.question = question
        self.columns = [str(c) for c in df.columns]
        self.numeric = numeric_columns(df)
        self.categorical = categorical_columns(df, max_unique=1000)
        self.dates = datetime_columns(df)
        self.adjustments: list[str] = []
        self.diagnostics: list[str] = []

    # -- helpers ----------------------------------------------------------- #

    def _resolve(
        self, term: str, *, role: str, allowed: list[str] | None = None
    ) -> ColumnMatch:
        """Resolve a column reference, recording any rename as an adjustment."""
        match = resolve_column(term, self.columns, allowed=allowed)
        if match.is_trusted and match.was_renamed:
            self.adjustments.append(
                f"Read {role} '{term}' as the column `{match.column}`."
            )
        if match.found and not match.is_trusted:
            self.diagnostics.append(
                f"{role} '{term}' matched '{match.column}' only by "
                f"{match.kind} ({match.confidence})."
            )
        return match

    def clarify(self, question: str, message: str | None = None) -> PlanValidation:
        return PlanValidation(
            verdict=Verdict.NEEDS_CLARIFICATION,
            adjustments=self.adjustments,
            message=message,
            clarification_question=question,
            diagnostics=self.diagnostics,
        )

    def reject(self, message: str) -> PlanValidation:
        return PlanValidation(
            verdict=Verdict.REJECTED,
            adjustments=self.adjustments,
            message=message,
            diagnostics=self.diagnostics,
        )

    # -- field checks ------------------------------------------------------ #

    def check_metric(self) -> PlanValidation | None:
        """Resolve the metric onto a real numeric column."""
        plan = self.plan
        if plan.metric is None:
            # Intents that cannot compute anything without a measure.
            if plan.intent in METRIC_INTENTS and plan.intent is not Intent.ANOMALY:
                if not self.numeric:
                    return self.reject(
                        "Aurevia understood the request, but this dataset has no "
                        "numeric column to measure."
                    )
            return None

        match = self._resolve(plan.metric, role="metric", allowed=self.numeric or None)

        if not match.found:
            # It may be a real column that simply is not numeric.
            anywhere = resolve_column(plan.metric, self.columns)
            if anywhere.is_trusted and anywhere.column not in self.numeric:
                if plan.intent in (Intent.COUNT, Intent.DATASET_QUESTION):
                    plan.metric = None
                    return None
                return self.reject(
                    f"Aurevia understood the request, but `{anywhere.column}` is "
                    "not a numeric column, so it cannot be measured. Try a "
                    "numeric field such as "
                    + ", ".join(f"`{c}`" for c in self.numeric[:3])
                    + "."
                )
            if not self.numeric:
                return self.reject(
                    "Aurevia understood the request, but this dataset has no "
                    "numeric column to measure."
                )
            return self.clarify(
                f"I could not find a field matching '{plan.metric}'. "
                "Which measure should I use?"
                + (
                    " Available: " + ", ".join(self.numeric[:5]) + "."
                    if self.numeric
                    else ""
                )
            )

        if not match.is_trusted:
            return self.clarify(
                f"Did you mean `{match.column}` when you said "
                f"'{plan.metric}'?"
            )

        plan.metric = match.column

        # The column exists -- but is it the one the user asked about?
        substituted = detect_metric_substitution(
            self.df, self.question, plan.metric
        )
        if substituted:
            self.diagnostics.append(
                f"planner chose '{plan.metric}' while the question names "
                f"'{substituted}', which does not exist"
            )
            return self.clarify(
                f"This dataset has no field matching '{substituted}'. "
                "Which measure should I use instead?"
                + (
                    " Available: " + ", ".join(self.numeric[:5]) + "."
                    if self.numeric
                    else ""
                )
            )
        return None

    def check_dimensions(self) -> PlanValidation | None:
        """Resolve grouping columns, dropping ones that cannot be grouped."""
        plan = self.plan
        resolved: list[str] = []
        for term in plan.dimensions:
            match = self._resolve(term, role="grouping field")
            if not match.found:
                self.adjustments.append(
                    f"Ignored '{term}': no matching column in this dataset."
                )
                continue
            if not match.is_trusted:
                return self.clarify(
                    f"Did you mean `{match.column}` when you said '{term}'?"
                )
            if match.column == plan.metric:
                self.adjustments.append(
                    f"Ignored `{match.column}` as a grouping field; it is the "
                    "measure being analysed."
                )
                continue
            if match.column not in resolved:
                resolved.append(match.column)

        # Grouping by a near-unique identifier produces one row per record,
        # which is never the answer to "which X is highest". The test is the
        # *ratio*, not an absolute count: a city column with 23 values out of
        # 5,000 rows summarises well, while 5,000 customer IDs do not -- and
        # neither do 12 IDs in a 12-row table.
        guarded: list[str] = []
        rows = max(len(self.df), 1)
        for column in resolved:
            unique = int(self.df[column].nunique(dropna=True))
            if unique >= max(ID_LIKE_MIN_DISTINCT, ID_LIKE_RATIO * rows):
                self.adjustments.append(
                    f"Ignored `{column}` as a grouping field: it has {unique:,} "
                    "distinct values, so grouping by it would not summarise anything."
                )
                continue
            guarded.append(column)

        plan.dimensions = guarded

        if not plan.dimensions and plan.intent in (
            Intent.RANKING, Intent.COMPARISON, Intent.SEGMENTATION, Intent.COUNT
        ):
            if not self.categorical:
                return self.reject(
                    "Aurevia understood the request, but this dataset has no "
                    "categorical column to group by."
                )
            return self.clarify(
                "Which field should I break the results down by?"
                " Available: " + ", ".join(self.categorical[:5]) + "."
            )
        return None

    def check_aggregation(self) -> PlanValidation | None:
        """Confirm the aggregation exists and suits the metric."""
        plan = self.plan

        if plan.aggregation is None:
            if plan.metric is not None and plan.intent not in (
                Intent.DISTRIBUTION, Intent.CORRELATION, Intent.ANOMALY,
                Intent.DATASET_QUESTION,
            ):
                plan.aggregation = "sum" if plan.intent is Intent.TREND else "mean"
                self.adjustments.append(
                    f"No aggregation was specified; used {plan.aggregation}."
                )
            elif plan.metric is None and plan.intent in (
                Intent.COUNT, Intent.RANKING, Intent.COMPARISON, Intent.SEGMENTATION
            ):
                plan.aggregation = "count"
            return None

        try:
            aggregator = resolve_aggregation(plan.aggregation)
        except ToolError:
            fallback = "mean"
            self.adjustments.append(
                f"'{plan.aggregation}' is not a supported aggregation; "
                f"used {fallback} instead. Supported: "
                + ", ".join(supported_aggregations()) + "."
            )
            self.diagnostics.append(f"unsupported aggregation {plan.aggregation!r}")
            plan.aggregation = fallback
            return None

        if aggregator.numeric_only and plan.metric is None:
            plan.aggregation = "count"
            self.adjustments.append(
                f"{aggregator.name} needs a numeric field; counted rows instead."
            )
            return None

        plan.aggregation = aggregator.name
        return None

    def check_time(self) -> PlanValidation | None:
        """Confirm time analysis has a usable date column and granularity."""
        plan = self.plan
        needs_time = plan.intent in TIME_INTENTS

        if plan.time_column is not None:
            match = self._resolve(
                plan.time_column, role="date field", allowed=self.dates or None
            )
            if match.found and match.is_trusted:
                plan.time_column = match.column
            else:
                if not self.dates:
                    if needs_time:
                        return self.reject(
                            "Aurevia understood the request, but this dataset "
                            "does not contain a valid date field required for "
                            "the comparison."
                        )
                    self.adjustments.append(
                        f"Ignored the date field '{plan.time_column}': this "
                        "dataset has no usable date column."
                    )
                    plan.time_column = None
                    plan.time_granularity = None
                else:
                    plan.time_column = self.dates[0]
                    self.adjustments.append(
                        f"Used `{plan.time_column}` as the date field."
                    )
        elif needs_time:
            if not self.dates:
                return self.reject(
                    "Aurevia understood the request, but this dataset does not "
                    "contain a valid date field required for time-based analysis."
                )
            plan.time_column = self.dates[0]
            self.adjustments.append(f"Used `{plan.time_column}` as the date field.")

        if plan.time_column is None:
            plan.time_granularity = None
            return None

        granularity = (plan.time_granularity or "").strip().lower()
        if granularity not in supported_frequencies():
            if granularity:
                self.adjustments.append(
                    f"'{plan.time_granularity}' is not a supported time "
                    "grouping; used monthly."
                )
            plan.time_granularity = "monthly"
        else:
            plan.time_granularity = granularity

        if plan.periods is not None:
            # A trend needs at least two points to be a trend; a ranking
            # scoped to "last quarter" legitimately wants exactly one.
            floor = 2 if plan.intent in TIME_INTENTS else 1
            plan.periods = max(floor, min(int(plan.periods), 120))
        return None

    def check_filters(self) -> PlanValidation | None:
        """Resolve, validate and test every filter against the data."""
        plan = self.plan
        if not plan.filters:
            return None

        kept: list[PlanFilter] = []
        for condition in plan.filters:
            match = self._resolve(condition.column, role="filter field")
            if not match.found or not match.is_trusted:
                self.adjustments.append(
                    f"Dropped the filter on '{condition.column}': "
                    "no matching column in this dataset."
                )
                self.diagnostics.append(f"unresolved filter column {condition.column!r}")
                continue

            try:
                operator = normalize_operator(condition.operator)
            except ToolError:
                self.adjustments.append(
                    f"Dropped the filter on `{match.column}`: "
                    f"'{condition.operator}' is not a supported comparison. "
                    "Supported: " + ", ".join(supported_operators()) + "."
                )
                continue

            resolved = PlanFilter(
                column=match.column,
                operator=operator,
                value=condition.value,
                values=list(condition.values),
            )

            # A date range needs a date column.
            if operator == "date_between" and match.column not in self.dates:
                self.adjustments.append(
                    f"Dropped the date filter on `{match.column}`: it is not a "
                    "date column."
                )
                continue

            # Dry-run the condition so a malformed value fails here, with a
            # message, rather than mid-execution.
            try:
                mask = build_mask(self.df, resolved.to_condition())
            except ToolError as exc:
                self.adjustments.append(
                    f"Dropped the filter on `{match.column}`: {exc}"
                )
                self.diagnostics.append(f"filter rejected: {exc}")
                continue

            matched = int(mask.sum())
            if matched == 0:
                return self.reject(
                    f"Aurevia understood the request, but no rows match "
                    f"{resolved.describe()}. "
                    + _value_hint(self.df, match.column)
                )
            kept.append(resolved)

        plan.filters = kept
        return None

    def check_limit(self) -> None:
        """Sanitise the row/category limit."""
        plan = self.plan
        if plan.limit is None:
            plan.limit = DEFAULT_LIMIT
            return
        try:
            limit = int(plan.limit)
        except (TypeError, ValueError):
            plan.limit = DEFAULT_LIMIT
            return

        clamped = max(1, min(limit, MAX_LIMIT))
        if clamped != limit:
            self.adjustments.append(
                f"Clamped the result limit from {limit} to {clamped}."
            )

        # A superlative question ("which region is highest?") makes the planner
        # ask for a single row. The top value is already reported in the
        # headline figures, so returning only that one row would throw away the
        # comparison -- and a one-row result has no chart worth drawing.
        if clamped == 1 and plan.intent in CONTEXT_INTENTS:
            clamped = DEFAULT_LIMIT
            self.adjustments.append(
                "Returned the full ranking for context, not just the top row."
            )

        plan.limit = clamped

    def check_sort(self) -> None:
        plan = self.plan
        if plan.sort_direction is None:
            plan.sort_direction = (
                SortDirection.ASCENDING
                if plan.comparison == "percentage_change"
                and plan.intent in (Intent.PERCENTAGE_CHANGE,)
                else SortDirection.DESCENDING
            )

    def check_comparison(self) -> None:
        plan = self.plan
        if plan.comparison is None:
            return
        comparison = plan.comparison.strip().lower()
        if comparison not in SUPPORTED_COMPARISONS:
            self.adjustments.append(
                f"Ignored the unsupported comparison '{plan.comparison}'."
            )
            plan.comparison = None
        else:
            plan.comparison = comparison

    def check_visualization(self) -> None:
        """Keep the chart suggestion only if it is a real chart type."""
        plan = self.plan
        if plan.visualization is None:
            return
        requested = plan.visualization.strip().lower()
        if requested in NO_CHART_TOKENS:
            plan.visualization = None
            return
        if requested not in supported_chart_types():
            # Try the chart registry's own aliases before giving up.
            try:
                from app.tools.charts import resolve_chart_type

                plan.visualization = resolve_chart_type(requested).name
                return
            except ToolError:
                self.adjustments.append(
                    f"'{plan.visualization}' is not a supported chart type; "
                    "Aurevia chose one to match the result."
                )
                self.diagnostics.append(f"unsupported chart {plan.visualization!r}")
                plan.visualization = None
                return
        plan.visualization = requested


#: Splits text into comparable word tokens.
_WORDS = re.compile("[a-z0-9]+")

#: Comparatives, superlatives and question scaffolding that shape a question
#: without naming a field. "Which region is biggest?" names no measure, so the
#: planner choosing a sensible one is helpful, not a substitution.
_RANKING_WORDS = frozenset(
    {
        "best", "worst", "biggest", "largest", "smallest", "highest", "lowest",
        "most", "least", "top", "bottom", "greatest", "poorest", "strongest",
        "weakest", "better", "worse", "high", "low", "many", "much", "often",
        "common", "popular", "unusual", "typical", "average", "mean", "median",
        "total", "count", "number", "overall", "across", "compare",
        "comparison", "show", "which", "what", "does", "generated", "generate",
        "vary", "varies", "change", "changed", "trend", "over", "time",
        "distribution", "correlate", "correlated", "correlation", "rate",
        "between", "with", "from", "into", "during", "latest", "previous",
        "recent", "last", "first", "this", "that", "there", "each", "their",
        "customer", "customers", "record", "records", "value", "values",
        "have", "were", "will", "they", "them", "about", "than", "then",
        # Time words scope a question; they never name a measure. Without
        # these, "what about last quarter?" reads as a request for a field
        # called "quarter".
        "quarter", "quarters", "month", "months", "year", "years", "week",
        "weeks", "day", "days", "period", "periods", "quarterly", "monthly",
        "yearly", "weekly", "daily", "annual", "annually", "date", "dates",
        "today", "yesterday", "tomorrow", "ytd",
    }
)

#: Minimum length for a question word to be treated as a possible field name.
_MIN_TERM_LENGTH = 4


def detect_metric_substitution(
    df: pd.DataFrame, question: str, metric: str | None
) -> str | None:
    """Return a field the user named that does not exist, if one was swapped.

    Guards against the worst silent failure available here: the user asks for
    "profit margin", the dataset has no such column, and the planner quietly
    answers about `monthly_charge` instead. Choosing a measure the user did not
    name is fine on its own ("which region is biggest?"); it is a substitution
    only when the question *also* names something that does not exist.
    """
    if metric is None or not question:
        return None

    columns = [str(c) for c in df.columns]

    def terms(text: str) -> list[str]:
        words = [
            word
            for word in re.split(r"[^A-Za-z_]+", (text or "").lower())
            if len(word) >= _MIN_TERM_LENGTH and word not in _RANKING_WORDS
        ]
        return [*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))]

    current = terms(question)
    if not current:
        return None

    # 1. Did this question name the chosen metric? Then nothing was swapped.
    for term in current:
        if resolve_column(term, columns).column == metric:
            return None

    # 2. Did this question name a field that does not exist? Phrases are
    #    checked before single words so "profit margin" is reported rather
    #    than "margin". This runs *before* consulting the previous question:
    #    a follow-up that names a nonexistent measure is still a substitution,
    #    whatever the earlier question established.
    for term in sorted(current, key=lambda t: (" " not in t, t)):
        if resolve_column(term, columns).found:
            continue
        if _is_known_value(df, term):
            continue
        if _looks_like_a_field(term):
            return term

    # 3. This question names no measure at all. A follow-up ("what about last
    #    quarter?") legitimately inherits one from the previous analysis, so
    #    there is nothing to object to.
    return None


def _looks_like_a_field(term: str) -> bool:
    """A crude test for "this reads like a field name, not a verb"."""
    if " " in term:
        return True  # "profit margin", "merchandise value"
    # A single unknown word is only suspicious when it is substantial.
    return len(term) >= 6


def _is_known_value(df: pd.DataFrame, term: str) -> bool:
    """True when `term` is part of a category *value* in the data.

    "What about Alexandria?" names a region, not a missing column. Matching is
    on whole words *within* a value, so "internet" is recognised as part of
    "Fiber Internet" -- otherwise asking about a two-word category value would
    flag each of its words as a nonexistent field.
    """
    wanted = set(_WORDS.findall(term.lower()))
    if not wanted:
        return False

    for column in categorical_columns(df, max_unique=200):
        try:
            values = df[column].dropna().astype(str).unique()
        except Exception:  # noqa: BLE001 - a hint is optional
            continue
        for value in values:
            # Token containment, not a substring match: "internet" is part of
            # "Fiber Internet", but "net" is not.
            if wanted <= set(_WORDS.findall(str(value).lower())):
                return True
    return False


def _value_hint(df: pd.DataFrame, column: str) -> str:
    """Suggest real values for a filter that matched nothing."""
    try:
        values = (
            df[column].dropna().astype(str).value_counts().head(5).index.tolist()
        )
    except Exception:  # noqa: BLE001 - a hint is optional
        return ""
    if not values:
        return ""
    return f"Values present in `{column}` include: " + ", ".join(values) + "."


def validate_plan(
    df: pd.DataFrame, plan: AnalysisPlan, question: str = ""
) -> PlanValidation:
    """Check `plan` against `df`, repairing what is safe and refusing the rest.

    The returned plan is a copy; the planner's original output is left intact
    for logging and display.
    """
    require_dataframe(df)

    if df.empty:
        return PlanValidation(
            verdict=Verdict.REJECTED,
            message="The loaded dataset has no rows to analyse.",
        )

    working = plan.model_copy(deep=True)

    # The planner itself may ask for clarification.
    if working.requires_clarification and working.clarification_question:
        return PlanValidation(
            verdict=Verdict.NEEDS_CLARIFICATION,
            plan=working,
            clarification_question=working.clarification_question,
        )

    validator = _Validator(df, working, question)

    for check in (
        validator.check_metric,
        validator.check_dimensions,
        validator.check_aggregation,
        validator.check_time,
        validator.check_filters,
    ):
        outcome = check()
        if outcome is not None:
            logger.info(
                "Plan %s: %s", outcome.verdict, outcome.message or
                outcome.clarification_question
            )
            return outcome

    validator.check_limit()
    validator.check_sort()
    validator.check_comparison()
    validator.check_visualization()

    if validator.diagnostics:
        logger.debug("Plan validation diagnostics: %s", validator.diagnostics)

    return PlanValidation(
        verdict=Verdict.OK,
        plan=working,
        adjustments=validator.adjustments,
        diagnostics=validator.diagnostics,
    )


__all__ = [
    "CONTEXT_INTENTS",
    "detect_metric_substitution",
    "ID_LIKE_RATIO",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "PlanValidation",
    "SUPPORTED_COMPARISONS",
    "Verdict",
    "validate_plan",
]
