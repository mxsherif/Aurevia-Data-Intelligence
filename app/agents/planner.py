"""The Planner Agent.

The one place where a natural-language question becomes a machine-checkable
plan. The model's whole job is to read intent and name fields; it receives a
compact dataset description (never data) and returns an
:class:`~app.models.plans.AnalysisPlan` via OpenAI structured output, so there
is no prose to parse and no path from model text into our control flow.

What the planner must *not* do is also part of the design: it does not compute,
does not choose Python functions, and does not see results. Those belong to
:mod:`app.services.analysis_executor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from app.models.plans import AnalysisPlan, Intent
from app.models.profile import DatasetProfile
from app.services.dataset_context import DatasetContext, build_dataset_context
from app.services.llm_service import (
    LLMError,
    LLMService,
    LLMUsage,
    get_llm_service,
)
from app.tools.aggregations import supported_aggregations
from app.tools.charts import supported_chart_types
from app.tools.filters import supported_operators
from app.tools.timeseries import supported_frequencies

logger = logging.getLogger(__name__)

#: Longest question we will send. Anything longer is almost certainly a paste.
MAX_QUESTION_LENGTH = 500


SYSTEM_PROMPT = """\
You are the planning component of Aurevia, a data-analysis platform. You turn \
a user's question about a specific dataset into a structured analysis plan.

You do NOT perform analysis. You do NOT compute, estimate or state any numbers. \
Python executes the plan you produce and calculates every value. Your output is \
a plan only.

Choose exactly one `intent` from the available set:
- summary: descriptive statistics for a measure or the dataset
- ranking: which categories are highest or lowest on a measure
- comparison: compare a measure across the values of one field
- segmentation: break a measure down by a field (same machinery as comparison)
- trend: how a measure moves over time
- distribution: the spread and shape of one numeric field
- correlation: which numeric fields move together
- time_comparison: one period against the preceding period
- percentage_change: how much each group moved between two points in time
- count: how many rows fall into each category
- anomaly: unusual or extreme values in a numeric field
- dataset_question: what the dataset contains, its fields or its coverage

Field rules:
- `metric` must name a numeric column from the dataset description, or be null \
when the question is about row counts or the dataset itself.
- `dimensions` must name categorical columns. Use one unless the question \
clearly asks for two.
- `time_column` must name a column whose type is "datetime".
- Use the EXACT column names given in the dataset description. If the user's \
wording differs from a column name, map it to the closest real column.
- If no column plausibly matches an essential part of the question, set \
`requires_clarification` to true and ask one short question.

`steps` is shown to the user as an execution summary. Write 3-6 short \
imperative lines describing what will be computed, in order. Never include \
your own reasoning, instructions, or any numbers.

Set `requires_clarification` only when the question is genuinely ambiguous and \
no sensible default exists -- for example "which customers are best?", where \
"best" could mean several different measures. Do not ask for clarification \
when a reasonable default is obvious.
"""


@dataclass
class PlannerOutcome:
    """A plan, or the reason there isn't one."""

    plan: AnalysisPlan | None = None
    error: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    #: The context that was sent, for the UI's transparency panel.
    context: DatasetContext | None = None

    @property
    def ok(self) -> bool:
        return self.plan is not None

    @property
    def needs_clarification(self) -> bool:
        return bool(
            self.plan is not None
            and self.plan.requires_clarification
            and self.plan.clarification_question
        )


class PlannerAgent:
    """Turns a question into an :class:`AnalysisPlan`."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or get_llm_service()

    @property
    def llm(self) -> LLMService:
        return self._llm

    @property
    def available(self) -> bool:
        return self._llm.available

    def plan(
        self,
        question: str,
        df: pd.DataFrame,
        profile: DatasetProfile | None = None,
        *,
        dataset_name: str = "dataset",
        context: DatasetContext | None = None,
        context_block: str = "",
        feedback: str | None = None,
    ) -> PlannerOutcome:
        """Plan the analysis for `question` against `df`.

        `context_block` is the previous analysis's state, supplied only when
        the context layer judged this question to be a continuation -- so a
        self-contained question is never influenced by what came before.

        `feedback` is a validation failure from an earlier attempt, so a retry
        is informed rather than a coin flip.

        Never raises: an LLM failure comes back as
        :attr:`PlannerOutcome.error` with a message fit for the UI.
        """
        cleaned = (question or "").strip()
        if not cleaned:
            return PlannerOutcome(error="Please enter a question first.")
        if len(cleaned) > MAX_QUESTION_LENGTH:
            return PlannerOutcome(
                error=(
                    f"That question is {len(cleaned)} characters long. Please "
                    f"keep it under {MAX_QUESTION_LENGTH}."
                )
            )

        if not self._llm.available:
            return PlannerOutcome(error=self._llm.unavailable_reason)

        dataset = context or build_dataset_context(df, profile, name=dataset_name)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": self._capabilities()},
            {"role": "system", "content": self._dataset_message(dataset)},
        ]
        if context_block:
            messages.append(
                {"role": "system", "content": self._context_message(context_block)}
            )
        if feedback:
            messages.append({"role": "system", "content": feedback})
        messages.append({"role": "user", "content": cleaned})

        try:
            response = self._llm.complete_structured(
                messages=messages, schema=AnalysisPlan
            )
        except LLMError as exc:
            logger.info("Planning failed: %s", exc)
            return PlannerOutcome(error=str(exc), context=dataset)
        except Exception as exc:  # noqa: BLE001 - this method promises not to raise
            logger.exception("Unexpected planner failure")
            return PlannerOutcome(
                error=(
                    "Aurevia could not plan this analysis because of an "
                    f"unexpected error ({type(exc).__name__}). The details were "
                    "logged."
                ),
                context=dataset,
            )

        plan = response.data
        if not isinstance(plan, AnalysisPlan):  # pragma: no cover - defensive
            return PlannerOutcome(
                error="The planner returned an unexpected result.", context=dataset
            )

        logger.info(
            "Planned %s | metric=%s dimensions=%s time=%s (%d tokens)",
            plan.intent, plan.metric, plan.dimensions, plan.time_column,
            response.usage.total_tokens,
        )
        return PlannerOutcome(plan=plan, usage=response.usage, context=dataset)

    # -- prompt construction ----------------------------------------------- #

    @staticmethod
    def _capabilities() -> str:
        """Tell the model exactly which vocabulary the executor accepts.

        Generated from the Phase 2 registries rather than hard-coded, so a new
        aggregation or chart type is offered to the planner automatically.
        """
        return (
            "Available vocabulary -- use only these values:\n"
            f"- intents: {', '.join(i.value for i in Intent)}\n"
            f"- aggregations: {', '.join(supported_aggregations())}\n"
            f"- time_granularity: {', '.join(supported_frequencies())}\n"
            f"- filter operators: {', '.join(supported_operators())}\n"
            f"- visualization: {', '.join(supported_chart_types())}, or none\n"
            "- comparison: percentage_change, absolute_change, vs_overall, or null\n"
            "- sort_direction: ascending or descending\n"
        )

    @staticmethod
    def _context_message(context_block: str) -> str:
        """The previous analysis's state, for a follow-up question."""
        return (
            "This question appears to follow on from the previous one. The "
            "state of that analysis was:\n"
            f"{context_block}\n"
            "Carry over only what the new question leaves unsaid. Anything the "
            "new question names explicitly replaces the old value. If the new "
            "question is about a different measure, do not keep the old one."
        )

    @staticmethod
    def _dataset_message(dataset: DatasetContext) -> str:
        return (
            "The dataset the user is asking about, as JSON. These are the only "
            "columns that exist; no data rows are included.\n"
            f"{dataset.to_prompt()}"
        )


def build_plan(
    question: str,
    df: pd.DataFrame,
    profile: DatasetProfile | None = None,
    *,
    llm: LLMService | None = None,
    dataset_name: str = "dataset",
    context: DatasetContext | None = None,
    context_block: str = "",
    feedback: str | None = None,
) -> PlannerOutcome:
    """Convenience wrapper around :class:`PlannerAgent`."""
    return PlannerAgent(llm).plan(
        question, df, profile, dataset_name=dataset_name, context=context,
        context_block=context_block, feedback=feedback,
    )


__all__ = [
    "MAX_QUESTION_LENGTH",
    "SYSTEM_PROMPT",
    "PlannerAgent",
    "PlannerOutcome",
    "build_plan",
]
