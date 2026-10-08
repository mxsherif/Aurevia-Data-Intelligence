"""The Investigation Planner and the investigation write-up.

Two LLM calls bracket an investigation, and nothing in between:

- :class:`InvestigationPlannerAgent` reads "why did revenue decline last
  quarter?" and names the metric, the date field, the period size and the
  dimensions worth decomposing. It also reports what direction the question
  *claims*, which the engine checks against the data.
- :class:`InvestigationSummaryAgent` receives the computed evidence and writes
  it up, under instructions that permit concentration and association but
  forbid cause.

The dimensions the planner suggests are screened by deterministic rules before
any of them is used, so a suggestion of `customer_id` costs nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from app.models.investigation import (
    InvestigationPlan,
    InvestigationResult,
    InvestigationSummary,
)
from app.models.profile import DatasetProfile
from app.services.dataset_context import DatasetContext, build_dataset_context
from app.services.grounding import GroundingReport, check_texts
from app.services.investigation_engine import MAX_DIMENSIONS
from app.services.llm_service import (
    LLMError,
    LLMService,
    LLMUsage,
    get_llm_service,
)
from app.tools.timeseries import supported_frequencies

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 500


PLANNER_PROMPT = """\
You are the investigation planner inside Aurevia, a data-analysis platform. \
The user wants to understand why a measure changed. You decide what to inspect; \
Python then computes every figure.

You do NOT analyse, compute, estimate or state any numbers. You produce a plan.

Choose:
- `metric`: the numeric column whose change is in question.
- `time_column`: a column whose type is "datetime".
- `aggregation`: sum for totals, count for volumes, mean for rates or averages.
- `granularity`: the period size the question implies. "last quarter" means \
quarterly; "last month" means monthly. Default to quarterly.
- `candidate_dimensions`: three to five CATEGORICAL columns that could explain \
where the change came from, most promising first. Prefer business dimensions \
such as region, product, segment, contract or channel. Never choose an \
identifier, a free-text field, a date, or a numeric measure.
- `claimed_direction`: what the question asserts happened -- "decrease" for \
decline/drop/fall wording, "increase" for growth/rise wording, "unknown" when \
the question asserts nothing. Report what the user claimed, not what you \
believe is true: Python verifies the claim against the data.
- `steps`: three to seven short lines describing what will be computed, in \
order. This is shown to the user. No reasoning, no numbers.

Set `requires_clarification` only when the question names no measure that \
exists in the dataset, or is too vague to identify one.
"""


SUMMARY_PROMPT = """\
You are the explanation component of Aurevia. A deterministic investigation has \
already run and computed every figure you are given. Write up what it found.

ABSOLUTE RULE ON NUMBERS
Only use numerical values contained in the supplied evidence. Do not calculate, \
infer, estimate, re-scale or invent any number. If a figure you want is not in \
the evidence, describe the finding without a number.

ASSOCIATION, NOT CAUSE
The investigation measured where a change was concentrated. It did not measure \
why. Write about concentration and association only.

Acceptable:
- "The decline was concentrated in the Western region."
- "The strongest negative contribution came from Fiber Internet."
- "This pattern was associated with month-to-month contracts."
- "The largest deterioration occurred in August."

Not acceptable:
- "The decline was caused by..."
- "Customers left because..."
- "This proves..."
- "...due to poor service"

WHAT TO WRITE
- `headline`: one or two sentences stating what changed, by how much, and where \
it was concentrated.
- `contributors`: two to four lines, each naming a category and its supplied \
figure.
- `timing`: one sentence on when within the period the movement was largest, if \
the timing evidence shows one. Otherwise null.
- `caution`: one short caution when the evidence warrants it -- offsetting \
movement, small samples, uneven period coverage, or that this is association \
rather than cause. Otherwise null.
- `next_questions`: two to four follow-up questions specific to what was found. \
Not generic prompts.

If `premise_confirmed` is false, say plainly that the change the question \
assumed did not happen, state what actually happened using the supplied \
figures, and do not speculate about the premise.
"""


@dataclass
class InvestigationPlanOutcome:
    plan: InvestigationPlan | None = None
    error: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    context: DatasetContext | None = None

    @property
    def ok(self) -> bool:
        return self.plan is not None


@dataclass
class InvestigationSummaryOutcome:
    summary: InvestigationSummary | None = None
    error: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    grounding: GroundingReport | None = None

    @property
    def ok(self) -> bool:
        return self.summary is not None

    @property
    def is_grounded(self) -> bool:
        return self.grounding is None or self.grounding.is_grounded


class InvestigationPlannerAgent:
    """Turns an investigation question into a structured plan."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or get_llm_service()

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
    ) -> InvestigationPlanOutcome:
        """Plan the investigation for `question`. Never raises."""
        cleaned = (question or "").strip()
        if not cleaned:
            return InvestigationPlanOutcome(
                error="Please describe what you would like investigated."
            )
        if len(cleaned) > MAX_QUESTION_LENGTH:
            return InvestigationPlanOutcome(
                error=(
                    f"That question is {len(cleaned)} characters long. Please "
                    f"keep it under {MAX_QUESTION_LENGTH}."
                )
            )
        if not self._llm.available:
            return InvestigationPlanOutcome(error=self._llm.unavailable_reason)

        dataset = context or build_dataset_context(df, profile, name=dataset_name)

        try:
            response = self._llm.complete_structured(
                messages=[
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "system", "content": self._capabilities()},
                    {
                        "role": "system",
                        "content": (
                            "The dataset, as JSON. These are the only columns "
                            "that exist; no data rows are included.\n"
                            f"{dataset.to_prompt()}"
                        ),
                    },
                    {"role": "user", "content": cleaned},
                ],
                schema=InvestigationPlan,
            )
        except LLMError as exc:
            logger.info("Investigation planning failed: %s", exc)
            return InvestigationPlanOutcome(error=str(exc), context=dataset)
        except Exception as exc:  # noqa: BLE001 - promises not to raise
            logger.exception("Unexpected investigation planner failure")
            return InvestigationPlanOutcome(
                error=(
                    "Aurevia could not plan this investigation because of an "
                    f"unexpected error ({type(exc).__name__}). The details "
                    "were logged."
                ),
                context=dataset,
            )

        plan = response.data
        if not isinstance(plan, InvestigationPlan):  # pragma: no cover
            return InvestigationPlanOutcome(
                error="The investigation planner returned an unexpected result.",
                context=dataset,
            )

        logger.info(
            "Investigation planned: metric=%s time=%s granularity=%s dims=%s "
            "claim=%s",
            plan.metric, plan.time_column, plan.granularity,
            plan.candidate_dimensions, plan.claimed_direction,
        )
        return InvestigationPlanOutcome(
            plan=plan, usage=response.usage, context=dataset
        )

    @staticmethod
    def _capabilities() -> str:
        return (
            "Available vocabulary -- use only these values:\n"
            f"- granularity: {', '.join(supported_frequencies())}\n"
            "- aggregation: sum, count, mean\n"
            "- claimed_direction: decrease, increase, unknown\n"
            f"- candidate_dimensions: at most {MAX_DIMENSIONS} column names\n"
        )


class InvestigationSummaryAgent:
    """Writes up a completed investigation."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or get_llm_service()

    @property
    def available(self) -> bool:
        return self._llm.available

    def summarise(
        self,
        question: str,
        result: InvestigationResult,
        *,
        dataset_name: str = "the dataset",
    ) -> InvestigationSummaryOutcome:
        """Explain `result`. Never raises; the write-up is grounding-checked."""
        if not result.success:
            return InvestigationSummaryOutcome(
                error="There is no completed investigation to explain."
            )
        if not self._llm.available:
            return InvestigationSummaryOutcome(error=self._llm.unavailable_reason)

        payload = json.dumps(
            {
                "question": question,
                "dataset": dataset_name,
                "evidence": result.to_evidence(),
            },
            separators=(",", ":"),
            default=str,
        )

        try:
            response = self._llm.complete_structured(
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": payload},
                ],
                schema=InvestigationSummary,
            )
        except LLMError as exc:
            logger.info("Investigation summary failed: %s", exc)
            return InvestigationSummaryOutcome(error=str(exc))
        except Exception as exc:  # noqa: BLE001 - promises not to raise
            logger.exception("Unexpected investigation summary failure")
            return InvestigationSummaryOutcome(
                error=(
                    f"an unexpected error occurred ({type(exc).__name__}); "
                    "the details were logged"
                )
            )

        summary = response.data
        if not isinstance(summary, InvestigationSummary):  # pragma: no cover
            return InvestigationSummaryOutcome(
                error="The investigation summary returned no content."
            )

        report = check_texts(summary.texts(), allowed_numbers(result))
        summary.ungrounded_numbers = list(report.ungrounded)
        if report.ungrounded:
            logger.warning(
                "Ungrounded figures in an investigation summary: %s",
                report.ungrounded,
            )

        return InvestigationSummaryOutcome(
            summary=summary, usage=response.usage, grounding=report
        )


def allowed_numbers(result: InvestigationResult) -> set[float]:
    """Every figure the investigation computed, for the grounding check.

    Built from the same evidence payload the model was given, so the two
    cannot drift apart.
    """
    from app.models.results import AnalysisResult

    # Reuse the one implementation of "which numbers count as computed".
    carrier = AnalysisResult(
        success=True,
        summary_data={
            "baseline_value": result.baseline_value,
            "comparison_value": result.comparison_value,
            "absolute_change": result.absolute_change,
            "percentage_change": result.percentage_change,
            "baseline_label": result.baseline_label,
            "comparison_label": result.comparison_label,
        },
        table_data=[f.to_dict() for f in result.all_findings]
        + [f.to_dict() for f in result.temporal_findings],
        metadata=result.evidence_summary,
    )
    return carrier.allowed_numbers()


__all__ = [
    "MAX_QUESTION_LENGTH",
    "PLANNER_PROMPT",
    "SUMMARY_PROMPT",
    "InvestigationPlanOutcome",
    "InvestigationPlannerAgent",
    "InvestigationSummaryAgent",
    "InvestigationSummaryOutcome",
    "allowed_numbers",
]
