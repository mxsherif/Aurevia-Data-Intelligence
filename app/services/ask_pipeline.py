"""The Ask Aurevia pipeline.

    question
       -> context resolver      does this continue the previous question?
       -> planner (LLM)         an AnalysisPlan
       -> plan validation       repaired plan, clarification, or refusal
       -> executor (Python)     an AnalysisResult -- every number lives here
       -> result validator      did we actually answer what was asked?
       -> visualization         a chart matched to the result shape
       -> insight (LLM)         prose, grounding-checked
       -> context update        what a follow-up might need
       -> follow-up suggestions

Keeping the whole flow in one function rather than in the Streamlit page means
it is testable without a UI, and the page is only presentation.

When validation fails, the retry targets the stage that could fix it rather
than re-running everything: a bad chart costs no API call, a bad explanation
costs one, and only a genuinely wrong plan costs a re-plan. Retries are capped
at :data:`MAX_RETRIES`, and a persistent failure is reported as a failure --
never papered over with an answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from app.agents.insights import InsightAgent, InsightOutcome
from app.agents.planner import PlannerAgent, PlannerOutcome
from app.agents.suggestions import follow_up_questions
from app.models.context import AnalysisHistoryItem, AnalyticalContext
from app.models.plans import AnalysisPlan
from app.models.profile import DatasetProfile
from app.models.results import AnalysisResult, Insight
from app.models.validation import (
    EvidenceStrength,
    RetryStage,
    Severity,
    ValidationResult,
)
from app.services.analysis_executor import execute_plan, select_visualization
from app.services.context_manager import (
    ContextResolution,
    apply_context,
    resolve_context,
    update_context,
)
from app.services.dataset_context import DatasetContext, build_dataset_context
from app.services.llm_service import LLMService, LLMUsage, get_llm_service
from app.services.plan_validator import PlanValidation, Verdict, validate_plan
from app.services.result_validator import (
    check_semantics,
    merge_finding,
    validate_result,
)

logger = logging.getLogger(__name__)

#: Automated retries per question. Two is enough to fix a bad explanation and
#: then a bad plan; more risks a loop and a surprising bill.
MAX_RETRIES = 2

#: The message shown when retries are exhausted. Deliberately plain.
EXHAUSTED_MESSAGE = (
    "Aurevia could not confidently complete this analysis from the available "
    "data."
)


class Stage(str, Enum):
    """How far the pipeline got."""

    UNAVAILABLE = "unavailable"
    PLANNING = "planning"
    VALIDATION = "validation"
    COMPUTATION = "computation"
    EXPLANATION = "explanation"
    VERIFICATION = "verification"
    COMPLETE = "complete"

    def __str__(self) -> str:
        return self.value


@dataclass
class AskOutcome:
    """Everything the Ask page needs to render one question's answer."""

    question: str
    stage: Stage = Stage.PLANNING

    plan: AnalysisPlan | None = None
    validation: PlanValidation | None = None
    result: AnalysisResult | None = None
    insight: Insight | None = None
    verification: ValidationResult | None = None
    context_resolution: ContextResolution | None = None

    error: str | None = None
    clarification_question: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: What the context layer carried forward, shown to the user.
    context_notes: list[str] = field(default_factory=list)

    retries: int = 0
    retry_log: list[str] = field(default_factory=list)

    usage: LLMUsage = field(default_factory=LLMUsage)
    llm_calls: int = 0
    context: DatasetContext | None = None
    elapsed_seconds: float = 0.0

    #: The analytical context after this question, for the caller to store.
    #: ``None`` when nothing was computed, so the caller keeps what it had.
    updated_context: AnalyticalContext | None = None

    # -- state ------------------------------------------------------------- #

    @property
    def answered(self) -> bool:
        """True when Python computed a result that passed validation."""
        if self.result is None or not self.result.success:
            return False
        return self.verification is None or self.verification.valid

    @property
    def computed(self) -> bool:
        """True when figures exist, even if validation rejected them."""
        return self.result is not None and self.result.success

    @property
    def needs_clarification(self) -> bool:
        return self.clarification_question is not None

    @property
    def failed(self) -> bool:
        return not self.answered and not self.needs_clarification

    @property
    def is_grounded(self) -> bool:
        return self.insight is None or self.insight.is_grounded

    @property
    def strength(self) -> EvidenceStrength | None:
        return self.verification.strength if self.verification else None

    @property
    def confidence(self) -> float | None:
        return self.verification.confidence if self.verification else None

    @property
    def steps(self) -> list[str]:
        return self.plan.describe_steps() if self.plan else []

    @property
    def follow_ups(self) -> list[str]:
        return list(self.insight.follow_up_questions) if self.insight else []

    @property
    def evidence(self) -> dict[str, Any]:
        """The transparent record of what was executed."""
        if self.result is None or self.plan is None:
            return {}
        metadata = self.result.metadata
        record: dict[str, Any] = {
            "Metric": self.plan.metric or "row counts",
            "Aggregation": self.plan.aggregation or "—",
            "Grouped by": ", ".join(self.plan.dimensions) or "—",
            "Filters": "; ".join(f.describe() for f in self.plan.filters) or "none",
            "Python functions": ", ".join(self.result.tools_used) or "—",
            "Rows included": metadata.get("rows_analysed"),
        }
        if self.plan.time_column:
            record["Date field"] = self.plan.time_column
            record["Time grouping"] = self.plan.time_granularity or "—"
        if metadata.get("rows_before_filter") is not None:
            record["Rows before filtering"] = metadata.get("rows_before_filter")
        record["Chart"] = metadata.get("visualization") or "none"
        record["Key computed values"] = dict(self.result.summary_data)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "stage": self.stage.value,
            "answered": self.answered,
            "intent": str(self.plan.intent) if self.plan else None,
            "metric": self.plan.metric if self.plan else None,
            "dimensions": list(self.plan.dimensions) if self.plan else [],
            "tools": self.result.tools_used if self.result else [],
            "grounded": self.is_grounded,
            "continuation": (
                self.context_resolution.is_continuation
                if self.context_resolution else False
            ),
            "valid": self.verification.valid if self.verification else None,
            "confidence": self.confidence,
            "strength": str(self.strength) if self.strength else None,
            "retries": self.retries,
            "llm_calls": self.llm_calls,
            "tokens": self.usage.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
        }

    def to_history(self) -> AnalysisHistoryItem:
        """A compact record of this analysis for the session history."""
        return AnalysisHistoryItem(
            question=self.question,
            intent=str(self.plan.intent) if self.plan else "unknown",
            title=self.result.title if self.result else "",
            plan_summary=self.steps[:6],
            result_summary=dict(
                list((self.result.summary_data if self.result else {}).items())[:6]
            ),
            valid=self.answered,
            strength=str(self.strength) if self.strength else "limited",
            source="ask",
        )


def _add_usage(total: LLMUsage, addition: LLMUsage) -> LLMUsage:
    return LLMUsage(
        model=addition.model or total.model,
        prompt_tokens=total.prompt_tokens + addition.prompt_tokens,
        completion_tokens=total.completion_tokens + addition.completion_tokens,
    )


def answer_question(
    question: str,
    df: pd.DataFrame,
    profile: DatasetProfile | None = None,
    *,
    dataset_name: str = "dataset",
    llm: LLMService | None = None,
    explain: bool = True,
    context: DatasetContext | None = None,
    analytical_context: AnalyticalContext | None = None,
    dataset_key: str | None = None,
    verify_semantics: bool = True,
    max_retries: int = MAX_RETRIES,
) -> AskOutcome:
    """Answer `question` about `df`.

    The updated analytical context is on :attr:`AskOutcome.updated_context`,
    and is ``None`` when the pipeline stopped before computing anything, so the
    caller keeps whatever it had.

    Never raises. Every failure mode -- no API key, a timeout, an unresolvable
    column, an empty filter, a tool refusal, a failed validation -- comes back
    as an :class:`AskOutcome` carrying a message fit to display.
    """
    started = time.perf_counter()
    outcome = AskOutcome(question=(question or "").strip())
    service = llm or get_llm_service()

    def finish(
        result: AskOutcome, updated: AnalyticalContext | None = None
    ) -> AskOutcome:
        result.updated_context = updated
        result.elapsed_seconds = time.perf_counter() - started
        logger.info("Ask pipeline: %s", result.to_dict())
        return result

    if not outcome.question:
        outcome.error = "Please enter a question first."
        return finish(outcome)

    if not service.available:
        outcome.stage = Stage.UNAVAILABLE
        outcome.error = service.unavailable_reason
        return finish(outcome)

    dataset = context or build_dataset_context(df, profile, name=dataset_name)
    outcome.context = dataset

    # -- Context resolver --------------------------------------------------- #
    resolution = resolve_context(
        outcome.question, analytical_context, df, dataset_key=dataset_key
    )
    outcome.context_resolution = resolution
    outcome.context_notes.extend(resolution.notes)

    planner = PlannerAgent(service)
    insight_agent = InsightAgent(service)

    # The loop body is one full attempt; `feedback` carries a validation
    # failure back into re-planning so the second attempt is informed.
    feedback: str | None = None
    plan: AnalysisPlan | None = None
    result: AnalysisResult | None = None
    insight: Insight | None = None
    verification: ValidationResult | None = None
    retry_stage: RetryStage = RetryStage.PLANNING  # the first pass plans

    while True:
        # -- Plan (LLM) ----------------------------------------------------- #
        if retry_stage is RetryStage.PLANNING:
            planned: PlannerOutcome = planner.plan(
                outcome.question,
                df,
                profile,
                dataset_name=dataset_name,
                context=dataset,
                context_block=resolution.context_prompt,
                feedback=feedback,
            )
            outcome.usage = _add_usage(outcome.usage, planned.usage)
            outcome.llm_calls += 1

            if not planned.ok:
                outcome.stage = Stage.PLANNING
                outcome.error = (
                    planned.error or "Aurevia could not plan this analysis."
                )
                return finish(outcome)

            raw_plan = planned.plan
            outcome.plan = raw_plan

            # Fill the plan's gaps from the previous question, if continuing.
            carried, notes = apply_context(raw_plan, analytical_context, resolution, df)
            if notes:
                outcome.context_notes.extend(notes)

            # -- Validate the plan (Python) --------------------------------- #
            plan_validation = validate_plan(df, carried, outcome.question)
            outcome.validation = plan_validation
            for note in plan_validation.adjustments:
                if note not in outcome.warnings:
                    outcome.warnings.append(note)

            if plan_validation.verdict is Verdict.NEEDS_CLARIFICATION:
                outcome.stage = Stage.VALIDATION
                outcome.clarification_question = (
                    plan_validation.clarification_question
                    or "Could you be more specific about what to analyse?"
                )
                outcome.error = plan_validation.message
                return finish(outcome)

            if plan_validation.verdict is Verdict.REJECTED or plan_validation.plan is None:
                outcome.stage = Stage.VALIDATION
                outcome.error = (
                    plan_validation.message
                    or "Aurevia understood the request, but this dataset "
                    "cannot answer it."
                )
                return finish(outcome)

            plan = plan_validation.plan
            outcome.plan = plan

        assert plan is not None  # every path above either sets it or returns

        # -- Compute (Python) ----------------------------------------------- #
        if retry_stage in (RetryStage.PLANNING, RetryStage.EXECUTION):
            result = execute_plan(df, plan)
            outcome.result = result
            if not result.success:
                outcome.stage = Stage.COMPUTATION
                outcome.error = result.error or "The analysis produced no result."
                return finish(outcome)
            for note in result.notes:
                if note not in outcome.warnings:
                    outcome.warnings.append(note)
            insight = None  # the figures changed, so any prose is stale

        assert result is not None

        # -- Chart (Python) ------------------------------------------------- #
        if retry_stage is RetryStage.CHARTING:
            _demote_chart(result, plan)

        # -- Explain (LLM) -------------------------------------------------- #
        if explain and (insight is None or retry_stage is RetryStage.INTERPRETATION):
            explained: InsightOutcome = insight_agent.explain(
                outcome.question,
                result,
                dataset_name=dataset_name,
                plan_steps=plan.describe_steps(),
                stricter=retry_stage is RetryStage.INTERPRETATION,
            )
            outcome.usage = _add_usage(outcome.usage, explained.usage)
            outcome.llm_calls += 1

            if explained.ok:
                insight = explained.insight
            else:
                insight = None
                outcome.warnings.append(
                    f"The written explanation could not be generated "
                    f"({explained.error}) but the computed results below are "
                    "complete."
                )
            outcome.insight = insight

        # -- Verify (Python, then optionally one LLM check) ----------------- #
        verification = validate_result(df, plan, result, insight)

        if (
            verify_semantics
            and verification.valid
            and insight is not None
            and service.available
            and _warrants_semantic_check(verification, insight)
        ):
            outcome.llm_calls += 1
            finding = check_semantics(service, outcome.question, result, insight)
            if finding is not None:
                verification = merge_finding(verification, finding)
        outcome.verification = verification

        if verification.valid:
            break

        # -- Retry, targeted at the stage that could fix it ----------------- #
        if outcome.retries >= max_retries or not verification.retry_recommended:
            outcome.stage = Stage.VERIFICATION
            outcome.error = _exhausted_message(verification)
            outcome.warnings.extend(verification.warnings)
            return finish(outcome, None)

        retry_stage = verification.retry_stage
        outcome.retries += 1
        outcome.retry_log.append(
            f"Attempt {outcome.retries}: retried {retry_stage} because "
            + "; ".join(verification.issues[:2])
        )
        logger.info("Retrying %s: %s", retry_stage, verification.issues[:2])

        if retry_stage is RetryStage.PLANNING:
            feedback = (
                "A previous attempt at this question failed validation: "
                + " ".join(verification.issues[:3])
                + " Choose columns and an analysis that avoid this."
            )
        elif retry_stage is RetryStage.EXECUTION:
            # Nothing to vary; re-running would fail identically.
            outcome.stage = Stage.VERIFICATION
            outcome.error = _exhausted_message(verification)
            return finish(outcome, None)

    # -- Success ------------------------------------------------------------ #
    outcome.warnings.extend(
        w for w in verification.warnings if w not in outcome.warnings
    )
    if insight is not None and not insight.is_grounded:
        outcome.warnings.append(
            "Some figures in the written answer could not be matched to the "
            "computed results. Trust the table and chart below."
        )
    if insight is not None and not insight.follow_up_questions:
        insight.follow_up_questions = follow_up_questions(
            df, exclude=[outcome.question]
        )

    outcome.stage = Stage.COMPLETE if insight is not None else Stage.EXPLANATION

    updated = update_context(
        analytical_context,
        outcome.question,
        plan,
        result,
        dataset_key=dataset_key,
        resolution=resolution,
    )
    return finish(outcome, updated)


def _warrants_semantic_check(
    verification: ValidationResult, insight: Insight
) -> bool:
    """Whether the one optional LLM validation call is worth making.

    Spending a third API call on every question would raise the cost of the
    common case by half to re-check an answer the deterministic pass already
    found spotless. So it is reserved for answers that show a reason to doubt:

    - the deterministic checks raised a warning, so the result is not clean
      (a NOTE is informational and does not justify a call);
    - the answer quotes no computed figure at all, which is how a model dodges
      a question it cannot answer;
    - the answer is long, which is where drift from the question accumulates.
    """
    if any(f.severity is not Severity.NOTE for f in verification.findings):
        return True
    from app.services.grounding import extract_numbers

    if not extract_numbers(insight.answer):
        return True
    return len(insight.observations) >= 3


def _demote_chart(result: AnalysisResult, plan: AnalysisPlan) -> None:
    """Replace an unusable chart without re-running the analysis.

    The result shape already tells us which chart suits it; the planner's
    suggestion is simply discarded, and if nothing fits, the chart is dropped.
    """
    original = (result.chart_spec or {}).get("chart_type")
    shape = str(result.metadata.get("result_shape", "none"))
    replacement = select_visualization(
        intent=plan.intent, requested=None, shape=shape,
        row_count=result.row_count or len(result.summary_data),
    )

    if replacement is None or replacement == original:
        result.chart_spec = None
        result.metadata["visualization"] = None
        result.metadata["chart_dropped"] = "no chart suited this result"
        return

    from app.services.analysis_executor import normalize_chart_spec

    result.chart_spec = normalize_chart_spec(result.chart_spec or {}, replacement)
    result.metadata["visualization"] = replacement
    result.metadata["chart_replaced_from"] = original


def _exhausted_message(verification: ValidationResult) -> str:
    """Say that we failed, and say why."""
    reasons = verification.issues[:3]
    if not reasons:
        return EXHAUSTED_MESSAGE
    return EXHAUSTED_MESSAGE + " " + " ".join(reasons)


__all__ = [
    "EXHAUSTED_MESSAGE",
    "MAX_RETRIES",
    "AskOutcome",
    "Stage",
    "answer_question",
]
