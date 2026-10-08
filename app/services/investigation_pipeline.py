"""The Investigate pipeline.

    question -> plan (LLM) -> validate (Python) -> run (Python) -> write up (LLM)

Exactly two LLM calls, regardless of how many dimensions get decomposed: the
planner is not consulted per dimension, and the engine never calls out. A
premise that fails verification costs one call, because there is nothing to
write up beyond the figures that disproved it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from app.agents.investigation_planner import (
    InvestigationPlannerAgent,
    InvestigationSummaryAgent,
)
from app.models.investigation import (
    InvestigationPlan,
    InvestigationResult,
    InvestigationSummary,
)
from app.models.profile import DatasetProfile
from app.services.dataset_context import DatasetContext, build_dataset_context
from app.services.investigation_engine import MAX_DIMENSIONS, run_investigation
from app.services.investigation_validator import validate_investigation_plan
from app.services.llm_service import LLMService, LLMUsage, get_llm_service

logger = logging.getLogger(__name__)


class InvestigationStage(str, Enum):
    UNAVAILABLE = "unavailable"
    PLANNING = "planning"
    VALIDATION = "validation"
    EXECUTION = "execution"
    SUMMARY = "summary"
    COMPLETE = "complete"

    def __str__(self) -> str:
        return self.value


@dataclass
class InvestigationOutcome:
    """Everything the Investigate page needs for one investigation."""

    question: str
    stage: InvestigationStage = InvestigationStage.PLANNING

    plan: InvestigationPlan | None = None
    result: InvestigationResult | None = None
    summary: InvestigationSummary | None = None

    error: str | None = None
    clarification_question: str | None = None
    warnings: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)

    usage: LLMUsage = field(default_factory=LLMUsage)
    llm_calls: int = 0
    context: DatasetContext | None = None
    elapsed_seconds: float = 0.0

    # -- state ------------------------------------------------------------- #

    @property
    def investigated(self) -> bool:
        return self.result is not None and self.result.success

    @property
    def premise_rejected(self) -> bool:
        return self.investigated and not self.result.premise_confirmed

    @property
    def needs_clarification(self) -> bool:
        return self.clarification_question is not None

    @property
    def failed(self) -> bool:
        return not self.investigated and not self.needs_clarification

    @property
    def is_grounded(self) -> bool:
        return self.summary is None or self.summary.is_grounded

    @property
    def steps(self) -> list[str]:
        return self.plan.describe_steps() if self.plan else []

    @property
    def follow_ups(self) -> list[str]:
        return list(self.summary.next_questions) if self.summary else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "stage": self.stage.value,
            "investigated": self.investigated,
            "premise_confirmed": (
                self.result.premise_confirmed if self.result else None
            ),
            "metric": self.plan.metric if self.plan else None,
            "dimensions": (
                self.result.dimensions_inspected if self.result else []
            ),
            "grounded": self.is_grounded,
            "llm_calls": self.llm_calls,
            "tokens": self.usage.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
        }


def investigate(
    question: str,
    df: pd.DataFrame,
    profile: DatasetProfile | None = None,
    *,
    dataset_name: str = "dataset",
    llm: LLMService | None = None,
    explain: bool = True,
    max_dimensions: int = MAX_DIMENSIONS,
    context: DatasetContext | None = None,
) -> InvestigationOutcome:
    """Investigate `question` against `df`. Never raises."""
    started = time.perf_counter()
    outcome = InvestigationOutcome(question=(question or "").strip())
    service = llm or get_llm_service()

    def finish(result: InvestigationOutcome) -> InvestigationOutcome:
        result.elapsed_seconds = time.perf_counter() - started
        logger.info("Investigation: %s", result.to_dict())
        return result

    if not outcome.question:
        outcome.error = "Please describe what you would like investigated."
        return finish(outcome)

    if not service.available:
        outcome.stage = InvestigationStage.UNAVAILABLE
        outcome.error = service.unavailable_reason
        return finish(outcome)

    # -- Plan (LLM call 1) -------------------------------------------------- #
    dataset = context or build_dataset_context(df, profile, name=dataset_name)
    outcome.context = dataset

    planned = InvestigationPlannerAgent(service).plan(
        outcome.question, df, profile, dataset_name=dataset_name, context=dataset
    )
    outcome.usage = planned.usage
    outcome.llm_calls += 1

    if not planned.ok:
        outcome.stage = InvestigationStage.PLANNING
        outcome.error = planned.error or "Aurevia could not plan this investigation."
        return finish(outcome)

    outcome.plan = planned.plan

    # -- Validate (Python) -------------------------------------------------- #
    validation = validate_investigation_plan(df, planned.plan)
    outcome.adjustments = list(validation.adjustments)

    if validation.needs_clarification:
        outcome.stage = InvestigationStage.VALIDATION
        outcome.clarification_question = validation.clarification_question
        return finish(outcome)
    if not validation.ok or validation.plan is None:
        outcome.stage = InvestigationStage.VALIDATION
        outcome.error = (
            validation.message
            or "Aurevia understood the request, but this dataset cannot "
            "support that investigation."
        )
        return finish(outcome)

    outcome.plan = validation.plan

    # -- Run (Python, no LLM) ----------------------------------------------- #
    result = run_investigation(
        df, validation.plan,
        question=outcome.question, max_dimensions=max_dimensions,
    )
    outcome.result = result

    if not result.success:
        outcome.stage = InvestigationStage.EXECUTION
        outcome.error = result.error or "The investigation produced no result."
        return finish(outcome)

    outcome.warnings.extend(result.warnings)

    # -- Write up (LLM call 2) ---------------------------------------------- #
    if not explain:
        outcome.stage = InvestigationStage.COMPLETE
        return finish(outcome)

    written = InvestigationSummaryAgent(service).summarise(
        outcome.question, result, dataset_name=dataset_name
    )
    outcome.llm_calls += 1
    outcome.usage = LLMUsage(
        model=written.usage.model or outcome.usage.model,
        prompt_tokens=outcome.usage.prompt_tokens + written.usage.prompt_tokens,
        completion_tokens=(
            outcome.usage.completion_tokens + written.usage.completion_tokens
        ),
    )

    if written.ok:
        outcome.summary = written.summary
        if not written.is_grounded:
            outcome.warnings.append(
                "Some figures in the written summary could not be matched to "
                "the computed evidence. Trust the tables below."
            )
        outcome.stage = InvestigationStage.COMPLETE
    else:
        # The evidence stands on its own without the prose.
        outcome.stage = InvestigationStage.SUMMARY
        outcome.warnings.append(
            f"The written summary could not be generated ({written.error}) but "
            "the computed evidence below is complete."
        )

    return finish(outcome)


__all__ = ["InvestigationOutcome", "InvestigationStage", "investigate"]
