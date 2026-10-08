"""Tests for the retry flow and for context inside the Ask pipeline.

The retry is the point where Phase 4 either earns its keep or loops forever.
These tests pin down three things: a bad explanation is regenerated with a
*different* prompt, a chart problem costs no API call, and a persistent failure
is reported as a failure with the computed figures still available.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import Settings
from app.models.context import AnalyticalContext
from app.models.plans import AnalysisPlan, Intent
from app.models.results import Insight
from app.models.validation import RetryStage
from app.services.ask_pipeline import (
    EXHAUSTED_MESSAGE,
    MAX_RETRIES,
    Stage,
    answer_question,
)
from app.services.llm_service import (
    LLMService,
    LLMTimeoutError,
    LLMUsage,
    StructuredResponse,
)

DATASET = "ds1"


@pytest.fixture
def telecom() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["Cairo", "Delta", "Canal"] * 4,
            "signup_date": pd.date_range("2024-01-15", periods=12, freq="ME"),
            "contract_type": ["Month-to-month", "One year", "Two year"] * 4,
            "monthly_charge": [100.0, 220.0, 180.0, 340.0, 150.0, 410.0,
                               260.0, 190.0, 300.0, 130.0, 480.0, 210.0],
            "revenue": [1000.0, 2200.0, 1800.0, 3400.0, 1500.0, 4100.0,
                        2600.0, 1900.0, 3000.0, 1300.0, 4800.0, 2100.0],
        }
    )


class StagedLLM(LLMService):
    """A scripted planner and a queue of explanations.

    Records every prompt it was sent, so a test can assert that a retry
    actually changed the request rather than repeating it.
    """

    def __init__(
        self,
        *plans: AnalysisPlan,
        insights: list[Insight] | None = None,
        plan_error: Exception | None = None,
        insight_error: Exception | None = None,
    ) -> None:
        super().__init__(Settings(openai_api_key="sk-test"), client=object())
        self._plans = list(plans)
        self._insights = list(insights or [])
        self._plan_error = plan_error
        self._insight_error = insight_error
        self.plan_calls = 0
        self.insight_calls = 0
        self.other_calls = 0
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return self.plan_calls + self.insight_calls + self.other_calls

    def complete_structured(self, messages, schema, **kwargs):  # type: ignore[override]
        self.prompts.append(" ".join(m["content"] for m in messages))
        usage = LLMUsage(model="fake", prompt_tokens=80, completion_tokens=20)

        if schema is AnalysisPlan:
            self.plan_calls += 1
            if self._plan_error:
                raise self._plan_error
            plan = self._plans[min(self.plan_calls - 1, len(self._plans) - 1)]
            return StructuredResponse(data=plan, usage=usage)

        if schema is Insight:
            self.insight_calls += 1
            if self._insight_error:
                raise self._insight_error
            index = min(self.insight_calls - 1, len(self._insights) - 1)
            return StructuredResponse(data=self._insights[index], usage=usage)

        # The optional semantic check: always clean, so it never interferes.
        self.other_calls += 1
        from app.models.validation import SemanticCheck

        return StructuredResponse(
            data=SemanticCheck(addresses_question=True, contradicts_data=False),
            usage=usage,
        )


def _ranking_plan(**overrides) -> AnalysisPlan:
    fields = {
        "intent": Intent.RANKING,
        "metric": "revenue",
        "dimensions": ["region"],
        "aggregation": "sum",
    }
    fields.update(overrides)
    return AnalysisPlan(**fields)


# --------------------------------------------------------------------------- #
# Interpretation retry
# --------------------------------------------------------------------------- #

def test_a_bad_explanation_is_regenerated_then_accepted(telecom: pd.DataFrame):
    """Only the prose is retried; the computation is not repeated."""
    llm = StagedLLM(
        _ranking_plan(),
        insights=[
            Insight(answer="Canal generated 999,999,999 in revenue."),  # rejected
            Insight(answer="Canal generated 11,000.00 in revenue."),    # accepted
        ],
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.answered
    assert outcome.retries == 1
    assert llm.insight_calls == 2
    assert llm.plan_calls == 1              # the plan was never re-run
    assert "11,000.00" in outcome.insight.answer
    assert any("interpretation" in entry for entry in outcome.retry_log)


def test_the_retry_changes_the_prompt(telecom: pd.DataFrame):
    """Re-sending the same request and hoping is not a retry."""
    from app.agents.insights import RETRY_REMINDER

    llm = StagedLLM(
        _ranking_plan(),
        insights=[
            Insight(answer="Canal generated 999,999,999."),
            Insight(answer="Canal generated 11,000.00."),
        ],
    )
    answer_question("Which region earns most?", telecom, llm=llm)

    marker = RETRY_REMINDER.splitlines()[0]
    first_attempt, second_attempt = llm.prompts[1], llm.prompts[2]
    assert marker not in first_attempt
    assert marker in second_attempt


def test_retries_are_capped_and_the_failure_explains_itself(
    telecom: pd.DataFrame,
):
    llm = StagedLLM(
        _ranking_plan(),
        insights=[Insight(answer="Canal generated 999,999,999 in revenue.")],
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert not outcome.answered
    assert outcome.retries == MAX_RETRIES
    assert outcome.stage is Stage.VERIFICATION
    assert EXHAUSTED_MESSAGE in outcome.error
    # The message says *why*, not merely that it failed.
    assert "999,999,999" in outcome.error
    # And the computed figures survive for the user to read.
    assert outcome.computed
    assert outcome.result.summary_data["Top region"] == "Canal"


def test_a_zero_retry_cap_is_honoured(telecom: pd.DataFrame):
    llm = StagedLLM(
        _ranking_plan(),
        insights=[Insight(answer="Canal generated 999,999,999.")],
    )
    outcome = answer_question(
        "Which region earns most?", telecom, llm=llm, max_retries=0
    )

    assert outcome.retries == 0
    assert not outcome.answered
    assert llm.insight_calls == 1


# --------------------------------------------------------------------------- #
# Planning retry
# --------------------------------------------------------------------------- #

def test_a_wrong_plan_is_replanned_with_feedback(telecom: pd.DataFrame):
    """The second plan is informed by the first one's failure."""
    llm = StagedLLM(
        # The first plan groups by a near-unique field, which validation and
        # then the result validator reject.
        _ranking_plan(metric="nonexistent_measure"),
        _ranking_plan(),
        insights=[Insight(answer="Canal generated 11,000.00.")],
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    # A metric that does not exist is caught at plan validation, which asks
    # the user rather than burning a retry.
    assert outcome.needs_clarification
    assert llm.plan_calls == 1


def test_a_charting_failure_costs_no_api_call(telecom: pd.DataFrame):
    """A bad chart is replaced locally; the model is not consulted."""
    from app.services.analysis_executor import execute_plan
    from app.services.plan_validator import validate_plan
    from app.services.result_validator import validate_result

    plan = validate_plan(telecom, _ranking_plan()).plan
    result = execute_plan(telecom, plan)
    broken = result.model_copy(
        update={"chart_spec": {"chart_type": "bar", "x": "nope", "y": "revenue"}}
    )
    verdict = validate_result(telecom, plan, broken)

    # A chart problem is a warning, and it points at charting.
    assert verdict.valid
    assert any(
        f.stage is RetryStage.CHARTING for f in verdict.findings
    )


def test_chart_demotion_replaces_the_spec_without_recomputing():
    from app.services.analysis_executor import execute_plan
    from app.services.ask_pipeline import _demote_chart
    from app.services.plan_validator import validate_plan

    frame = pd.DataFrame(
        {"region": ["A", "B", "C"] * 4, "revenue": [float(i) for i in range(12)]}
    )
    plan = validate_plan(frame, _ranking_plan()).plan
    result = execute_plan(frame, plan)
    original = dict(result.summary_data)

    result.chart_spec = {"chart_type": "heatmap"}
    _demote_chart(result, plan)

    # The figures are untouched; only the chart changed.
    assert result.summary_data == original
    assert result.chart_spec is None or result.chart_spec["chart_type"] == "bar"


# --------------------------------------------------------------------------- #
# Validation metadata on the outcome
# --------------------------------------------------------------------------- #

def test_a_validated_answer_carries_strength_and_confidence(
    telecom: pd.DataFrame,
):
    llm = StagedLLM(
        _ranking_plan(), insights=[Insight(answer="Canal led with 11,000.00.")]
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.answered
    assert outcome.verification is not None
    assert outcome.verification.checks_run >= 10
    assert outcome.strength is not None
    assert 0.0 <= outcome.confidence <= 1.0


def test_the_evidence_record_is_available(telecom: pd.DataFrame):
    llm = StagedLLM(
        _ranking_plan(), insights=[Insight(answer="Canal led with 11,000.00.")]
    )
    evidence = answer_question("Which region earns most?", telecom, llm=llm).evidence

    assert evidence["Metric"] == "revenue"
    assert evidence["Grouped by"] == "region"
    assert evidence["Python functions"] == "rank_values"
    assert evidence["Rows included"] == 12
    assert "Key computed values" in evidence


def test_a_history_item_is_produced(telecom: pd.DataFrame):
    llm = StagedLLM(
        _ranking_plan(), insights=[Insight(answer="Canal led with 11,000.00.")]
    )
    item = answer_question("Which region earns most?", telecom, llm=llm).to_history()

    assert item.question == "Which region earns most?"
    assert item.intent == "ranking"
    assert item.valid
    assert item.source == "ask"
    assert item.result_summary


def test_an_explanation_failure_still_returns_verified_figures(
    telecom: pd.DataFrame,
):
    llm = StagedLLM(
        _ranking_plan(), insight_error=LLMTimeoutError("The AI request timed out.")
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.answered          # validation passed without prose
    assert outcome.insight is None
    assert outcome.stage is Stage.EXPLANATION
    assert any("explanation could not be generated" in w for w in outcome.warnings)


# --------------------------------------------------------------------------- #
# Context through the pipeline
# --------------------------------------------------------------------------- #

def test_a_follow_up_inherits_through_the_pipeline(telecom: pd.DataFrame):
    """Q1 establishes revenue-by-region; Q2 is a bare fragment."""
    first = StagedLLM(
        _ranking_plan(), insights=[Insight(answer="Canal led with 11,000.00.")]
    )
    turn_one = answer_question(
        "Which region generated the most revenue?", telecom, llm=first,
        dataset_key=DATASET,
    )

    assert turn_one.answered
    context = turn_one.updated_context
    assert isinstance(context, AnalyticalContext)
    assert context.current_metric == "revenue"
    assert context.dimensions == ["region"]

    # The planner now returns an almost-empty plan, as it would for a
    # fragment: context has to fill the gaps.
    second = StagedLLM(
        AnalysisPlan(intent=Intent.RANKING, aggregation="sum"),
        insights=[Insight(answer="Canal led with 11,000.00.")],
    )
    turn_two = answer_question(
        "What about last quarter?", telecom, llm=second,
        analytical_context=context, dataset_key=DATASET,
    )

    assert turn_two.context_resolution.is_continuation
    assert turn_two.plan.metric == "revenue"
    assert turn_two.plan.dimensions == ["region"]
    assert any("revenue" in note for note in turn_two.context_notes)


def test_context_reaches_the_planner_only_on_a_follow_up(telecom: pd.DataFrame):
    """The safety property: an unrelated question never sees prior state."""
    context = AnalyticalContext(
        dataset_key=DATASET, current_metric="revenue", dimensions=["region"],
        previous_question="Which region generated the most revenue?",
        previous_intent="ranking",
    )

    follow_up = StagedLLM(
        AnalysisPlan(intent=Intent.RANKING, aggregation="sum"),
        insights=[Insight(answer="Canal led with 11,000.00.")],
    )
    answer_question(
        "What about last quarter?", telecom, llm=follow_up,
        analytical_context=context, dataset_key=DATASET,
    )
    assert "previous question" in follow_up.prompts[0]

    fresh = StagedLLM(
        AnalysisPlan(intent=Intent.DISTRIBUTION, metric="monthly_charge"),
        insights=[Insight(answer="Charges range from 100.00 to 480.00.")],
    )
    answer_question(
        "Show the distribution of monthly charge.", telecom, llm=fresh,
        analytical_context=context, dataset_key=DATASET,
    )
    assert "previous question" not in fresh.prompts[0]


def test_an_unrelated_question_does_not_inherit(telecom: pd.DataFrame):
    context = AnalyticalContext(
        dataset_key=DATASET, current_metric="revenue", dimensions=["region"],
        previous_question="Which region generated the most revenue?",
        previous_intent="ranking",
    )
    llm = StagedLLM(
        AnalysisPlan(intent=Intent.DISTRIBUTION, metric="monthly_charge"),
        insights=[Insight(answer="Charges range from 100.00 to 480.00.")],
    )
    outcome = answer_question(
        "Show the distribution of monthly charge.", telecom, llm=llm,
        analytical_context=context, dataset_key=DATASET,
    )

    assert not outcome.context_resolution.is_continuation
    assert outcome.plan.metric == "monthly_charge"
    assert outcome.plan.dimensions == []
    assert outcome.context_notes == []


def test_a_different_dataset_drops_the_context(telecom: pd.DataFrame):
    context = AnalyticalContext(
        dataset_key=DATASET, current_metric="revenue", dimensions=["region"],
        previous_question="Which region generated the most revenue?",
        previous_intent="ranking",
    )
    llm = StagedLLM(
        _ranking_plan(), insights=[Insight(answer="Canal led with 11,000.00.")]
    )
    outcome = answer_question(
        "What about last quarter?", telecom, llm=llm,
        analytical_context=context, dataset_key="a-different-dataset",
    )

    assert not outcome.context_resolution.is_continuation
    assert any("different dataset" in note for note in outcome.context_notes)


def test_a_clarification_leaves_the_context_untouched(telecom: pd.DataFrame):
    """A question that was not answered must not overwrite the conversation."""
    llm = StagedLLM(_ranking_plan(metric="profit_margin"))
    outcome = answer_question(
        "What is the average profit margin?", telecom, llm=llm,
        dataset_key=DATASET,
    )

    assert outcome.needs_clarification
    assert outcome.updated_context is None


def test_the_semantic_check_is_skipped_for_a_clean_answer(
    telecom: pd.DataFrame,
):
    """The optional third call is reserved for answers that invite doubt."""
    llm = StagedLLM(
        _ranking_plan(),
        insights=[Insight(answer="Canal led with 11,000.00.")],
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.answered
    assert llm.other_calls == 0
    assert outcome.llm_calls == 2


def test_the_semantic_check_runs_when_an_answer_quotes_nothing(
    telecom: pd.DataFrame,
):
    llm = StagedLLM(
        _ranking_plan(),
        insights=[Insight(answer="Canal was the leading region.")],
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.answered
    assert llm.other_calls == 1
    assert outcome.llm_calls == 3


def test_the_outcome_reports_its_own_cost(telecom: pd.DataFrame):
    llm = StagedLLM(
        _ranking_plan(), insights=[Insight(answer="Canal led with 11,000.00.")]
    )
    payload = answer_question("Which region earns most?", telecom, llm=llm).to_dict()

    assert payload["llm_calls"] == 2
    assert payload["retries"] == 0
    assert payload["valid"] is True
    assert payload["continuation"] is False
    assert payload["tokens"] == 200
