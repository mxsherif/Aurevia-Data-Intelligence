"""End-to-end tests for the Ask Aurevia pipeline, with the LLM scripted.

These are the tests that prove the division of labour holds: the stub planner
can be made to emit a wrong, lazy or impossible plan, and the pipeline must
still either compute the right answer or refuse clearly — never produce a
number the model made up.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import Settings
from app.models.plans import AnalysisPlan, Intent, PlanFilter
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


@pytest.fixture
def telecom() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(12)],
            "region": ["Cairo", "Delta", "Canal"] * 4,
            "signup_date": pd.to_datetime(
                [
                    "2024-01-10", "2024-01-25", "2024-02-08", "2024-02-20",
                    "2024-03-04", "2024-03-21", "2024-04-11", "2024-04-27",
                    "2024-05-06", "2024-05-19", "2024-06-02", "2024-06-28",
                ]
            ),
            "contract_type": ["Month-to-month", "One year", "Two year"] * 4,
            "monthly_charge": [100.0, 220.0, 180.0, 340.0, 150.0, 410.0,
                               260.0, 190.0, 300.0, 130.0, 480.0, 210.0],
            "revenue": [1000.0, 2200.0, 1800.0, 3400.0, 1500.0, 4100.0,
                        2600.0, 1900.0, 3000.0, 1300.0, 4800.0, 2100.0],
            "churn": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        }
    )


class ScriptedLLM(LLMService):
    """Returns a scripted plan, then a scripted insight."""

    def __init__(
        self,
        plan: AnalysisPlan | None = None,
        insight: Insight | None = None,
        *,
        plan_error: Exception | None = None,
        insight_error: Exception | None = None,
    ) -> None:
        super().__init__(Settings(openai_api_key="sk-test"), client=object())
        self._plan = plan
        self._insight = insight or Insight(
            answer="Computed.", follow_up_questions=["What else?"]
        )
        self._plan_error = plan_error
        self._insight_error = insight_error
        self.calls = 0

    def complete_structured(self, messages, schema, **kwargs):  # type: ignore[override]
        self.calls += 1
        usage = LLMUsage(model="fake", prompt_tokens=80, completion_tokens=20)
        if schema is AnalysisPlan:
            if self._plan_error:
                raise self._plan_error
            return StructuredResponse(data=self._plan, usage=usage)
        if self._insight_error:
            raise self._insight_error
        return StructuredResponse(data=self._insight, usage=usage)


class NoKeyLLM(LLMService):
    def __init__(self) -> None:
        super().__init__(Settings(openai_api_key=""))


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #

def test_a_ranking_question_is_answered_end_to_end(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="revenue", dimensions=["region"],
            aggregation="sum", visualization="bar",
            steps=["Aggregate revenue by region", "Rank regions"],
        ),
        Insight(
            answer="Canal generated the most revenue at 11,000.00.",
            observations=["Canal leads the three regions."],
            follow_up_questions=["How does revenue vary by contract type?"],
        ),
    )
    outcome = answer_question("Which region generated the most revenue?", telecom, llm=llm)

    assert outcome.stage is Stage.COMPLETE
    assert outcome.answered
    assert outcome.is_grounded
    # Canal: 1800 + 4100 + 3000 + 2100 = 11000
    assert outcome.result.summary_data["Top region"] == "Canal"
    assert outcome.result.summary_data["Total revenue (Canal)"] == pytest.approx(11000.0)
    assert outcome.result.chart_spec["chart_type"] == "bar"
    assert outcome.steps == ["Aggregate revenue by region", "Rank regions"]
    assert outcome.follow_ups
    # Exactly two LLM calls for a clean answer: one to plan, one to explain.
    # The optional semantic check is skipped when nothing looks doubtful.
    assert llm.calls == 2
    assert outcome.llm_calls == 2
    assert outcome.usage.total_tokens == 200


def test_a_trend_question_is_answered(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.TREND, metric="revenue", time_column="signup_date",
            time_granularity="monthly", aggregation="sum", visualization="line",
        )
    )
    outcome = answer_question("Show monthly revenue over time.", telecom, llm=llm)

    assert outcome.answered
    assert outcome.result.summary_data["Periods covered"] == 6
    assert outcome.result.chart_spec["chart_type"] == "line"
    assert outcome.result.metadata["result_shape"] == "time_series"


def test_a_count_question_is_answered(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(intent=Intent.COUNT, dimensions=["region"])
    )
    outcome = answer_question("How many customers in each region?", telecom, llm=llm)

    assert outcome.answered
    assert outcome.result.summary_data["Rows counted"] == 12


def test_a_correlation_question_is_answered(telecom: pd.DataFrame):
    llm = ScriptedLLM(AnalysisPlan(intent=Intent.CORRELATION, metric="revenue"))
    outcome = answer_question(
        "Which variables correlate with revenue?", telecom, llm=llm
    )

    assert outcome.answered
    assert "revenue" in outcome.result.summary_data["Strongest pair"]
    assert outcome.result.chart_spec == {"chart_type": "heatmap"}


def test_a_filtered_question_is_answered(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
            filters=[PlanFilter(column="region", operator="eq", value="Cairo")],
        )
    )
    outcome = answer_question(
        "What is the average revenue for customers in Cairo?", telecom, llm=llm
    )

    assert outcome.answered
    assert outcome.result.summary_data["Rows"] == 4
    assert outcome.result.metadata["rows_after_filter"] == 4


def test_the_validated_plan_is_what_gets_displayed(telecom: pd.DataFrame):
    """The user must see the plan that ran, not the planner's raw guess."""
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="sales", dimensions=["area"],
            aggregation="geomean", limit=9_999,
        )
    )
    outcome = answer_question("Which area made the most sales?", telecom, llm=llm)

    assert outcome.answered
    assert outcome.plan.metric == "revenue"      # resolved
    assert outcome.plan.dimensions == ["region"]  # resolved
    assert outcome.plan.aggregation == "mean"     # repaired
    assert outcome.plan.limit == 100              # clamped
    assert len(outcome.warnings) >= 3


# --------------------------------------------------------------------------- #
# Numbers always come from Python
# --------------------------------------------------------------------------- #

def test_an_invented_number_is_rejected_not_shown(telecom: pd.DataFrame):
    """Phase 4 blocks an ungrounded answer instead of flagging it.

    The explanation is regenerated; when the regeneration is no better, the
    question is reported as unanswered rather than shown with a caveat. The
    computed figures remain available either way.
    """
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="revenue", dimensions=["region"],
            aggregation="sum",
        ),
        Insight(answer="Canal generated 999,999,999 in revenue, up 42.7%."),
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    # Python's figures are intact and visible...
    assert outcome.computed
    assert outcome.result.summary_data["Total revenue (Canal)"] == pytest.approx(11000.0)
    # ...but the answer is withheld, and the retry was attempted.
    assert not outcome.answered
    assert outcome.stage is Stage.VERIFICATION
    assert outcome.retries == MAX_RETRIES
    assert EXHAUSTED_MESSAGE in outcome.error
    assert "999,999,999" in " ".join(outcome.verification.issues)
    assert outcome.verification.retry_stage is RetryStage.INTERPRETATION


def test_the_computed_answer_is_independent_of_what_the_model_says(
    telecom: pd.DataFrame,
):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.COMPARISON, metric="monthly_charge",
            dimensions=["contract_type"], aggregation="mean",
        ),
        Insight(answer="Nonsense that mentions no figures at all."),
    )
    outcome = answer_question("Compare charges by contract.", telecom, llm=llm)

    # Month-to-month: (100 + 340 + 260 + 130) / 4 = 207.5
    frame = pd.DataFrame(outcome.result.table_data)
    row = frame[frame["contract_type"] == "Month-to-month"].iloc[0]
    assert row["Average monthly charge"] == pytest.approx(207.5)


# --------------------------------------------------------------------------- #
# Clarification and refusal
# --------------------------------------------------------------------------- #

def test_an_ambiguous_question_asks_for_clarification(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING,
            requires_clarification=True,
            clarification_question="Should I rank customers by revenue or by churn?",
        )
    )
    outcome = answer_question("Which customers are best?", telecom, llm=llm)

    assert outcome.stage is Stage.VALIDATION
    assert outcome.needs_clarification
    assert not outcome.answered
    assert "revenue or by churn" in outcome.clarification_question
    # No explanation call is made when nothing was computed.
    assert llm.calls == 1


def test_a_nonexistent_column_asks_rather_than_guessing(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="profit_margin",
            dimensions=["region"], aggregation="sum",
        )
    )
    outcome = answer_question("Which region has the best margin?", telecom, llm=llm)

    assert outcome.needs_clarification
    assert "profit_margin" in outcome.clarification_question
    assert outcome.result is None


def test_a_trend_without_a_date_column_is_refused_clearly():
    undated = pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]})
    llm = ScriptedLLM(
        AnalysisPlan(intent=Intent.TREND, metric="revenue", aggregation="sum")
    )
    outcome = answer_question("Show revenue over time.", undated, llm=llm)

    assert outcome.failed
    assert outcome.stage is Stage.VALIDATION
    assert "does not contain a valid date field" in outcome.error


def test_a_filter_matching_nothing_is_refused_with_real_values(
    telecom: pd.DataFrame,
):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
            filters=[PlanFilter(column="region", operator="eq", value="Dubai")],
        )
    )
    outcome = answer_question("Average revenue in Dubai?", telecom, llm=llm)

    assert outcome.failed
    assert "no rows match" in outcome.error
    assert "Cairo" in outcome.error  # tells the user what exists


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #

def test_a_missing_api_key_stops_the_pipeline_cleanly(telecom: pd.DataFrame):
    outcome = answer_question("Which region earns most?", telecom, llm=NoKeyLLM())

    assert outcome.stage is Stage.UNAVAILABLE
    assert outcome.failed
    assert "requires an OpenAI API key" in outcome.error
    assert outcome.result is None


def test_a_planning_timeout_is_reported(telecom: pd.DataFrame):
    llm = ScriptedLLM(plan_error=LLMTimeoutError("The AI request timed out after 45s."))
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.stage is Stage.PLANNING
    assert outcome.failed
    assert "timed out" in outcome.error


def test_an_explanation_failure_still_returns_the_computed_answer(
    telecom: pd.DataFrame,
):
    """Python computed the answer; losing the prose must not lose the result."""
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="revenue", dimensions=["region"],
            aggregation="sum",
        ),
        insight_error=LLMTimeoutError("The AI request timed out."),
    )
    outcome = answer_question("Which region earns most?", telecom, llm=llm)

    assert outcome.stage is Stage.EXPLANATION
    assert outcome.answered          # the figures are there
    assert outcome.insight is None   # the prose is not
    assert outcome.result.summary_data["Top region"] == "Canal"
    assert any("explanation could not be generated" in w for w in outcome.warnings)


def test_an_empty_question_is_refused_without_an_api_call(telecom: pd.DataFrame):
    llm = ScriptedLLM(AnalysisPlan(intent=Intent.SUMMARY))
    outcome = answer_question("   ", telecom, llm=llm)

    assert outcome.failed
    assert "enter a question" in outcome.error
    assert llm.calls == 0


def test_an_empty_dataset_is_refused(telecom: pd.DataFrame):
    llm = ScriptedLLM(AnalysisPlan(intent=Intent.SUMMARY, metric="revenue"))
    outcome = answer_question("Summarise revenue.", telecom.head(0), llm=llm)

    assert outcome.failed
    assert "no rows" in outcome.error


def test_explanation_can_be_skipped(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="revenue", dimensions=["region"],
            aggregation="sum",
        )
    )
    outcome = answer_question(
        "Which region earns most?", telecom, llm=llm, explain=False
    )

    assert outcome.answered
    assert outcome.insight is None
    assert llm.calls == 1  # planning only


def test_no_stack_trace_ever_reaches_the_user(telecom: pd.DataFrame):
    for llm in (
        ScriptedLLM(plan_error=RuntimeError("internal explosion")),
        ScriptedLLM(
            AnalysisPlan(intent=Intent.TREND, metric="revenue"),
        ),
        NoKeyLLM(),
    ):
        outcome = answer_question("anything", telecom, llm=llm)
        message = outcome.error or outcome.clarification_question or ""
        assert "Traceback" not in message
        assert ".py" not in message


# --------------------------------------------------------------------------- #
# The outcome object
# --------------------------------------------------------------------------- #

def test_the_outcome_is_serialisable_for_logging(telecom: pd.DataFrame):
    llm = ScriptedLLM(
        AnalysisPlan(
            intent=Intent.RANKING, metric="revenue", dimensions=["region"],
            aggregation="sum",
        )
    )
    payload = answer_question("Which region?", telecom, llm=llm).to_dict()

    assert payload["answered"] is True
    assert payload["intent"] == "ranking"
    assert payload["metric"] == "revenue"
    assert payload["tools"] == ["rank_values"]
    assert payload["grounded"] is True
    assert payload["tokens"] == 200
    assert payload["elapsed_seconds"] >= 0


def test_the_outcome_records_the_context_that_was_sent(telecom: pd.DataFrame):
    llm = ScriptedLLM(AnalysisPlan(intent=Intent.DATASET_QUESTION))
    outcome = answer_question("What is in this dataset?", telecom, llm=llm)

    assert outcome.context is not None
    assert outcome.context.row_count == 12
    assert outcome.context.approx_tokens() > 0
