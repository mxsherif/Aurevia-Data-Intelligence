"""Tests for the planner, the insight agent, grounding and suggestions.

The LLM is always a stub. What is tested is our side of the contract: that the
planner sends a compact schema and no data, that an LLM failure degrades
cleanly, that the insight agent's numbers are checked against the computed
result, and that suggestions need no API call at all.
"""

from __future__ import annotations

import json


import pandas as pd
import pytest

from app.agents.insights import InsightAgent, SYSTEM_PROMPT as INSIGHT_PROMPT
from app.agents.planner import (
    MAX_QUESTION_LENGTH,
    PlannerAgent,
    SYSTEM_PROMPT as PLANNER_PROMPT,
)
from app.agents.suggestions import detect_roles, suggest_questions
from app.config import Settings
from app.models.plans import AnalysisPlan, Intent
from app.models.results import AnalysisResult, Insight
from app.services.dataset_context import build_dataset_context
from app.services.grounding import check_text, extract_numbers, verify_insight
from app.services.llm_service import (
    LLMRateLimitError,
    LLMService,
    LLMTimeoutError,
    LLMUnavailableError,
    StructuredResponse,
    LLMUsage,
)


@pytest.fixture
def telecom() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(10)],
            "region": ["Cairo", "Delta"] * 5,
            "signup_date": pd.date_range("2023-01-01", periods=10, freq="ME"),
            "contract_type": ["Month-to-month", "One year"] * 5,
            "monthly_charge": [100.0, 200.0, 150.0, 300.0, 120.0,
                               250.0, 180.0, 220.0, 160.0, 280.0],
            "revenue": [1000.0, 2000.0, 1500.0, 3000.0, 1200.0,
                        2500.0, 1800.0, 2200.0, 1600.0, 2800.0],
            "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        }
    )


class FakeLLM(LLMService):
    """An LLMService whose single call is scripted."""

    def __init__(self, *responses, error: Exception | None = None) -> None:
        super().__init__(Settings(openai_api_key="sk-test"), client=object())
        self._responses = list(responses)
        self._error = error
        self.messages: list[list[dict]] = []
        self.schemas: list[type] = []

    def complete_structured(self, messages, schema, **kwargs):  # type: ignore[override]
        self.messages.append(list(messages))
        self.schemas.append(schema)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("FakeLLM was called more times than scripted")
        return StructuredResponse(
            data=self._responses.pop(0),
            usage=LLMUsage(model="fake-model", prompt_tokens=100, completion_tokens=40),
        )


class UnavailableLLM(LLMService):
    def __init__(self) -> None:
        super().__init__(Settings(openai_api_key=""))


# --------------------------------------------------------------------------- #
# The planner
# --------------------------------------------------------------------------- #

def _sample_plan(**overrides) -> AnalysisPlan:
    payload = {
        "intent": Intent.RANKING,
        "metric": "revenue",
        "dimensions": ["region"],
        "aggregation": "sum",
        "visualization": "bar",
        "steps": ["Aggregate revenue by region", "Rank regions"],
    }
    payload.update(overrides)
    return AnalysisPlan(**payload)


def test_the_planner_returns_a_validated_plan(telecom: pd.DataFrame):
    llm = FakeLLM(_sample_plan())
    outcome = PlannerAgent(llm).plan("Which region earns most?", telecom)

    assert outcome.ok
    assert outcome.plan.intent is Intent.RANKING
    assert outcome.plan.metric == "revenue"
    assert outcome.usage.total_tokens == 140
    assert llm.schemas == [AnalysisPlan]


def test_the_planner_asks_for_structured_output(telecom: pd.DataFrame):
    llm = FakeLLM(_sample_plan())
    PlannerAgent(llm).plan("anything", telecom)
    assert llm.schemas[0] is AnalysisPlan


def test_the_planner_sends_the_schema_but_never_the_data(telecom: pd.DataFrame):
    llm = FakeLLM(_sample_plan())
    PlannerAgent(llm).plan("Which region earns most?", telecom)

    prompt = "\n".join(m["content"] for m in llm.messages[0])

    # Column names and roles are present...
    assert "revenue" in prompt and "region" in prompt
    assert '"roles"' in prompt
    # ...identifier values are not, and neither is any row.
    assert "C000" not in prompt
    assert "C009" not in prompt

    context = build_dataset_context(telecom, name="telecom").to_dict()
    assert set(context) == {"dataset", "rows", "columns", "roles", "notes"}
    # "rows" is a count, never a payload.
    assert context["rows"] == 10
    # Each column carries a bounded description, not its values.
    allowed_keys = {
        "name", "type", "missing_pct", "range", "mean",
        "distinct", "examples", "covers",
    }
    for column in context["columns"]:
        assert set(column) <= allowed_keys
        # Category examples are capped; numeric columns list no values at all.
        assert len(column.get("examples", [])) <= 8


def test_the_context_size_barely_grows_with_the_dataset(telecom: pd.DataFrame):
    small = build_dataset_context(telecom, name="t")
    large = build_dataset_context(
        pd.concat([telecom] * 500, ignore_index=True), name="t"
    )

    assert large.row_count == 5_000
    # 500x the rows must not mean 500x the prompt.
    assert large.approx_tokens() < small.approx_tokens() * 1.2


def test_the_planner_prompt_forbids_computing(telecom: pd.DataFrame):
    assert "do NOT compute" in PLANNER_PROMPT.replace("You do NOT", "you do NOT")
    assert "Python executes the plan" in PLANNER_PROMPT


def test_the_planner_advertises_only_real_vocabulary(telecom: pd.DataFrame):
    llm = FakeLLM(_sample_plan())
    PlannerAgent(llm).plan("q", telecom)
    prompt = "\n".join(m["content"] for m in llm.messages[0])

    from app.tools.aggregations import supported_aggregations
    from app.tools.charts import supported_chart_types

    for name in supported_aggregations():
        assert name in prompt
    for chart in supported_chart_types():
        assert chart in prompt


def test_the_dataset_context_stays_small(telecom: pd.DataFrame):
    context = build_dataset_context(telecom, name="telecom")
    # A compact schema, not a data dump.
    assert context.approx_tokens() < 1_000
    assert context.row_count == 10
    assert "revenue" in context.column_names()


def test_the_planner_reuses_a_supplied_context(telecom: pd.DataFrame):
    context = build_dataset_context(telecom, name="telecom")
    llm = FakeLLM(_sample_plan())
    outcome = PlannerAgent(llm).plan("q", telecom, context=context)
    assert outcome.context is context


def test_an_empty_question_is_refused_without_calling_the_model(
    telecom: pd.DataFrame,
):
    llm = FakeLLM()
    outcome = PlannerAgent(llm).plan("   ", telecom)

    assert not outcome.ok
    assert "enter a question" in outcome.error
    assert llm.messages == []


def test_an_overlong_question_is_refused(telecom: pd.DataFrame):
    llm = FakeLLM()
    outcome = PlannerAgent(llm).plan("x" * (MAX_QUESTION_LENGTH + 1), telecom)

    assert not outcome.ok
    assert str(MAX_QUESTION_LENGTH) in outcome.error
    assert llm.messages == []


def test_the_planner_reports_a_missing_api_key(telecom: pd.DataFrame):
    outcome = PlannerAgent(UnavailableLLM()).plan("Which region earns most?", telecom)

    assert not outcome.ok
    assert "requires an OpenAI API key" in outcome.error


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (LLMTimeoutError("The AI request timed out after 45s."), "timed out"),
        (LLMRateLimitError("The OpenAI rate limit was reached."), "rate limit"),
        (LLMUnavailableError("no key"), "no key"),
    ],
)
def test_llm_failures_become_planner_errors(
    telecom: pd.DataFrame, error, fragment
):
    outcome = PlannerAgent(FakeLLM(error=error)).plan("q", telecom)

    assert not outcome.ok
    assert outcome.plan is None
    assert fragment in outcome.error


def test_a_planner_clarification_is_surfaced(telecom: pd.DataFrame):
    plan = _sample_plan(
        requires_clarification=True,
        clarification_question="Rank by revenue or by satisfaction?",
    )
    outcome = PlannerAgent(FakeLLM(plan)).plan("Which customers are best?", telecom)

    assert outcome.ok
    assert outcome.needs_clarification


# --------------------------------------------------------------------------- #
# The insight agent
# --------------------------------------------------------------------------- #

@pytest.fixture
def ranking_result() -> AnalysisResult:
    return AnalysisResult(
        success=True,
        title="Total revenue by region",
        summary_data={
            "Top region": "Cairo",
            "Total revenue (Cairo)": 15556354.13,
            "Share of the listed total": 34.41,
        },
        table_data=[
            {"region": "Cairo", "Total revenue": 15556354.13},
            {"region": "Delta", "Total revenue": 8054012.36},
        ],
        metadata={"tools": ["rank_values"], "limit": 10},
    )


def test_the_insight_agent_returns_an_answer(ranking_result: AnalysisResult):
    insight = Insight(
        answer="Cairo generated the most revenue at 15,556,354.13.",
        observations=["Cairo accounts for 34.41% of the listed total."],
        follow_up_questions=["How does revenue vary by city?"],
    )
    outcome = InsightAgent(FakeLLM(insight)).explain("Which region?", ranking_result)

    assert outcome.ok
    assert outcome.is_grounded
    assert outcome.insight.answer.startswith("Cairo")
    assert outcome.usage.total_tokens == 140


def test_the_insight_agent_receives_only_computed_values(
    ranking_result: AnalysisResult, telecom: pd.DataFrame
):
    llm = FakeLLM(Insight(answer="ok"))
    InsightAgent(llm).explain("Which region?", ranking_result)

    payload = json.loads(llm.messages[0][-1]["content"])

    assert payload["question"] == "Which region?"
    assert "computed_results" in payload
    assert payload["computed_results"]["summary"]["Top region"] == "Cairo"
    # The dataset itself is never sent to this stage.
    assert "C000" not in json.dumps(payload)


def test_the_insight_prompt_forbids_arithmetic_and_causation():
    assert "Do not calculate, infer, estimate, or invent" in INSIGHT_PROMPT
    assert "never why" in INSIGHT_PROMPT
    assert "Do not explain causes" in INSIGHT_PROMPT


def test_an_ungrounded_figure_is_flagged(ranking_result: AnalysisResult):
    insight = Insight(
        answer="Cairo generated 99,999,999 in revenue.",
        observations=["Revenue grew 12.7% year on year."],
    )
    outcome = InsightAgent(FakeLLM(insight)).explain("Which region?", ranking_result)

    assert outcome.ok  # the answer is still returned...
    assert not outcome.is_grounded  # ...but marked unverified
    assert "99,999,999" in outcome.grounding.ungrounded
    assert "12.7%" in outcome.grounding.ungrounded
    assert outcome.insight.ungrounded_numbers


def test_a_failed_result_is_not_explained():
    outcome = InsightAgent(FakeLLM()).explain(
        "q", AnalysisResult.failure("no rows matched")
    )
    assert not outcome.ok
    assert "no successful result" in outcome.error


def test_the_insight_agent_reports_a_missing_api_key(
    ranking_result: AnalysisResult,
):
    outcome = InsightAgent(UnavailableLLM()).explain("q", ranking_result)
    assert not outcome.ok
    assert "requires an OpenAI API key" in outcome.error


def test_an_llm_failure_becomes_an_insight_error(ranking_result: AnalysisResult):
    outcome = InsightAgent(
        FakeLLM(error=LLMTimeoutError("timed out"))
    ).explain("q", ranking_result)

    assert not outcome.ok
    assert "timed out" in outcome.error


# --------------------------------------------------------------------------- #
# Numerical grounding
# --------------------------------------------------------------------------- #

def test_numbers_are_extracted_with_separators_and_suffixes():
    found = dict(extract_numbers("1,240,000 and 1.24M and 36.04% and -5"))
    assert found["1,240,000"] == 1_240_000.0
    assert found["36.04%"] == 36.04
    assert found["-5"] == -5.0


def test_a_magnitude_suffix_is_read_both_ways():
    readings = [value for raw, value in extract_numbers("1.24M") if raw == "1.24M"]
    assert 1.24 in readings
    assert 1_240_000.0 in readings


def test_computed_figures_are_grounded(ranking_result: AnalysisResult):
    allowed = ranking_result.allowed_numbers()

    for text in (
        "Cairo generated 15,556,354.13.",
        "Cairo generated 15.56M.",
        "That is 34.41% of the total.",
        "That is 34.4% of the total.",
        "Delta follows at 8,054,012.36.",
    ):
        assert check_text(text, allowed).is_grounded, text


def test_invented_figures_are_not_grounded(ranking_result: AnalysisResult):
    allowed = ranking_result.allowed_numbers()

    for text in (
        "Cairo generated 20,000,000.",
        "Revenue grew by 12.7%.",
        "The average is 11,805,183.25.",
    ):
        assert not check_text(text, allowed).is_grounded, text


def test_small_ordinals_are_always_allowed(ranking_result: AnalysisResult):
    allowed = ranking_result.allowed_numbers()
    assert check_text("The top 3 regions lead the table.", allowed).is_grounded


def test_rank_references_within_the_limit_are_allowed(
    ranking_result: AnalysisResult,
):
    allowed = ranking_result.allowed_numbers()
    assert check_text("All 10 listed regions are shown.", allowed).is_grounded


def test_text_without_numbers_is_grounded(ranking_result: AnalysisResult):
    report = check_text("Cairo leads the other regions.", ranking_result.allowed_numbers())
    assert report.is_grounded
    assert report.checked_count == 0


def test_verify_insight_records_findings_on_the_insight(
    ranking_result: AnalysisResult,
):
    insight = Insight(answer="Revenue was 42,424,242.")
    report = verify_insight(insight, ranking_result)

    assert not report.is_grounded
    assert insight.ungrounded_numbers == ["42,424,242"]
    assert insight.is_grounded is False


def test_period_labels_count_as_computed():
    result = AnalysisResult(
        success=True,
        summary_data={"First period": "2021-01", "Last period": "2024-12"},
    )
    allowed = result.allowed_numbers()
    assert check_text("Revenue ran from 2021 to 2024.", allowed).is_grounded


# --------------------------------------------------------------------------- #
# Suggestions -- deterministic, no API call
# --------------------------------------------------------------------------- #

def test_suggestions_are_dataset_aware(telecom: pd.DataFrame):
    questions = suggest_questions(telecom)

    assert 4 <= len(questions) <= 6
    joined = " ".join(questions).lower()
    assert "revenue" in joined
    assert "churn" in joined


def test_suggestions_are_deterministic(telecom: pd.DataFrame):
    assert suggest_questions(telecom) == suggest_questions(telecom)


def test_suggestions_need_no_llm(telecom: pd.DataFrame, monkeypatch):
    # Any attempt to reach the LLM layer from here is a design error.
    import app.services.llm_service as llm_module

    def explode(*args, **kwargs):
        raise AssertionError("suggestions must not call the LLM")

    monkeypatch.setattr(llm_module.LLMService, "complete_structured", explode)
    assert suggest_questions(telecom)


def test_semantic_roles_are_detected(telecom: pd.DataFrame):
    roles = detect_roles(telecom)

    assert roles.semantic["revenue"] == "revenue"
    assert roles.semantic["churn"] == "churn"
    assert roles.semantic["region"] == "region"
    assert roles.semantic["contract"] == "contract_type"
    assert roles.primary_measure == "revenue"
    assert roles.primary_dimension == "region"


def test_an_unfamiliar_dataset_still_gets_suggestions():
    frame = pd.DataFrame(
        {
            "widget_ref": [f"W{i}" for i in range(20)],
            "depot": ["A", "B", "C", "D"] * 5,
            "mass_kg": [float(i) for i in range(20)],
        }
    )
    questions = suggest_questions(frame)

    assert len(questions) >= 4
    joined = " ".join(questions).lower()
    assert "depot" in joined or "mass" in joined


def test_suggestions_for_an_empty_dataset():
    assert suggest_questions(pd.DataFrame()) == ["What does this dataset contain?"]


def test_suggestions_respect_the_limit(telecom: pd.DataFrame):
    assert len(suggest_questions(telecom, limit=4)) == 4
