"""Tests for the result validator and the retry routing it drives.

The validator's job is to catch an analysis that ran cleanly but answered the
wrong question. Each test constructs that specific mismatch and asserts both
the verdict and the stage the failure points at — because the stage is what
makes the retry cheap rather than a blind re-run.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from app.models.plans import AnalysisPlan, Intent, PlanFilter, SortDirection
from app.models.results import AnalysisResult, Insight
from app.models.validation import (
    EvidenceStrength,
    RetryStage,
    SemanticCheck,
    Severity,
    ValidationIssue,
)
from app.services.analysis_executor import execute_plan
from app.services.plan_validator import validate_plan
from app.services.result_validator import (
    MIN_ROWS_FOR_CONFIDENCE,
    check_semantics,
    merge_finding,
    validate_result,
)


@pytest.fixture
def shop() -> pd.DataFrame:
    """Enough rows to clear the sufficiency thresholds."""
    regions = ["North", "South", "East", "West"]
    rows = []
    for index in range(80):
        rows.append(
            {
                "order_id": f"O{index:03d}",
                "region": regions[index % 4],
                "ordered_on": pd.Timestamp("2024-01-01")
                + pd.Timedelta(days=index * 3),
                "revenue": 100.0 + index * 5,
                "units": 1 + index % 4,
            }
        )
    return pd.DataFrame(rows)


def _ran(df: pd.DataFrame, **plan_fields) -> tuple[AnalysisPlan, AnalysisResult]:
    """A validated plan and the result it actually produced."""
    plan_fields.setdefault("intent", Intent.RANKING)
    validated = validate_plan(df, AnalysisPlan(**plan_fields))
    assert validated.ok, validated.message or validated.clarification_question
    return validated.plan, execute_plan(df, validated.plan)


# --------------------------------------------------------------------------- #
# A valid result
# --------------------------------------------------------------------------- #

def test_a_good_result_validates(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    insight = Insight(answer="West led the four regions.")

    outcome = validate_result(shop, plan, result, insight)

    assert outcome.valid
    assert outcome.confidence == 1.0
    assert outcome.issues == []
    assert outcome.retry_recommended is False
    assert outcome.retry_stage is RetryStage.NONE
    assert outcome.strength is EvidenceStrength.STRONG
    # Enough checks ran that "valid" means something.
    assert outcome.checks_run >= 10


def test_validation_runs_without_an_explanation(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    outcome = validate_result(shop, plan, result, None)
    assert outcome.valid


def test_the_verdict_is_serialisable(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    payload = validate_result(shop, plan, result).to_dict()

    assert payload["valid"] is True
    assert payload["strength"] == "strong"
    assert isinstance(payload["findings"], list)


# --------------------------------------------------------------------------- #
# Execution and emptiness
# --------------------------------------------------------------------------- #

def test_a_failed_analysis_is_invalid(shop: pd.DataFrame):
    plan, _ = _ran(shop, metric="revenue", dimensions=["region"], aggregation="sum")

    outcome = validate_result(
        shop, plan, AnalysisResult.failure("no rows matched the filters")
    )

    assert not outcome.valid
    assert outcome.retry_stage is RetryStage.EXECUTION
    assert "no rows matched" in outcome.issues[0]
    # No point running the remaining checks on a result that does not exist.
    assert outcome.checks_run == 1


def test_an_empty_result_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    empty = result.model_copy(update={"summary_data": {}, "table_data": []})

    outcome = validate_result(shop, plan, empty)

    assert not outcome.valid
    assert outcome.has("empty_result")
    assert outcome.retry_stage is RetryStage.PLANNING


def test_headline_figures_without_rows_warn_but_pass(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    thin = result.model_copy(update={"table_data": []})

    outcome = validate_result(shop, plan, thin)

    assert outcome.valid
    assert outcome.has("empty_table")


# --------------------------------------------------------------------------- #
# Columns, grouping, filters
# --------------------------------------------------------------------------- #

def test_a_plan_naming_a_missing_column_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    # A plan that slipped past planning with a column that is not there.
    broken = plan.model_copy(update={"metric": "profit"})

    outcome = validate_result(shop, broken, result)

    assert not outcome.valid
    assert outcome.has("column_missing")
    assert outcome.retry_stage is RetryStage.PLANNING


def test_a_result_grouped_by_the_wrong_field_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    # The plan asked for one field; the result reflects another.
    mismatched = plan.model_copy(update={"dimensions": ["units"]})

    outcome = validate_result(shop, mismatched, result)

    assert not outcome.valid
    assert outcome.has("grouping_not_applied")
    assert outcome.retry_stage is RetryStage.PLANNING


def test_a_summary_may_ignore_a_dimension(shop: pd.DataFrame):
    plan, result = _ran(shop, intent=Intent.SUMMARY, metric="revenue")
    with_dimension = plan.model_copy(update={"dimensions": ["region"]})

    # A summary legitimately does not group, so this is not a failure.
    assert validate_result(shop, with_dimension, result).valid


def test_filters_in_the_plan_must_appear_in_the_result(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    with_filter = plan.model_copy(
        update={
            "filters": [PlanFilter(column="region", operator="eq", value="North")]
        }
    )

    outcome = validate_result(shop, with_filter, result)

    assert not outcome.valid
    assert outcome.has("filters_not_applied")
    assert outcome.retry_stage is RetryStage.EXECUTION


def test_an_applied_filter_passes(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum",
        filters=[PlanFilter(column="region", operator="eq", value="North")],
    )
    outcome = validate_result(shop, plan, result)

    assert outcome.valid
    assert result.metadata["rows_after_filter"] == 20


def test_a_filter_that_removed_everything_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum",
        filters=[PlanFilter(column="region", operator="eq", value="North")],
    )
    emptied = result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "rows_before_filter": 80,
                "rows_after_filter": 0,
            }
        }
    )
    outcome = validate_result(shop, plan, emptied)

    assert not outcome.valid
    assert outcome.has("filter_removed_everything")


# --------------------------------------------------------------------------- #
# Shape, ordering, time window
# --------------------------------------------------------------------------- #

def test_a_shape_that_does_not_match_the_intent_warns(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    as_trend = plan.model_copy(update={"intent": Intent.TREND})

    outcome = validate_result(shop, as_trend, result)
    assert outcome.has("shape_mismatch")


def test_a_descending_ranking_must_be_descending(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum",
        sort_direction=SortDirection.DESCENDING,
    )
    # Reverse the rows without telling the plan.
    reversed_rows = result.model_copy(
        update={"table_data": list(reversed(result.table_data))}
    )

    outcome = validate_result(shop, plan, reversed_rows)

    assert outcome.has("ranking_order_wrong")
    assert any(
        f.stage is RetryStage.EXECUTION for f in outcome.findings
        if f.code == "ranking_order_wrong"
    )


def test_an_ascending_ranking_is_accepted(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum",
        sort_direction=SortDirection.ASCENDING,
    )
    assert not validate_result(shop, plan, result).has("ranking_order_wrong")


def test_a_different_time_column_than_planned_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, intent=Intent.TREND, metric="revenue", time_column="ordered_on",
        time_granularity="monthly", aggregation="sum",
    )
    # Pretend the executor used a different date field.
    swapped = result.model_copy(
        update={"metadata": {**result.metadata, "time_column": "something_else"}}
    )
    outcome = validate_result(shop, plan, swapped)

    assert not outcome.valid
    assert outcome.has("wrong_time_column")
    assert outcome.retry_stage is RetryStage.PLANNING


def test_a_different_granularity_warns(shop: pd.DataFrame):
    plan, result = _ran(
        shop, intent=Intent.TREND, metric="revenue", time_column="ordered_on",
        time_granularity="monthly", aggregation="sum",
    )
    swapped = result.model_copy(
        update={"metadata": {**result.metadata, "granularity": "yearly"}}
    )
    assert validate_result(shop, plan, swapped).has("wrong_granularity")


def test_a_requested_window_that_was_not_applied_warns(shop: pd.DataFrame):
    plan, result = _ran(
        shop, intent=Intent.TREND, metric="revenue", time_column="ordered_on",
        time_granularity="monthly", aggregation="sum", periods=2,
    )
    ignored = result.model_copy(
        update={"metadata": {**result.metadata, "periods": 8}}
    )
    assert validate_result(shop, plan, ignored).has("time_window_not_applied")


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

def test_a_non_finite_headline_figure_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    broken = result.model_copy(
        update={
            "summary_data": {**result.summary_data, "Total revenue": float("nan")}
        }
    )
    outcome = validate_result(shop, plan, broken)

    assert not outcome.valid
    assert outcome.has("non_finite_values")
    assert outcome.retry_stage is RetryStage.EXECUTION


def test_non_finite_table_cells_warn(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    rows = [dict(row) for row in result.table_data]
    rows[0][list(rows[0])[-1]] = math.inf
    outcome = validate_result(shop, plan, result.model_copy(update={"table_data": rows}))

    assert outcome.has("non_finite_cells")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def test_a_chart_on_a_missing_column_warns_and_points_at_charting(
    shop: pd.DataFrame,
):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    broken = result.model_copy(
        update={"chart_spec": {"chart_type": "bar", "x": "nope", "y": "revenue"}}
    )
    outcome = validate_result(shop, plan, broken)

    assert outcome.has("chart_column_missing")
    assert any(
        f.stage is RetryStage.CHARTING for f in outcome.findings
    )


def test_an_unbuildable_chart_warns(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    # A scatter needs two numeric axes; region is categorical.
    broken = result.model_copy(
        update={"chart_spec": {"chart_type": "scatter", "x": "region", "y": "revenue"}}
    )
    outcome = validate_result(shop, plan, broken)

    assert outcome.has("chart_unbuildable")


def test_no_chart_is_not_a_problem(shop: pd.DataFrame):
    plan, result = _ran(shop, intent=Intent.DATASET_QUESTION)
    outcome = validate_result(shop, plan, result)

    assert result.chart_spec is None
    assert outcome.valid


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #

def test_an_explanation_with_invented_numbers_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    insight = Insight(answer="West generated 987,654,321 in revenue.")

    outcome = validate_result(shop, plan, result, insight)

    assert not outcome.valid
    assert outcome.has("ungrounded_numbers")
    assert outcome.retry_stage is RetryStage.INTERPRETATION
    assert "987,654,321" in outcome.issues[0]
    # The finding is recorded on the insight too, for the UI.
    assert insight.ungrounded_numbers == ["987,654,321"]


def test_an_explanation_quoting_computed_numbers_is_valid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    total = result.summary_data[
        next(k for k in result.summary_data if k.startswith("Total revenue"))
    ]
    insight = Insight(answer=f"The leading region totalled {total:,.2f}.")

    assert validate_result(shop, plan, result, insight).valid


def test_a_blank_explanation_is_invalid(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    outcome = validate_result(shop, plan, result, Insight(answer="   "))

    assert not outcome.valid
    assert outcome.has("empty_explanation")
    assert outcome.retry_stage is RetryStage.INTERPRETATION


# --------------------------------------------------------------------------- #
# Sufficiency and strength
# --------------------------------------------------------------------------- #

def test_a_tiny_sample_is_flagged_and_limits_the_strength():
    tiny = pd.DataFrame(
        {"region": ["A", "B", "A"], "revenue": [1.0, 2.0, 3.0]}
    )
    plan, result = _ran(
        tiny, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    outcome = validate_result(tiny, plan, result)

    assert outcome.valid  # thin, not wrong
    assert outcome.has("very_few_rows")
    assert outcome.strength is EvidenceStrength.LIMITED
    assert any("3 record" in w for w in outcome.warnings)


def test_a_small_sample_is_a_note_not_a_warning():
    small = pd.DataFrame(
        {
            "region": ["A", "B"] * 10,
            "revenue": [float(i) for i in range(20)],
        }
    )
    plan, result = _ran(
        small, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    outcome = validate_result(small, plan, result)

    assert outcome.valid
    assert outcome.has("few_rows")
    assert outcome.strength is EvidenceStrength.MODERATE
    assert 20 < MIN_ROWS_FOR_CONFIDENCE


def test_a_warning_lowers_the_strength_to_moderate(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    with_bad_chart = result.model_copy(
        update={"chart_spec": {"chart_type": "bar", "x": "nope"}}
    )
    outcome = validate_result(shop, plan, with_bad_chart)

    assert outcome.valid
    assert outcome.strength is EvidenceStrength.MODERATE


def test_confidence_falls_with_each_finding(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    clean = validate_result(shop, plan, result)
    broken = validate_result(
        shop, plan, result, Insight(answer="Revenue was 42,424,242.")
    )

    assert clean.confidence > broken.confidence
    assert 0.0 <= broken.confidence <= 1.0


# --------------------------------------------------------------------------- #
# Retry routing
# --------------------------------------------------------------------------- #

def test_the_cheapest_fixable_stage_is_chosen(shop: pd.DataFrame):
    """A bad explanation and a bad plan together should retry the explanation.

    Regenerating prose costs one call; re-planning costs a call and may change
    the answer. The cheaper fix is tried first.
    """
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    both = plan.model_copy(update={"dimensions": ["units"]})  # planning failure
    insight = Insight(answer="Revenue was 42,424,242.")       # interpretation failure

    outcome = validate_result(shop, both, result, insight)

    assert not outcome.valid
    assert outcome.retry_stage is RetryStage.INTERPRETATION


def test_no_retry_is_recommended_when_nothing_can_fix_it(shop: pd.DataFrame):
    plan, _ = _ran(shop, metric="revenue", dimensions=["region"], aggregation="sum")
    outcome = validate_result(
        shop, plan, AnalysisResult(success=True, summary_data={}, table_data=[])
    )
    # An empty result points at planning, which is retryable.
    assert outcome.retry_recommended


# --------------------------------------------------------------------------- #
# The optional semantic check
# --------------------------------------------------------------------------- #

class _SemanticLLM:
    """Returns a scripted SemanticCheck, or raises."""

    def __init__(self, check=None, error: Exception | None = None) -> None:
        self._check = check
        self._error = error
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def complete_structured(self, messages, schema, **kwargs):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return SimpleNamespace(data=self._check, usage=SimpleNamespace())


def test_a_clean_semantic_check_returns_nothing(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    llm = _SemanticLLM(
        SemanticCheck(addresses_question=True, contradicts_data=False)
    )
    finding = check_semantics(llm, "Which region?", result, Insight(answer="West."))

    assert finding is None
    assert llm.calls == 1


def test_a_contradiction_becomes_a_blocking_finding(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    llm = _SemanticLLM(
        SemanticCheck(
            addresses_question=True, contradicts_data=True,
            reason="says North led when the figures show West",
        )
    )
    finding = check_semantics(llm, "Which region?", result, Insight(answer="North."))

    assert finding is not None
    assert finding.code == "explanation_contradicts_data"
    assert finding.severity is Severity.ERROR
    assert finding.stage is RetryStage.INTERPRETATION
    assert "North led" in finding.message


def test_an_off_topic_explanation_becomes_a_finding(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    llm = _SemanticLLM(
        SemanticCheck(
            addresses_question=False, contradicts_data=False,
            reason="describes units, not revenue",
        )
    )
    finding = check_semantics(llm, "Which region?", result, Insight(answer="..."))

    assert finding is not None
    assert finding.code == "explanation_off_topic"


def test_a_failing_semantic_check_never_blocks_an_answer(shop: pd.DataFrame):
    """The validator being unavailable must not reject a good answer."""
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    llm = _SemanticLLM(error=RuntimeError("the validator exploded"))

    assert check_semantics(llm, "Which region?", result, Insight(answer="West.")) is None


def test_merging_a_finding_invalidates_the_verdict(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    clean = validate_result(shop, plan, result, Insight(answer="West led."))
    assert clean.valid

    merged = merge_finding(
        clean,
        ValidationIssue(
            code="explanation_off_topic", severity=Severity.ERROR,
            message="answers a different question",
            stage=RetryStage.INTERPRETATION,
        ),
    )

    assert not merged.valid
    assert merged.retry_stage is RetryStage.INTERPRETATION
    assert merged.strength is EvidenceStrength.LIMITED
    assert merged.checks_run == clean.checks_run + 1
    assert merged.confidence < clean.confidence


def test_merging_nothing_is_a_no_op(shop: pd.DataFrame):
    plan, result = _ran(
        shop, metric="revenue", dimensions=["region"], aggregation="sum"
    )
    clean = validate_result(shop, plan, result)
    assert merge_finding(clean, None) is clean
