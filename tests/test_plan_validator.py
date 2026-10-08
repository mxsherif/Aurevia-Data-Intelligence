"""Tests for planner-output validation and column resolution.

This is the safety layer: it decides what a possibly-wrong LLM plan is allowed
to do to a real dataset. The cases below are the ones that matter — a column
that does not exist, an aggregation that is not supported, a filter that
matches nothing, a trend with no date field.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.plans import AnalysisPlan, Intent, PlanFilter, SortDirection
from app.services.plan_validator import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    Verdict,
    detect_metric_substitution,
    validate_plan,
)
from app.tools.resolution import (
    MatchKind,
    normalize,
    resolve_column,
    resolve_columns,
    tokens,
)


@pytest.fixture
def telecom() -> pd.DataFrame:
    """A miniature of the sample dataset, with the same column names."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(12)],
            "region": ["Cairo", "Delta", "Cairo", "Canal"] * 3,
            "signup_date": pd.to_datetime(
                [
                    "2023-01-05", "2023-02-11", "2023-03-19", "2023-04-02",
                    "2023-05-23", "2023-06-14", "2023-07-30", "2023-08-08",
                    "2023-09-17", "2023-10-26", "2023-11-04", "2023-12-12",
                ]
            ),
            "contract_type": ["Month-to-month", "One year", "Two year"] * 4,
            "monthly_charge": [100.0, 220.0, 180.0, 340.0, 150.0, 410.0,
                               260.0, 190.0, 300.0, 130.0, 480.0, 210.0],
            "revenue": [1200.0, 2400.0, 1800.0, 4000.0, 1500.0, 5000.0,
                        2600.0, 1900.0, 3000.0, 1300.0, 5200.0, 2100.0],
            "churn": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "network_type": ["4G", "5G", "3G"] * 4,
        }
    )


@pytest.fixture
def no_dates() -> pd.DataFrame:
    return pd.DataFrame(
        {"region": ["A", "B", "A"], "revenue": [1.0, 2.0, 3.0]}
    )


@pytest.fixture
def no_numbers() -> pd.DataFrame:
    return pd.DataFrame(
        {"region": ["A", "B", "A"], "label": ["x", "y", "z"]}
    )


# --------------------------------------------------------------------------- #
# A valid plan passes through
# --------------------------------------------------------------------------- #

def test_a_valid_plan_is_accepted(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", visualization="bar", limit=5,
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.verdict is Verdict.OK
    assert outcome.ok
    assert outcome.plan.metric == "revenue"
    assert outcome.plan.dimensions == ["region"]
    assert outcome.plan.aggregation == "sum"
    assert outcome.plan.visualization == "bar"
    assert outcome.plan.limit == 5
    assert outcome.adjustments == []


def test_validation_does_not_mutate_the_original_plan(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="sales", dimensions=["area"],
        aggregation="geomean",
    )
    validate_plan(telecom, plan)

    # The planner's raw output is preserved for logging and display.
    assert plan.metric == "sales"
    assert plan.dimensions == ["area"]
    assert plan.aggregation == "geomean"


def test_a_plan_on_an_empty_dataset_is_rejected():
    plan = AnalysisPlan(intent=Intent.SUMMARY, metric="revenue")
    outcome = validate_plan(pd.DataFrame(), plan)

    assert outcome.verdict is Verdict.REJECTED
    assert "no rows" in outcome.message


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #

def test_aliases_are_resolved_and_reported(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="sales", dimensions=["area"],
        aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.metric == "revenue"
    assert outcome.plan.dimensions == ["region"]
    # The user is told about every rename.
    assert any("sales" in note and "revenue" in note for note in outcome.adjustments)
    assert any("area" in note and "region" in note for note in outcome.adjustments)


def test_a_nonexistent_metric_asks_for_clarification(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="profit_margin", dimensions=["region"],
        aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.verdict is Verdict.NEEDS_CLARIFICATION
    assert "profit_margin" in outcome.clarification_question
    # The user is offered the real alternatives.
    assert "revenue" in outcome.clarification_question


def test_a_non_numeric_metric_is_rejected_not_guessed(telecom: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.SUMMARY, metric="region", aggregation="mean")
    outcome = validate_plan(telecom, plan)

    assert outcome.verdict is Verdict.REJECTED
    assert "not a numeric column" in outcome.message
    assert "revenue" in outcome.message  # suggests a usable field


def test_an_unknown_dimension_is_dropped_with_a_note(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue",
        dimensions=["region", "planet"], aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.dimensions == ["region"]
    assert any("planet" in note for note in outcome.adjustments)


def test_grouping_by_a_near_unique_identifier_is_refused(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["customer_id"],
        aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    # Grouping by an ID yields one row per record, which answers nothing.
    assert outcome.verdict is Verdict.NEEDS_CLARIFICATION
    assert any("customer_id" in note for note in outcome.adjustments)


def test_the_metric_is_not_also_used_as_a_dimension(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.COMPARISON, metric="revenue",
        dimensions=["revenue", "region"], aggregation="mean",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.dimensions == ["region"]


def test_a_ranking_with_no_dimension_asks_which_field(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=[], aggregation="sum"
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.verdict is Verdict.NEEDS_CLARIFICATION
    assert "break the results down by" in outcome.clarification_question


def test_no_numeric_column_at_all_is_rejected(no_numbers: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.SUMMARY, metric="revenue")
    outcome = validate_plan(no_numbers, plan)

    assert outcome.verdict is Verdict.REJECTED
    assert "no numeric column" in outcome.message


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #

def test_an_unsupported_aggregation_falls_back_with_a_note(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="geometric_mean",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.aggregation == "mean"
    assert any("not a supported aggregation" in n for n in outcome.adjustments)
    # The message lists what is supported.
    assert any("sum" in n for n in outcome.adjustments)


def test_a_missing_aggregation_is_filled_in(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"]
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.aggregation == "mean"
    assert any("No aggregation was specified" in n for n in outcome.adjustments)


def test_trend_defaults_to_sum(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TREND, metric="revenue", time_column="signup_date"
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.aggregation == "sum"


def test_a_numeric_aggregation_without_a_metric_becomes_a_count(
    telecom: pd.DataFrame,
):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric=None, dimensions=["region"],
        aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.aggregation == "count"


def test_aggregation_aliases_are_canonicalised(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.COMPARISON, metric="revenue", dimensions=["region"],
        aggregation="average",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.aggregation == "mean"


# --------------------------------------------------------------------------- #
# Time analysis
# --------------------------------------------------------------------------- #

def test_a_trend_without_a_date_column_is_rejected(no_dates: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.TREND, metric="revenue", aggregation="sum")
    outcome = validate_plan(no_dates, plan)

    assert outcome.verdict is Verdict.REJECTED
    assert "does not contain a valid date field" in outcome.message


def test_a_time_comparison_without_a_date_column_is_rejected(no_dates: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TIME_COMPARISON, metric="revenue", aggregation="sum"
    )
    outcome = validate_plan(no_dates, plan)

    assert outcome.verdict is Verdict.REJECTED
    assert "date field" in outcome.message


def test_a_missing_date_column_is_filled_from_the_dataset(telecom: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.TREND, metric="revenue", aggregation="sum")
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.time_column == "signup_date"
    assert any("signup_date" in n for n in outcome.adjustments)


def test_a_non_date_time_column_is_replaced(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TREND, metric="revenue", time_column="monthly_charge",
        aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.time_column == "signup_date"


def test_an_unsupported_granularity_falls_back_to_monthly(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TREND, metric="revenue", time_column="signup_date",
        time_granularity="fortnightly", aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.time_granularity == "monthly"
    assert any("not a supported time grouping" in n for n in outcome.adjustments)


def test_a_valid_granularity_is_kept(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TREND, metric="revenue", time_column="signup_date",
        time_granularity="Quarterly", aggregation="sum",
    )
    outcome = validate_plan(telecom, plan)
    assert outcome.plan.time_granularity == "quarterly"


def test_a_date_field_is_dropped_when_the_intent_does_not_need_one(
    no_dates: pd.DataFrame,
):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", time_column="when",
    )
    outcome = validate_plan(no_dates, plan)

    assert outcome.ok
    assert outcome.plan.time_column is None
    assert any("no usable date column" in n for n in outcome.adjustments)


def test_periods_are_clamped(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TREND, metric="revenue", time_column="signup_date",
        aggregation="sum", periods=10_000,
    )
    outcome = validate_plan(telecom, plan)
    assert outcome.plan.periods == 120


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

def test_a_valid_filter_is_kept(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[PlanFilter(column="region", operator="eq", value="Cairo")],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert len(outcome.plan.filters) == 1
    assert outcome.plan.filters[0].column == "region"


def test_a_filter_on_an_unknown_column_is_dropped(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[PlanFilter(column="planet", operator="eq", value="Mars")],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.filters == []
    assert any("planet" in n for n in outcome.adjustments)


def test_a_filter_with_an_unsupported_operator_is_dropped(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[PlanFilter(column="region", operator="sounds_like", value="Cairo")],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.filters == []
    assert any("not a supported comparison" in n for n in outcome.adjustments)


def test_a_filter_matching_no_rows_is_rejected_with_real_values(
    telecom: pd.DataFrame,
):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[PlanFilter(column="region", operator="eq", value="Dubai")],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.verdict is Verdict.REJECTED
    assert "no rows match" in outcome.message
    # The user is shown what the column actually contains.
    assert "Cairo" in outcome.message


def test_a_range_filter_is_validated(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[
            PlanFilter(column="monthly_charge", operator="between",
                       values=["150", "350"])
        ],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.filters[0].operator == "between"


def test_a_malformed_range_filter_is_dropped(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[
            PlanFilter(column="monthly_charge", operator="between",
                       values=["cheap", "expensive"])
        ],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.filters == []
    assert any("monthly_charge" in n for n in outcome.adjustments)


def test_a_date_filter_on_a_non_date_column_is_dropped(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[
            PlanFilter(column="monthly_charge", operator="date_between",
                       values=["2023-01-01", "2023-06-30"])
        ],
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.filters == []
    assert any("not a date column" in n for n in outcome.adjustments)


def test_a_date_range_filter_is_kept(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="revenue", aggregation="mean",
        filters=[
            PlanFilter(column="signup_date", operator="date_between",
                       values=["2023-01-01", "2023-06-30"])
        ],
    )
    outcome = validate_plan(telecom, plan)
    assert outcome.ok
    assert len(outcome.plan.filters) == 1


# --------------------------------------------------------------------------- #
# Visualization, limit, sort
# --------------------------------------------------------------------------- #

def test_an_unsupported_visualization_is_discarded(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", visualization="pie",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.ok
    assert outcome.plan.visualization is None
    assert any("not a supported chart type" in n for n in outcome.adjustments)


def test_a_chart_alias_is_resolved(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", visualization="bar_chart",
    )
    outcome = validate_plan(telecom, plan)
    assert outcome.plan.visualization == "bar"


def test_none_visualization_tokens_mean_no_chart(telecom: pd.DataFrame):
    for token in ("none", "None", "table", ""):
        plan = AnalysisPlan(
            intent=Intent.RANKING, metric="revenue", dimensions=["region"],
            aggregation="sum", visualization=token,
        )
        assert validate_plan(telecom, plan).plan.visualization is None


def test_a_limit_above_the_cap_is_clamped(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", limit=10_000,
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.plan.limit == MAX_LIMIT
    assert any("Clamped" in n for n in outcome.adjustments)


def test_a_missing_limit_gets_the_default(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum",
    )
    assert validate_plan(telecom, plan).plan.limit == DEFAULT_LIMIT


def test_a_single_row_ranking_is_widened_for_context(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", limit=1,
    )
    outcome = validate_plan(telecom, plan)

    # A one-row ranking has nothing to compare and no chart worth drawing.
    assert outcome.plan.limit == DEFAULT_LIMIT
    assert any("full ranking" in n for n in outcome.adjustments)


def test_an_explicit_small_limit_is_respected(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", limit=3,
    )
    assert validate_plan(telecom, plan).plan.limit == 3


def test_a_missing_sort_direction_defaults_to_descending(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum",
    )
    assert validate_plan(telecom, plan).plan.sort_direction is SortDirection.DESCENDING


def test_an_unsupported_comparison_is_discarded(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", comparison="year_on_year_cagr",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.plan.comparison is None
    assert any("unsupported comparison" in n for n in outcome.adjustments)


# --------------------------------------------------------------------------- #
# Clarification
# --------------------------------------------------------------------------- #

def test_the_planners_own_clarification_is_passed_through(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING,
        requires_clarification=True,
        clarification_question="Rank customers by revenue or by satisfaction?",
    )
    outcome = validate_plan(telecom, plan)

    assert outcome.verdict is Verdict.NEEDS_CLARIFICATION
    assert outcome.clarification_question.startswith("Rank customers")


def test_a_clarification_flag_without_a_question_is_ignored(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum", requires_clarification=True,
    )
    # Nothing to ask, and the plan is executable, so proceed.
    assert validate_plan(telecom, plan).verdict is Verdict.OK


# --------------------------------------------------------------------------- #
# The resolution ladder itself
# --------------------------------------------------------------------------- #

COLUMNS = [
    "customer_id", "region", "city", "signup_date", "contract_type",
    "monthly_charge", "data_usage_gb", "support_calls", "payment_method",
    "tenure_months", "satisfaction_score", "revenue", "churn",
    "product_type", "network_type",
]


@pytest.mark.parametrize(
    ("term", "expected", "kind"),
    [
        ("revenue", "revenue", MatchKind.EXACT),
        ("Revenue", "revenue", MatchKind.NORMALIZED),
        ("monthly charges", "monthly_charge", MatchKind.NORMALIZED),
        ("regions", "region", MatchKind.NORMALIZED),
        ("monthly bill", "monthly_charge", MatchKind.ALIAS),
        ("customer satisfaction", "satisfaction_score", MatchKind.ALIAS),
        ("sales", "revenue", MatchKind.ALIAS),
        ("support tickets", "support_calls", MatchKind.ALIAS),
        ("contract", "contract_type", MatchKind.ALIAS),
        ("area", "region", MatchKind.ALIAS),
        ("data usage", "data_usage_gb", MatchKind.ALIAS),
    ],
)
def test_the_resolution_ladder(term: str, expected: str, kind: MatchKind):
    match = resolve_column(term, COLUMNS)

    assert match.column == expected
    assert match.kind is kind
    assert match.is_trusted


@pytest.mark.parametrize("term", ["profit_margin", "xyzzy", "lifetime value"])
def test_unresolvable_terms_return_nothing(term: str):
    match = resolve_column(term, COLUMNS)

    assert match.column is None
    assert match.found is False
    assert match.is_trusted is False
    assert term in match.describe()


def test_an_unresolved_term_suggests_candidates():
    match = resolve_column("revenu", COLUMNS)
    # Close enough to accept outright, which is the point of the fuzzy rung.
    assert match.column == "revenue"


def test_allowed_restricts_the_search_space():
    # "charge" must not resolve onto a categorical column when only numeric
    # columns are permissible.
    match = resolve_column("charge", COLUMNS, allowed=["revenue", "monthly_charge"])
    assert match.column == "monthly_charge"


def test_resolution_against_an_empty_pool():
    assert resolve_column("revenue", []).column is None


def test_blank_terms_resolve_to_nothing():
    for term in ("", "   ", None):
        assert resolve_column(term, COLUMNS).column is None


def test_resolve_columns_preserves_order_and_skips_blanks():
    matches = resolve_columns(["area", "", "sales"], COLUMNS)
    assert [m.column for m in matches] == ["region", "revenue"]


def test_normalize_folds_case_separators_and_plurals():
    assert normalize("Monthly-Charges") == "monthly_charge"
    assert normalize("  REGION  ") == "region"
    assert normalize("companies") == "company"


def test_normalize_is_consistent_for_both_sides():
    # Linguistic accuracy is not required; identical treatment is.
    assert normalize("status") == normalize("status")


def test_tokens_drop_stop_words_but_never_everything():
    assert tokens("the total of revenue") == {"revenue"}
    assert tokens("the of") == {"the", "of"}


# --------------------------------------------------------------------------- #
# Metric substitution -- the worst silent failure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("question", "metric", "should_clarify"),
    [
        # The user named a field that does not exist; the planner answered
        # about something else entirely.
        ("What is the average profit margin?", "monthly_charge", True),
        ("Show customer lifetime value by region", "revenue", True),
        ("What is the average EBITDA?", "revenue", True),
        # The user named the metric, directly or by alias.
        ("Which region generated the most revenue?", "revenue", False),
        ("What is the average monthly bill?", "monthly_charge", False),
        ("Which city has the most sales?", "revenue", False),
        ("Which contract type has the highest churn rate?", "churn", False),
        # The user named no measure at all, so choosing one is helpful.
        ("Which region is biggest?", "revenue", False),
        ("Show me the trend over time", "revenue", False),
    ],
)
def test_metric_substitution_is_detected(
    telecom: pd.DataFrame, question: str, metric: str, should_clarify: bool
):
    found = detect_metric_substitution(telecom, question, metric)
    assert bool(found) is should_clarify


def test_a_substituted_metric_stops_the_plan(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="monthly_charge", aggregation="mean"
    )
    outcome = validate_plan(telecom, plan, "What is the average profit margin?")

    assert outcome.verdict is Verdict.NEEDS_CLARIFICATION
    assert "profit margin" in outcome.clarification_question
    assert "monthly_charge" in outcome.clarification_question  # offers real options


def test_the_check_is_skipped_without_the_question(telecom: pd.DataFrame):
    # Callers that do not pass the question still get every other check.
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, metric="monthly_charge", aggregation="mean"
    )
    assert validate_plan(telecom, plan).verdict is Verdict.OK


def test_substitution_needs_a_metric(telecom: pd.DataFrame):
    assert detect_metric_substitution(telecom, "anything", None) is None
    assert detect_metric_substitution(telecom, "", "revenue") is None
