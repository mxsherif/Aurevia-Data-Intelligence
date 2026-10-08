"""Tests for the investigation engine, its validator, and the pipeline.

Every fixture here has hand-computed answers, because the point of an
investigation is the arithmetic: a contribution figure that is subtly wrong is
worse than no figure at all. The offsetting cases matter most — that is where
"share of the decline" turns into nonsense if nobody is watching.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.investigation import (
    Direction,
    InvestigationPlan,
    InvestigationSummary,
)
from app.services.investigation_engine import (
    MAX_CARDINALITY,
    MAX_CATEGORIES,
    OFFSETTING_RATIO,
    claimed_direction,
    discover_dimensions,
    run_investigation,
)
from app.services.investigation_pipeline import (
    InvestigationStage,
    investigate,
)
from app.services.investigation_validator import validate_investigation_plan


# --------------------------------------------------------------------------- #
# Fixtures with known answers
# --------------------------------------------------------------------------- #

def build(spec: list[tuple[str, float, float]], *, product: str = "A") -> pd.DataFrame:
    """Two quarters of data from (region, Q1 value, Q2 value) triples."""
    rows = []
    for region, baseline, comparison in spec:
        rows.append(
            {"d": "2024-01-15", "region": region, "revenue": baseline,
             "product": product, "tier": "std"}
        )
        rows.append(
            {"d": "2024-04-15", "region": region, "revenue": comparison,
             "product": product, "tier": "std"}
        )
    frame = pd.DataFrame(rows)
    frame["d"] = pd.to_datetime(frame["d"])
    return frame


def plan_for(
    df: pd.DataFrame, claim: str = "decrease", **overrides
) -> InvestigationPlan:
    fields = {
        "metric": "revenue",
        "time_column": "d",
        "granularity": "quarterly",
        "aggregation": "sum",
        "candidate_dimensions": ["region"],
        "claimed_direction": claim,
    }
    fields.update(overrides)
    plan = InvestigationPlan(**fields)
    validated = validate_investigation_plan(df, plan)
    assert validated.ok, validated.message or validated.clarification_question
    return validated.plan


@pytest.fixture
def clean_decline() -> pd.DataFrame:
    """Total 1000 -> 700. North falls 300; the rest are flat."""
    return build([("North", 500, 200), ("South", 300, 300), ("East", 200, 200)])


@pytest.fixture
def offsetting() -> pd.DataFrame:
    """North -500, South +400, East flat. Net -100, gross movement 900."""
    return build([("North", 1000, 500), ("South", 200, 600), ("East", 300, 300)])


@pytest.fixture
def growth() -> pd.DataFrame:
    """Total 500 -> 800: an increase, whatever the question claims."""
    return build([("North", 200, 500), ("South", 300, 300)])


# --------------------------------------------------------------------------- #
# Premise verification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Why did revenue decline last quarter?", Direction.DOWN),
        ("Why did revenue drop?", Direction.DOWN),
        ("What caused the fall in revenue?", Direction.DOWN),
        ("Why did revenue increase?", Direction.UP),
        ("What drove the growth in revenue?", Direction.UP),
        ("Why did revenue change?", None),
        ("Tell me about revenue", None),
    ],
)
def test_the_claimed_direction_is_read_from_the_question(question, expected):
    assert claimed_direction(question) is expected


def test_a_true_decline_is_confirmed(clean_decline: pd.DataFrame):
    result = run_investigation(
        clean_decline, plan_for(clean_decline),
        question="Why did revenue decline last quarter?",
    )

    assert result.success
    assert result.premise_confirmed
    assert result.direction is Direction.DOWN
    assert result.baseline_value == pytest.approx(1000.0)
    assert result.comparison_value == pytest.approx(700.0)
    assert result.absolute_change == pytest.approx(-300.0)
    assert result.percentage_change == pytest.approx(-30.0)
    assert "Confirmed the premise" in result.steps_completed


def test_a_false_decline_is_challenged(growth: pd.DataFrame):
    """The headline quality feature: do not accept a false premise."""
    result = run_investigation(
        growth, plan_for(growth), question="Why did revenue decline last quarter?"
    )

    assert result.success           # the investigation ran
    assert not result.premise_confirmed
    assert result.direction is Direction.UP
    assert result.claimed_direction is Direction.DOWN
    assert "did not decline" in result.premise_message
    assert "60.00%" in result.premise_message
    assert "increase instead" in result.premise_message
    # No contributor analysis of a change that did not happen.
    assert result.breakdowns == []
    assert "Checked the premise — it does not hold" in result.steps_completed


def test_a_false_increase_is_challenged(clean_decline: pd.DataFrame):
    result = run_investigation(
        clean_decline, plan_for(clean_decline, claim="increase"),
        question="Why did revenue increase last quarter?",
    )
    assert not result.premise_confirmed
    assert "did not increase" in result.premise_message


def test_a_question_with_no_claim_investigates_whatever_happened(
    growth: pd.DataFrame,
):
    result = run_investigation(
        growth, plan_for(growth, claim="unknown"),
        question="What changed in revenue last quarter?",
    )
    assert result.premise_confirmed
    assert result.direction is Direction.UP
    assert result.breakdowns  # it proceeded to decompose


def test_a_flat_metric_is_reported_as_unchanged():
    flat = build([("North", 500, 501), ("South", 500, 499)])
    result = run_investigation(
        flat, plan_for(flat), question="Why did revenue decline last quarter?"
    )
    assert result.direction is Direction.FLAT
    assert not result.premise_confirmed
    assert "unchanged" in result.premise_message


# --------------------------------------------------------------------------- #
# Contribution arithmetic
# --------------------------------------------------------------------------- #

def test_contribution_shares_of_a_clean_decline(clean_decline: pd.DataFrame):
    result = run_investigation(
        clean_decline, plan_for(clean_decline), question="Why the decline?"
    )
    breakdown = next(b for b in result.breakdowns if b.dimension == "region")

    assert not breakdown.offsetting
    assert breakdown.gross_movement == pytest.approx(300.0)

    north = next(f for f in breakdown.findings if f.category == "North")
    assert north.rank == 1
    assert north.absolute_change == pytest.approx(-300.0)
    # The whole net change, and the whole movement.
    assert north.contribution == pytest.approx(100.0)
    assert north.share_of_movement == pytest.approx(100.0)
    assert north.percentage_change == pytest.approx(-60.0)
    assert north.direction is Direction.DOWN


def test_flat_categories_contribute_nothing(clean_decline: pd.DataFrame):
    result = run_investigation(
        clean_decline, plan_for(clean_decline), question="Why the decline?"
    )
    breakdown = result.breakdowns[0]
    south = next(f for f in breakdown.findings if f.category == "South")

    assert south.absolute_change == pytest.approx(0.0)
    assert south.share_of_movement == pytest.approx(0.0)
    assert south.direction is Direction.FLAT


def test_contributions_sum_to_the_whole_net_change():
    """The decomposition must account for the change, not approximate it."""
    frame = build([("North", 500, 300), ("South", 300, 200), ("East", 200, 150)])
    result = run_investigation(frame, plan_for(frame), question="Why the decline?")
    breakdown = result.breakdowns[0]

    total = sum(f.absolute_change for f in breakdown.findings)
    assert total == pytest.approx(result.absolute_change)
    shares = sum(f.contribution for f in breakdown.findings)
    assert shares == pytest.approx(100.0, abs=0.1)


def test_offsetting_movement_suppresses_the_net_share(offsetting: pd.DataFrame):
    """Shares of a small net change are misleading, so they are withheld.

    North falls 500 while South rises 400: the net is -100, so North's "share
    of the decline" would be 500%. The gross-movement share is reported
    instead.
    """
    result = run_investigation(
        offsetting, plan_for(offsetting), question="Why the decline?"
    )
    breakdown = result.breakdowns[0]

    assert result.absolute_change == pytest.approx(-100.0)
    assert breakdown.gross_movement == pytest.approx(900.0)
    assert breakdown.offsetting
    assert all(f.contribution is None for f in breakdown.findings)

    north = next(f for f in breakdown.findings if f.category == "North")
    south = next(f for f in breakdown.findings if f.category == "South")
    assert north.share_of_movement == pytest.approx(500 / 900 * 100, abs=0.01)
    assert south.share_of_movement == pytest.approx(400 / 900 * 100, abs=0.01)
    assert any("cancel out" in note for note in breakdown.notes)


def test_the_offsetting_threshold(offsetting: pd.DataFrame):
    result = run_investigation(
        offsetting, plan_for(offsetting), question="Why the decline?"
    )
    breakdown = result.breakdowns[0]
    # |net| / gross = 100/900 = 0.11, below the threshold.
    ratio = abs(result.absolute_change) / breakdown.gross_movement
    assert ratio < OFFSETTING_RATIO


def test_counter_trend_categories_are_identified(offsetting: pd.DataFrame):
    result = run_investigation(
        offsetting, plan_for(offsetting), question="Why the decline?"
    )
    counter = result.counter_findings()

    assert [f.category for f in counter] == ["South"]
    assert counter[0].absolute_change == pytest.approx(400.0)


def test_a_non_additive_aggregation_skips_contribution_shares():
    """A mean does not decompose additively, so no share is claimed."""
    frame = build([("North", 500, 200), ("South", 300, 300)])
    result = run_investigation(
        frame, plan_for(frame, aggregation="mean"), question="Why the decline?"
    )
    breakdown = result.breakdowns[0]

    assert all(f.contribution is None for f in breakdown.findings)
    assert any("additively" in note for note in breakdown.notes)


def test_a_category_present_in_only_one_period_is_handled():
    """A category that appears or vanishes contributes its whole value."""
    rows = [
        {"d": "2024-01-15", "region": "North", "revenue": 500.0, "tier": "a"},
        {"d": "2024-01-15", "region": "Gone", "revenue": 200.0, "tier": "a"},
        {"d": "2024-04-15", "region": "North", "revenue": 500.0, "tier": "a"},
        {"d": "2024-04-15", "region": "New", "revenue": 50.0, "tier": "a"},
    ]
    frame = pd.DataFrame(rows)
    frame["d"] = pd.to_datetime(frame["d"])

    result = run_investigation(frame, plan_for(frame), question="Why the decline?")
    breakdown = result.breakdowns[0]
    gone = next(f for f in breakdown.findings if f.category == "Gone")
    new = next(f for f in breakdown.findings if f.category == "New")

    assert gone.baseline_value == pytest.approx(200.0)
    assert gone.comparison_value == pytest.approx(0.0)
    assert gone.absolute_change == pytest.approx(-200.0)
    assert new.baseline_value == pytest.approx(0.0)
    assert new.absolute_change == pytest.approx(50.0)


def test_categories_are_capped_with_a_note():
    # Several rows per category, so the column is a real dimension rather
    # than something the ID-like guard would (correctly) reject.
    rows = []
    for index in range(MAX_CATEGORIES + 4):
        for _ in range(4):
            rows.append(
                {"d": "2024-01-15", "region": f"R{index:02d}", "revenue": 25.0,
                 "product": "A", "tier": "std"}
            )
            rows.append(
                {"d": "2024-04-15", "region": f"R{index:02d}",
                 "revenue": 25.0 - index * 0.25, "product": "A", "tier": "std"}
            )
    frame = pd.DataFrame(rows)
    frame["d"] = pd.to_datetime(frame["d"])

    result = run_investigation(frame, plan_for(frame), question="Why the decline?")
    breakdown = result.breakdowns[0]

    assert len(breakdown.findings) == MAX_CATEGORIES
    assert breakdown.categories_examined == MAX_CATEGORIES + 4
    assert any("largest movers" in note for note in breakdown.notes)


def test_findings_are_ranked_by_absolute_movement():
    frame = build([("A", 100, 90), ("B", 100, 40), ("C", 100, 70)])
    result = run_investigation(frame, plan_for(frame), question="Why the decline?")
    changes = [abs(f.absolute_change) for f in result.breakdowns[0].findings]

    assert changes == sorted(changes, reverse=True)
    assert result.breakdowns[0].findings[0].category == "B"


def test_ranked_findings_cross_dimensions(clean_decline: pd.DataFrame):
    frame = clean_decline.copy()
    frame["tier"] = ["gold", "silver"] * (len(frame) // 2)

    result = run_investigation(
        frame, plan_for(frame, candidate_dimensions=["region", "tier"]),
        question="Why the decline?",
    )
    ranked = result.ranked_findings(limit=3)

    assert len(result.breakdowns) == 2
    assert ranked
    assert all(f.direction is Direction.DOWN for f in ranked)
    magnitudes = [abs(f.absolute_change) for f in ranked]
    assert magnitudes == sorted(magnitudes, reverse=True)


# --------------------------------------------------------------------------- #
# Dimension discovery
# --------------------------------------------------------------------------- #

def test_the_planners_dimensions_are_honoured_when_usable(
    clean_decline: pd.DataFrame,
):
    chosen, skipped = discover_dimensions(clean_decline, ["region"])
    assert chosen[0] == "region"
    assert "region" not in skipped


def test_a_high_cardinality_column_is_skipped():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-15"] * 100 + ["2024-04-15"] * 100),
            "revenue": [10.0] * 200,
            "code": [f"C{i:04d}" for i in range(200)],
            "region": ["North", "South"] * 100,
        }
    )
    chosen, skipped = discover_dimensions(frame, ["code", "region"])

    assert "code" not in chosen
    assert "code" in skipped
    assert "region" in chosen


def test_an_identifier_is_skipped():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-15", "2024-04-15"] * 30),
            "revenue": [10.0] * 60,
            "customer_id": [f"ID{i}" for i in range(60)],
            "region": ["North", "South"] * 30,
        }
    )
    chosen, skipped = discover_dimensions(frame, ["customer_id"])

    assert "customer_id" not in chosen
    assert "customer_id" in skipped


def test_a_numeric_column_is_not_a_dimension(clean_decline: pd.DataFrame):
    chosen, skipped = discover_dimensions(clean_decline, ["revenue"])
    assert "revenue" not in chosen
    assert "numeric" in skipped["revenue"]


def test_a_date_column_is_not_a_dimension(clean_decline: pd.DataFrame):
    chosen, skipped = discover_dimensions(clean_decline, ["d"])
    assert "d" not in chosen
    assert "date" in skipped["d"]


def test_a_single_valued_column_is_skipped(clean_decline: pd.DataFrame):
    chosen, skipped = discover_dimensions(clean_decline, ["product"])
    assert "product" not in chosen
    assert "one value" in skipped["product"]


def test_a_missing_column_is_skipped(clean_decline: pd.DataFrame):
    chosen, skipped = discover_dimensions(clean_decline, ["planet"])
    assert "planet" not in chosen
    assert "not a column" in skipped["planet"]


def test_the_metric_and_date_are_excluded(clean_decline: pd.DataFrame):
    chosen, _ = discover_dimensions(
        clean_decline, None, exclude={"revenue", "d"}
    )
    assert "revenue" not in chosen
    assert "d" not in chosen


def test_discovery_fills_in_from_the_dataset(clean_decline: pd.DataFrame):
    """With no suggestions, usable columns are found anyway."""
    chosen, _ = discover_dimensions(
        clean_decline, None, exclude={"revenue", "d"}
    )
    assert "region" in chosen


def test_discovery_respects_the_limit():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-15", "2024-04-15"] * 10),
            "revenue": [1.0] * 20,
            **{f"dim{i}": ["a", "b"] * 10 for i in range(6)},
        }
    )
    chosen, _ = discover_dimensions(frame, None, exclude={"revenue", "d"}, limit=3)
    assert len(chosen) == 3


def test_cardinality_cap_is_documented():
    assert MAX_CARDINALITY >= 10


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def test_the_timing_breakdown_covers_the_compared_window():
    dates = pd.date_range("2024-01-01", "2024-06-30", freq="15D")
    frame = pd.DataFrame(
        {
            "d": dates,
            "region": ["North", "South"] * (len(dates) // 2) + ["North"] * (len(dates) % 2),
            "revenue": [100.0 - i * 5 for i in range(len(dates))],
        }
    )
    result = run_investigation(frame, plan_for(frame), question="Why the decline?")

    assert result.temporal_findings
    periods = [f.period for f in result.temporal_findings]
    # Monthly sub-periods inside (and one before) the two compared quarters.
    assert all(p.startswith("2024-") for p in periods)
    assert len(periods) <= 8
    # Period-over-period movement is attached after the first.
    assert result.temporal_findings[0].change_vs_previous is None
    assert result.temporal_findings[-1].change_vs_previous is not None


def test_the_worst_sub_period_is_identified():
    dates = pd.to_datetime(
        ["2024-01-15", "2024-02-15", "2024-03-15",
         "2024-04-15", "2024-05-15", "2024-06-15"]
    )
    frame = pd.DataFrame(
        {
            "d": dates,
            "region": ["North"] * 6,
            "revenue": [100.0, 95.0, 90.0, 85.0, 20.0, 18.0],
        }
    )
    result = run_investigation(frame, plan_for(frame), question="Why the decline?")
    worst = result.worst_period

    assert worst is not None
    assert worst.period == "2024-05"  # the 85 -> 20 drop


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #

def test_a_single_period_cannot_be_compared():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-02-05"]),
            "region": ["North", "South", "North"],
            "revenue": [100.0, 200.0, 150.0],
        }
    )
    plan = InvestigationPlan(
        metric="revenue", time_column="d", granularity="yearly",
        candidate_dimensions=["region"],
    )
    # The validator steps down to a granularity the data actually spans.
    validated = validate_investigation_plan(frame, plan)
    assert validated.ok
    assert validated.plan.granularity in ("monthly", "weekly", "daily")
    assert any("only one" in note for note in validated.adjustments)


def test_a_dataset_with_one_period_at_every_granularity_is_refused():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "region": ["North", "South"],
            "revenue": [100.0, 200.0],
        }
    )
    validated = validate_investigation_plan(
        frame,
        InvestigationPlan(metric="revenue", time_column="d", granularity="yearly"),
    )
    assert not validated.ok
    assert "only one period" in validated.message


def test_an_unparseable_date_column_is_refused():
    frame = pd.DataFrame(
        {
            "d": ["not a date", "nor this", "nope"],
            "region": ["A", "B", "A"],
            "revenue": [1.0, 2.0, 3.0],
        }
    )
    validated = validate_investigation_plan(
        frame, InvestigationPlan(metric="revenue", time_column="d")
    )
    assert not validated.ok
    assert "date field" in validated.message


def test_a_dataset_with_no_date_column_is_refused():
    frame = pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]})
    validated = validate_investigation_plan(
        frame, InvestigationPlan(metric="revenue", time_column="whenever")
    )
    assert not validated.ok
    assert "does not contain a valid date field" in validated.message


def test_a_dataset_with_no_numeric_column_is_refused():
    frame = pd.DataFrame(
        {"d": pd.to_datetime(["2024-01-01", "2024-02-01"]), "label": ["a", "b"]}
    )
    validated = validate_investigation_plan(
        frame, InvestigationPlan(metric="revenue", time_column="d")
    )
    assert not validated.ok
    assert "no numeric column" in validated.message


def test_a_non_numeric_metric_is_refused(clean_decline: pd.DataFrame):
    validated = validate_investigation_plan(
        clean_decline,
        InvestigationPlan(metric="region", time_column="d"),
    )
    assert not validated.ok
    assert "not a numeric column" in validated.message


def test_an_unknown_metric_asks_for_clarification(clean_decline: pd.DataFrame):
    validated = validate_investigation_plan(
        clean_decline,
        InvestigationPlan(metric="profit_margin", time_column="d"),
    )
    assert not validated.ok
    assert validated.needs_clarification
    assert "profit_margin" in validated.clarification_question


def test_an_empty_dataset_is_refused():
    validated = validate_investigation_plan(
        pd.DataFrame({"d": [], "revenue": []}),
        InvestigationPlan(metric="revenue", time_column="d"),
    )
    assert not validated.ok
    assert "no rows" in validated.message


def test_an_unsupported_aggregation_falls_back(clean_decline: pd.DataFrame):
    validated = validate_investigation_plan(
        clean_decline,
        InvestigationPlan(metric="revenue", time_column="d", aggregation="median"),
    )
    assert validated.ok
    # A median cannot be decomposed into contributions.
    assert validated.plan.aggregation == "sum"
    assert any("decomposed" in note for note in validated.adjustments)


def test_a_metric_alias_is_resolved(clean_decline: pd.DataFrame):
    validated = validate_investigation_plan(
        clean_decline, InvestigationPlan(metric="sales", time_column="d")
    )
    assert validated.ok
    assert validated.plan.metric == "revenue"
    assert any("sales" in note for note in validated.adjustments)


def test_the_planners_clarification_is_passed_through(clean_decline: pd.DataFrame):
    validated = validate_investigation_plan(
        clean_decline,
        InvestigationPlan(
            metric="revenue", time_column="d",
            requires_clarification=True,
            clarification_question="Which measure declined?",
        ),
    )
    assert not validated.ok
    assert validated.clarification_question == "Which measure declined?"


def test_a_metric_with_no_values_in_a_period_is_reported():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-15", "2024-04-15"]),
            "region": ["North", "South"],
            "revenue": [100.0, None],
        }
    )
    result = run_investigation(
        frame, plan_for(frame), question="Why the decline?"
    )
    assert not result.success
    assert "no usable values" in result.error


# --------------------------------------------------------------------------- #
# Data sufficiency
# --------------------------------------------------------------------------- #

def test_a_thin_period_is_flagged(clean_decline: pd.DataFrame):
    result = run_investigation(
        clean_decline, plan_for(clean_decline), question="Why the decline?"
    )
    # Three rows per period is well below the threshold.
    assert any("interpreted cautiously" in w for w in result.warnings)


def test_uneven_period_coverage_is_flagged():
    rows = [{"d": "2024-01-15", "region": "North", "revenue": 10.0} for _ in range(80)]
    rows += [{"d": "2024-04-15", "region": "North", "revenue": 5.0} for _ in range(35)]
    frame = pd.DataFrame(rows)
    frame["d"] = pd.to_datetime(frame["d"])

    result = run_investigation(
        frame, plan_for(frame), question="Why the decline?"
    )
    assert any("different volumes" in w for w in result.warnings)


def test_rows_without_a_date_are_counted():
    rows = [{"d": "2024-01-15", "region": "North", "revenue": 10.0} for _ in range(20)]
    rows += [{"d": "2024-04-15", "region": "North", "revenue": 5.0} for _ in range(20)]
    rows += [{"d": None, "region": "North", "revenue": 99.0} for _ in range(10)]
    frame = pd.DataFrame(rows)
    frame["d"] = pd.to_datetime(frame["d"], errors="coerce")

    result = run_investigation(frame, plan_for(frame), question="Why the decline?")

    assert result.evidence_summary["rows_excluded_no_date"] == 10
    assert any("no usable date" in w for w in result.warnings)


def test_offsetting_adds_a_caution(offsetting: pd.DataFrame):
    result = run_investigation(
        offsetting, plan_for(offsetting), question="Why the decline?"
    )
    assert any("offset" in w for w in result.warnings)


def test_the_evidence_record_is_complete(clean_decline: pd.DataFrame):
    result = run_investigation(
        clean_decline, plan_for(clean_decline), question="Why the decline?"
    )
    evidence = result.evidence_summary

    assert evidence["metric"] == "revenue"
    assert evidence["primary_computation"] == "SUM(revenue)"
    assert evidence["periods_compared"] == "2024Q1 vs 2024Q2"
    assert evidence["baseline_rows"] == 3
    assert evidence["comparison_rows"] == 3
    assert evidence["rows_included"] == 6
    assert "region" in evidence["dimensions_inspected"]


def test_the_result_is_serialisable(clean_decline: pd.DataFrame):
    import json

    result = run_investigation(
        clean_decline, plan_for(clean_decline), question="Why the decline?"
    )
    json.dumps(result.to_dict(), default=str)
    json.dumps(result.to_evidence(), default=str)


def test_the_evidence_payload_carries_only_computed_values(
    clean_decline: pd.DataFrame,
):
    result = run_investigation(
        clean_decline, plan_for(clean_decline), question="Why the decline?"
    )
    evidence = result.to_evidence()

    assert evidence["overall_change"]["absolute"] == pytest.approx(-300.0)
    assert evidence["premise_confirmed"] is True
    assert evidence["contributors"]
    assert "region" in evidence["dimensions_inspected"]


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

class ScriptedLLM:
    """A scripted investigation planner and summariser."""

    def __init__(
        self,
        plan: InvestigationPlan | None = None,
        summary: InvestigationSummary | None = None,
        *,
        plan_error: Exception | None = None,
        summary_error: Exception | None = None,
    ) -> None:
        from app.config import Settings
        from app.services.llm_service import LLMService

        self._service = LLMService(Settings(openai_api_key="sk-test"), client=object())
        self._plan = plan
        self._summary = summary or InvestigationSummary(headline="Computed.")
        self._plan_error = plan_error
        self._summary_error = summary_error
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self):
        return None

    def complete_structured(self, messages, schema, **kwargs):
        from types import SimpleNamespace

        from app.services.llm_service import LLMUsage

        self.calls += 1
        usage = LLMUsage(model="fake", prompt_tokens=100, completion_tokens=50)
        if schema is InvestigationPlan:
            if self._plan_error:
                raise self._plan_error
            return SimpleNamespace(data=self._plan, usage=usage)
        if self._summary_error:
            raise self._summary_error
        return SimpleNamespace(data=self._summary, usage=usage)


def _raw_plan(**overrides) -> InvestigationPlan:
    fields = {
        "metric": "revenue", "time_column": "d", "granularity": "quarterly",
        "aggregation": "sum", "candidate_dimensions": ["region"],
        "claimed_direction": "decrease",
    }
    fields.update(overrides)
    return InvestigationPlan(**fields)


def test_the_pipeline_uses_exactly_two_llm_calls(clean_decline: pd.DataFrame):
    """One to plan, one to write up -- regardless of dimension count."""
    frame = clean_decline.copy()
    frame["tier"] = ["gold", "silver"] * (len(frame) // 2)
    llm = ScriptedLLM(
        _raw_plan(candidate_dimensions=["region", "tier"]),
        InvestigationSummary(
            headline="Revenue fell by 300.00, concentrated in North.",
            contributors=["North fell by 300.00."],
        ),
    )
    outcome = investigate(
        "Why did revenue decline last quarter?", frame, llm=llm
    )

    assert outcome.stage is InvestigationStage.COMPLETE
    assert outcome.investigated
    assert outcome.llm_calls == 2
    assert llm.calls == 2
    assert len(outcome.result.breakdowns) == 2
    assert outcome.is_grounded


def test_a_rejected_premise_costs_one_fewer_decomposition(growth: pd.DataFrame):
    llm = ScriptedLLM(
        _raw_plan(),
        InvestigationSummary(headline="Revenue rose by 60.00%, not fell."),
    )
    outcome = investigate(
        "Why did revenue decline last quarter?", growth, llm=llm
    )

    assert outcome.investigated
    assert outcome.premise_rejected
    assert outcome.result.breakdowns == []
    # Still written up, so the user hears why.
    assert outcome.summary is not None


def test_an_invented_figure_in_the_summary_is_flagged(
    clean_decline: pd.DataFrame,
):
    llm = ScriptedLLM(
        _raw_plan(),
        InvestigationSummary(headline="Revenue fell by 987,654,321."),
    )
    outcome = investigate("Why the decline?", clean_decline, llm=llm)

    assert outcome.investigated      # the evidence stands
    assert not outcome.is_grounded
    assert "987,654,321" in outcome.summary.ungrounded_numbers
    assert any("could not be matched" in w for w in outcome.warnings)


def test_a_planning_failure_is_reported(clean_decline: pd.DataFrame):
    from app.services.llm_service import LLMTimeoutError

    llm = ScriptedLLM(plan_error=LLMTimeoutError("The AI request timed out."))
    outcome = investigate("Why the decline?", clean_decline, llm=llm)

    assert outcome.stage is InvestigationStage.PLANNING
    assert outcome.failed
    assert "timed out" in outcome.error


def test_a_summary_failure_keeps_the_evidence(clean_decline: pd.DataFrame):
    from app.services.llm_service import LLMTimeoutError

    llm = ScriptedLLM(
        _raw_plan(), summary_error=LLMTimeoutError("The AI request timed out.")
    )
    outcome = investigate("Why the decline?", clean_decline, llm=llm)

    assert outcome.stage is InvestigationStage.SUMMARY
    assert outcome.investigated          # the figures are there
    assert outcome.summary is None       # the prose is not
    assert outcome.result.absolute_change == pytest.approx(-300.0)
    assert any("written summary could not be generated" in w for w in outcome.warnings)


def test_a_validation_refusal_is_reported():
    frame = pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]})
    llm = ScriptedLLM(_raw_plan(time_column="nothing"))
    outcome = investigate("Why the decline?", frame, llm=llm)

    assert outcome.stage is InvestigationStage.VALIDATION
    assert outcome.failed
    assert "date field" in outcome.error
    assert outcome.llm_calls == 1  # no point writing up nothing


def test_a_missing_api_key_stops_cleanly(clean_decline: pd.DataFrame):
    from app.config import Settings
    from app.services.llm_service import LLMService

    outcome = investigate(
        "Why the decline?", clean_decline,
        llm=LLMService(Settings(openai_api_key="")),
    )
    assert outcome.stage is InvestigationStage.UNAVAILABLE
    assert "requires an OpenAI API key" in outcome.error


def test_an_empty_question_is_refused(clean_decline: pd.DataFrame):
    llm = ScriptedLLM(_raw_plan())
    outcome = investigate("   ", clean_decline, llm=llm)

    assert outcome.failed
    assert llm.calls == 0


def test_the_summary_can_be_skipped(clean_decline: pd.DataFrame):
    llm = ScriptedLLM(_raw_plan())
    outcome = investigate(
        "Why the decline?", clean_decline, llm=llm, explain=False
    )
    assert outcome.investigated
    assert outcome.summary is None
    assert llm.calls == 1


def test_the_outcome_is_serialisable(clean_decline: pd.DataFrame):
    llm = ScriptedLLM(_raw_plan())
    payload = investigate("Why the decline?", clean_decline, llm=llm).to_dict()

    assert payload["investigated"] is True
    assert payload["premise_confirmed"] is True
    assert payload["llm_calls"] == 2
    assert payload["metric"] == "revenue"


def test_no_stack_trace_reaches_the_user(clean_decline: pd.DataFrame):
    for llm in (
        ScriptedLLM(plan_error=RuntimeError("internal explosion")),
        ScriptedLLM(_raw_plan(metric="nope")),
    ):
        outcome = investigate("Why the decline?", clean_decline, llm=llm)
        message = outcome.error or outcome.clarification_question or ""
        assert "Traceback" not in message
        assert ".py" not in message


def test_the_dimension_limit_is_respected(clean_decline: pd.DataFrame):
    frame = clean_decline.copy()
    for index in range(4):
        frame[f"extra{index}"] = ["x", "y"] * (len(frame) // 2)

    llm = ScriptedLLM(_raw_plan(candidate_dimensions=[]))
    outcome = investigate(
        "Why the decline?", frame, llm=llm, max_dimensions=2
    )
    assert len(outcome.result.breakdowns) <= 2
