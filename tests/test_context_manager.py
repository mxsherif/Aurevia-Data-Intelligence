"""Tests for conversational context and the session manager.

Two failure modes matter here, and they pull in opposite directions:

- *Not* carrying context makes "what about last quarter?" unanswerable.
- Carrying it too eagerly makes "show the distribution of satisfaction
  scores" answer a question about revenue by region.

The second is worse, so most of these tests are about refusing to carry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.context import AnalysisHistoryItem, AnalyticalContext, TimeRange
from app.models.plans import AnalysisPlan, Intent, PlanFilter
from app.models.profile import DatasetProfile
from app.models.results import AnalysisResult
from app.services.context_manager import (
    apply_context,
    detect_time_phrase,
    resolve_context,
    update_context,
)
from app.services.session_manager import (
    KEY_CONTEXT,
    MAX_HISTORY,
    SessionManager,
    dataset_key,
)

KEY = "dataset-one"
OTHER_KEY = "dataset-two"


@pytest.fixture
def telecom() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(12)],
            "region": ["Cairo", "Delta", "Canal"] * 4,
            "city": ["Maadi", "Tanta", "Suez"] * 4,
            "signup_date": pd.date_range("2024-01-15", periods=12, freq="ME"),
            "contract_type": ["Month-to-month", "One year", "Two year"] * 4,
            "monthly_charge": [100.0, 220.0, 180.0, 340.0, 150.0, 410.0,
                               260.0, 190.0, 300.0, 130.0, 480.0, 210.0],
            "satisfaction_score": [5, 4, 3, 5, 2, 4, 3, 5, 4, 3, 2, 5],
            "revenue": [1000.0, 2200.0, 1800.0, 3400.0, 1500.0, 4100.0,
                        2600.0, 1900.0, 3000.0, 1300.0, 4800.0, 2100.0],
        }
    )


@pytest.fixture
def after_revenue_by_region() -> AnalyticalContext:
    """The context left behind by "which region generated the most revenue?"."""
    return AnalyticalContext(
        dataset_key=KEY,
        current_metric="revenue",
        dimensions=["region"],
        previous_question="Which region generated the most revenue?",
        previous_intent="ranking",
        turn_count=1,
    )


# --------------------------------------------------------------------------- #
# Continuation detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "question",
    [
        "What about last quarter?",
        "How about by city?",
        "And for Delta?",
        "Now show it monthly",
        "Compare that with Delta",
        "Only for Cairo",
        "Break that down by city",
        "last quarter?",
        "What changed?",
        "Delta?",
    ],
)
def test_continuation_phrases_are_detected(
    telecom: pd.DataFrame, after_revenue_by_region, question: str
):
    resolution = resolve_context(
        question, after_revenue_by_region, telecom, dataset_key=KEY
    )
    assert resolution.is_continuation, question
    assert resolution.signals
    assert resolution.context_prompt


@pytest.mark.parametrize(
    "question",
    [
        "Show the distribution of satisfaction scores.",
        "Which contract type has the highest monthly charge?",
        "How many customers are in each city?",
        "What does this dataset contain?",
        "Which numeric variables are most strongly correlated?",
    ],
)
def test_self_contained_questions_are_not_continuations(
    telecom: pd.DataFrame, after_revenue_by_region, question: str
):
    """A question that stands alone must not inherit anything."""
    resolution = resolve_context(
        question, after_revenue_by_region, telecom, dataset_key=KEY
    )
    assert not resolution.is_continuation, question
    assert resolution.context_prompt == ""


def test_no_prior_context_means_no_continuation(telecom: pd.DataFrame):
    resolution = resolve_context(
        "What about last quarter?", AnalyticalContext(dataset_key=KEY),
        telecom, dataset_key=KEY,
    )
    assert not resolution.is_continuation


def test_a_different_dataset_drops_the_context(
    telecom: pd.DataFrame, after_revenue_by_region
):
    resolution = resolve_context(
        "What about last quarter?", after_revenue_by_region, telecom,
        dataset_key=OTHER_KEY,
    )
    assert not resolution.is_continuation
    assert any("different dataset" in note for note in resolution.notes)


def test_an_empty_question_resolves_to_nothing(
    telecom: pd.DataFrame, after_revenue_by_region
):
    assert not resolve_context("", after_revenue_by_region, telecom).is_continuation


# --------------------------------------------------------------------------- #
# What the question itself names
# --------------------------------------------------------------------------- #

def test_explicit_columns_are_detected(telecom: pd.DataFrame):
    resolution = resolve_context(
        "Compare average monthly charge by contract type", None, telecom
    )
    assert resolution.explicit_metric == "monthly_charge"
    assert "contract_type" in resolution.explicit_dimensions


def test_category_values_are_detected(telecom: pd.DataFrame):
    resolution = resolve_context("What about Delta?", None, telecom)
    assert "Delta" in resolution.entities


def test_entities_are_not_invented(telecom: pd.DataFrame):
    resolution = resolve_context("What about Atlantis?", None, telecom)
    assert resolution.entities == []


@pytest.mark.parametrize(
    ("phrase", "granularity", "periods"),
    [
        ("What about last quarter?", "quarterly", 1),
        ("And last month?", "monthly", 1),
        ("What about last year?", "yearly", 1),
        ("Show the last six months", "monthly", 6),
        ("Over time", "monthly", None),
    ],
)
def test_time_phrases_are_parsed(phrase, granularity, periods):
    scope = detect_time_phrase(phrase)
    assert scope is not None
    assert scope.granularity == granularity
    assert scope.periods == periods


def test_a_question_with_no_time_phrase_has_no_scope():
    assert detect_time_phrase("Which region earns most?") is None


# --------------------------------------------------------------------------- #
# Applying context
# --------------------------------------------------------------------------- #

def test_a_follow_up_inherits_metric_and_dimension(
    telecom: pd.DataFrame, after_revenue_by_region
):
    """Q1 "which region..." then Q2 "what about last quarter?"."""
    resolution = resolve_context(
        "What about last quarter?", after_revenue_by_region, telecom,
        dataset_key=KEY,
    )
    # A planner seeing only a fragment returns almost nothing.
    bare = AnalysisPlan(intent=Intent.RANKING, aggregation="sum")

    applied, notes = apply_context(
        bare, after_revenue_by_region, resolution, telecom
    )

    assert applied.metric == "revenue"
    assert applied.dimensions == ["region"]
    assert applied.time_granularity == "quarterly"
    assert applied.periods == 1
    assert any("revenue" in note for note in notes)
    assert any("region" in note for note in notes)
    assert any("last quarter" in note for note in notes)


def test_an_unrelated_question_inherits_nothing(
    telecom: pd.DataFrame, after_revenue_by_region
):
    """Q3 "show the distribution of satisfaction scores" must start fresh."""
    question = "Show the distribution of satisfaction scores."
    resolution = resolve_context(
        question, after_revenue_by_region, telecom, dataset_key=KEY
    )
    plan = AnalysisPlan(intent=Intent.DISTRIBUTION, metric="satisfaction_score")

    applied, notes = apply_context(
        plan, after_revenue_by_region, resolution, telecom
    )

    assert applied.metric == "satisfaction_score"
    assert applied.dimensions == []          # not region
    assert applied.time_column is None
    assert notes == []


def test_an_explicit_new_metric_replaces_the_old_one(
    telecom: pd.DataFrame, after_revenue_by_region
):
    resolution = resolve_context(
        "What about monthly charge?", after_revenue_by_region, telecom,
        dataset_key=KEY,
    )
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="monthly_charge", aggregation="mean"
    )
    applied, notes = apply_context(
        plan, after_revenue_by_region, resolution, telecom
    )

    assert applied.metric == "monthly_charge"
    assert any("Switched the measure" in note for note in notes)


def test_an_explicit_new_dimension_replaces_the_old_one(
    telecom: pd.DataFrame, after_revenue_by_region
):
    resolution = resolve_context(
        "And by city?", after_revenue_by_region, telecom, dataset_key=KEY
    )
    plan = AnalysisPlan(
        intent=Intent.RANKING, dimensions=["city"], aggregation="sum"
    )
    applied, notes = apply_context(
        plan, after_revenue_by_region, resolution, telecom
    )

    assert applied.dimensions == ["city"]
    assert applied.metric == "revenue"  # the measure still carries
    assert any("Changed the breakdown" in note for note in notes)


def test_a_resetting_intent_drops_the_previous_measure(
    telecom: pd.DataFrame, after_revenue_by_region
):
    """A correlation question starts a new line of enquiry."""
    resolution = resolve_context(
        "What about correlations?", after_revenue_by_region, telecom,
        dataset_key=KEY,
    )
    plan = AnalysisPlan(intent=Intent.CORRELATION)
    applied, notes = apply_context(
        plan, after_revenue_by_region, resolution, telecom
    )

    assert applied.metric is None
    assert applied.dimensions == []
    assert any("fresh" in note.lower() for note in notes)


def test_filters_carry_only_when_the_question_adds_none(telecom: pd.DataFrame):
    context = AnalyticalContext(
        dataset_key=KEY,
        current_metric="revenue",
        dimensions=["region"],
        filters=[{"column": "contract_type", "operator": "eq", "value": "One year"}],
        previous_question="Revenue by region for one-year contracts",
        previous_intent="ranking",
    )
    resolution = resolve_context("And by city?", context, telecom, dataset_key=KEY)
    plan = AnalysisPlan(intent=Intent.RANKING, dimensions=["city"], aggregation="sum")

    applied, notes = apply_context(plan, context, resolution, telecom)

    assert len(applied.filters) == 1
    assert applied.filters[0].column == "contract_type"
    assert any("Kept the filter" in note for note in notes)


def test_a_filter_is_not_carried_onto_a_named_entity(telecom: pd.DataFrame):
    """"What about Delta?" names its own scope, so the old filter is dropped."""
    context = AnalyticalContext(
        dataset_key=KEY,
        current_metric="revenue",
        dimensions=["region"],
        filters=[{"column": "region", "operator": "eq", "value": "Cairo"}],
        previous_question="Revenue in Cairo",
        previous_intent="summary",
    )
    resolution = resolve_context("What about Delta?", context, telecom, dataset_key=KEY)
    plan = AnalysisPlan(intent=Intent.SUMMARY, aggregation="sum")

    applied, _ = apply_context(plan, context, resolution, telecom)
    assert applied.filters == []


def test_the_planners_own_filters_are_never_overwritten(telecom: pd.DataFrame):
    context = AnalyticalContext(
        dataset_key=KEY,
        current_metric="revenue",
        filters=[{"column": "region", "operator": "eq", "value": "Cairo"}],
        previous_question="Revenue in Cairo",
        previous_intent="summary",
    )
    resolution = resolve_context("What about last quarter?", context, telecom,
                                 dataset_key=KEY)
    plan = AnalysisPlan(
        intent=Intent.SUMMARY, aggregation="sum",
        filters=[PlanFilter(column="region", operator="eq", value="Delta")],
    )
    applied, _ = apply_context(plan, context, resolution, telecom)

    assert len(applied.filters) == 1
    assert applied.filters[0].value == "Delta"


def test_a_carried_column_must_still_exist(after_revenue_by_region):
    """Context naming a column the new dataframe lacks is ignored."""
    other = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    resolution = resolve_context(
        "What about last quarter?", after_revenue_by_region, other, dataset_key=KEY
    )
    plan = AnalysisPlan(intent=Intent.RANKING)
    applied, _ = apply_context(plan, after_revenue_by_region, resolution, other)

    assert applied.metric is None
    assert applied.dimensions == []


def test_no_context_means_no_changes(telecom: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.RANKING, metric="revenue")
    applied, notes = apply_context(
        plan, None, resolve_context("anything", None, telecom), telecom
    )
    assert applied is plan
    assert notes == []


# --------------------------------------------------------------------------- #
# Updating context
# --------------------------------------------------------------------------- #

def test_context_records_the_analysis_just_run(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.RANKING, metric="revenue", dimensions=["region"],
        aggregation="sum",
    )
    result = AnalysisResult(
        success=True, title="Total revenue by region",
        summary_data={"Top region": "Canal", "Total revenue (Canal)": 11000.0},
    )
    context = update_context(
        None, "Which region earns most?", plan, result, dataset_key=KEY
    )

    assert context.dataset_key == KEY
    assert context.current_metric == "revenue"
    assert context.dimensions == ["region"]
    assert context.previous_question == "Which region earns most?"
    assert context.previous_intent == "ranking"
    assert context.turn_count == 1
    # The leading category is what a follow-up is most likely to reference.
    assert "Canal" in context.referenced_entities


def test_turn_count_accumulates(telecom: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.RANKING, metric="revenue", dimensions=["region"])
    context = update_context(None, "q1", plan, None, dataset_key=KEY)
    context = update_context(context, "q2", plan, None, dataset_key=KEY)
    assert context.turn_count == 2


def test_a_dataset_change_restarts_the_count(telecom: pd.DataFrame):
    plan = AnalysisPlan(intent=Intent.RANKING, metric="revenue", dimensions=["region"])
    first = update_context(None, "q1", plan, None, dataset_key=KEY)
    second = update_context(first, "q2", plan, None, dataset_key=OTHER_KEY)

    assert second.turn_count == 1
    assert second.dataset_key == OTHER_KEY


def test_the_time_range_is_recorded(telecom: pd.DataFrame):
    plan = AnalysisPlan(
        intent=Intent.TREND, metric="revenue", time_column="signup_date",
        time_granularity="quarterly", periods=1,
    )
    resolution = resolve_context("What about last quarter?", None, telecom)
    context = update_context(
        None, "What about last quarter?", plan, None,
        dataset_key=KEY, resolution=resolution,
    )

    assert context.time_range is not None
    assert context.time_range.column == "signup_date"
    assert context.time_range.granularity == "quarterly"
    assert "quarter" in context.time_range.describe()


# --------------------------------------------------------------------------- #
# The context model
# --------------------------------------------------------------------------- #

def test_an_empty_context_renders_no_prompt():
    context = AnalyticalContext(dataset_key=KEY)
    assert context.is_empty
    assert context.to_prompt() == ""
    assert "No prior analysis" in context.describe()


def test_the_prompt_is_compact(after_revenue_by_region):
    prompt = after_revenue_by_region.to_prompt()
    assert "revenue" in prompt
    assert "region" in prompt
    # A few lines, not a transcript.
    assert len(prompt) < 400
    assert len(prompt.splitlines()) <= 8


def test_belongs_to_is_strict():
    context = AnalyticalContext(dataset_key=KEY)
    assert context.belongs_to(KEY)
    assert not context.belongs_to(OTHER_KEY)
    assert not context.belongs_to(None)
    assert not AnalyticalContext().belongs_to(None)


def test_time_range_description():
    assert TimeRange(label="the last quarter").describe() == "the last quarter"
    assert "2024-01-01 to 2024-03-31" == TimeRange(
        start="2024-01-01", end="2024-03-31"
    ).describe()
    assert TimeRange(granularity="monthly", periods=6).describe().startswith(
        "the last 6"
    )
    assert TimeRange().is_empty


# --------------------------------------------------------------------------- #
# The session manager
# --------------------------------------------------------------------------- #

def _load_result(df: pd.DataFrame, name: str = "data.csv"):
    """A minimal stand-in for LoadResult."""
    from app.services.data_loader import LoadResult

    return LoadResult(
        dataframe=df, source_name=name, file_format="csv",
        rows=len(df), columns=df.shape[1],
    )


def test_the_session_starts_empty():
    session = SessionManager({})
    session.init()

    assert not session.has_dataset
    assert session.dataframe is None
    assert session.history == []
    assert session.context.is_empty


def test_installing_a_dataset(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    changed = session.set_dataset(_load_result(telecom), DatasetProfile())

    assert changed
    assert session.has_dataset
    assert session.dataset_key is not None
    assert session.dataset_name == "data.csv"
    assert session.context.belongs_to(session.dataset_key)


def test_reloading_the_same_dataset_keeps_the_conversation(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())

    plan = AnalysisPlan(intent=Intent.RANKING, metric="revenue", dimensions=["region"])
    session.set_context(
        update_context(None, "q1", plan, None, dataset_key=session.dataset_key)
    )
    session.add_history(
        AnalysisHistoryItem(question="q1", intent="ranking")
    )

    changed = session.set_dataset(_load_result(telecom), DatasetProfile())

    assert not changed
    assert session.context.current_metric == "revenue"
    assert len(session.history) == 1


def test_a_new_dataset_clears_the_analytical_session(telecom: pd.DataFrame):
    """The headline context-safety rule, enforced in one place."""
    session = SessionManager({})
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())

    plan = AnalysisPlan(intent=Intent.RANKING, metric="revenue", dimensions=["region"])
    session.set_context(
        update_context(None, "q1", plan, None, dataset_key=session.dataset_key)
    )
    session.add_history(AnalysisHistoryItem(question="q1", intent="ranking"))
    session.set_last_outcome("an answer")
    session.set_investigation("an investigation")

    other = telecom.rename(columns={"revenue": "turnover"})
    changed = session.set_dataset(_load_result(other, "other.csv"), DatasetProfile())

    assert changed
    assert session.context.is_empty
    assert session.context.current_metric is None
    assert session.history == []
    assert session.last_outcome is None
    assert session.investigation is None


def test_context_from_another_dataset_is_never_returned(telecom: pd.DataFrame):
    """Even a hand-planted mismatch is discarded on read."""
    state: dict = {}
    session = SessionManager(state)
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())

    state[KEY_CONTEXT] = AnalyticalContext(
        dataset_key="some-other-dataset", current_metric="revenue"
    )
    assert session.context.is_empty
    assert session.context.belongs_to(session.dataset_key)


def test_clearing_the_dataset_clears_everything(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())
    session.add_history(AnalysisHistoryItem(question="q", intent="ranking"))

    session.clear_dataset()

    assert not session.has_dataset
    assert session.history == []
    assert session.dataset_key is None


def test_a_load_error_clears_the_session(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())
    session.add_history(AnalysisHistoryItem(question="q", intent="ranking"))

    session.set_error("that file is empty")

    assert session.error == "that file is empty"
    assert not session.has_dataset
    assert session.history == []


def test_history_is_capped_and_ordered(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    for index in range(MAX_HISTORY + 5):
        session.add_history(
            AnalysisHistoryItem(question=f"q{index}", intent="ranking")
        )

    assert len(session.history) == MAX_HISTORY
    # The oldest were dropped, newest kept.
    assert session.history[-1].question == f"q{MAX_HISTORY + 4}"
    assert session.recent_history(limit=2)[0].question == f"q{MAX_HISTORY + 4}"


def test_resetting_the_context_keeps_the_dataset(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())
    plan = AnalysisPlan(intent=Intent.RANKING, metric="revenue")
    session.set_context(
        update_context(None, "q", plan, None, dataset_key=session.dataset_key)
    )

    session.reset_context()

    assert session.has_dataset
    assert session.context.is_empty


def test_dataset_keys_distinguish_datasets(telecom: pd.DataFrame):
    same = dataset_key(telecom, "a.csv") == dataset_key(telecom, "a.csv")
    renamed = dataset_key(telecom, "a.csv") == dataset_key(telecom, "b.csv")
    altered = dataset_key(telecom, "a.csv") == dataset_key(
        telecom.rename(columns={"revenue": "turnover"}), "a.csv"
    )

    assert same
    assert not renamed
    assert not altered


def test_dataset_key_survives_a_broken_frame():
    class Awkward:
        @property
        def shape(self):
            raise RuntimeError("no shape for you")

    assert dataset_key(Awkward(), "x.csv")  # a key, not an exception


def test_describe_is_serialisable(telecom: pd.DataFrame):
    session = SessionManager({})
    session.init()
    session.set_dataset(_load_result(telecom), DatasetProfile())

    summary = session.describe()
    assert summary["has_dataset"] is True
    assert summary["history_length"] == 0


def test_history_items_render_a_headline():
    item = AnalysisHistoryItem(
        question="Which region?", intent="ranking",
        result_summary={"Top region": "Cairo", "Total revenue": 15556354.13},
    )
    # A figure is more informative in a one-line history row than a label.
    assert item.headline() == "Total revenue: 15,556,354.13"
    assert item.when
    assert item.to_dict()["intent"] == "ranking"


def test_an_empty_history_item_has_no_headline():
    assert AnalysisHistoryItem(question="q", intent="ranking").headline() == ""
