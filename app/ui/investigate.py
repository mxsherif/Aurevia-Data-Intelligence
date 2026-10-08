"""The Investigate page.

Where Ask Aurevia answers a defined question, Investigate takes a *why* and
works through it: confirm the premise, quantify the change, decompose it across
several dimensions, check the timing, rank what moved, and write it up.

The page shows the progress of that sequence and then the evidence behind every
claim. Phase 6 redesigns the presentation; this is the working version.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.config import Settings, get_settings
from app.models.investigation import Direction, InvestigationResult
from app.services.investigation_pipeline import (
    InvestigationOutcome,
    InvestigationStage,
    investigate,
)
from app.services.llm_service import get_llm_service
from app.services.session_manager import get_session
from app.services.visualization import build_chart
from app.tools import ToolError, generate_chart_data
from app.tools.charts import lower_label
from app.ui.components import render_chart
from app.ui.evidence import format_value, render_cautions, render_record
from app.ui.state import render_no_dataset_notice

logger = logging.getLogger(__name__)

STATE_QUESTION = "investigate_question_text"
STATE_PENDING = "investigate_pending"

#: Contributor rows shown per dimension before the rest are collapsed.
ROWS_PER_DIMENSION = 6


def init_state() -> None:
    st.session_state.setdefault(STATE_QUESTION, "")
    st.session_state.setdefault(STATE_PENDING, None)


def _starters(df: pd.DataFrame) -> list[str]:
    """Investigation prompts this dataset can actually support.

    Built from the schema, like the Ask suggestions, so nothing is offered
    that the engine would then refuse.
    """
    from app.agents.suggestions import detect_roles

    roles = detect_roles(df)
    if not roles.dates:
        return []

    measure = roles.primary_measure
    starters: list[str] = []
    if measure:
        label = lower_label(measure)
        starters.append(f"Why did {label} decline in the latest quarter?")
        starters.append(f"What changed in {label} most recently?")
    churn = roles.column("churn")
    if churn and churn != measure:
        starters.append("Why did churn increase in the latest period?")
    satisfaction = roles.column("satisfaction")
    if satisfaction:
        starters.append(
            f"Why did {lower_label(satisfaction)} decrease last month?"
        )
    return starters[:4]


def _queue(question: str) -> None:
    st.session_state[STATE_PENDING] = question
    st.session_state[STATE_QUESTION] = question


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #

def _render_form(session) -> None:
    cleared = False
    with st.form("investigate_form", clear_on_submit=False, border=False):
        typed = st.text_area(
            "What would you like investigated?",
            value=st.session_state.get(STATE_QUESTION, ""),
            height=80,
            placeholder="e.g. Why did revenue decline in the latest quarter?",
            label_visibility="collapsed",
        )
        left, right = st.columns([1, 4])
        submitted = left.form_submit_button(
            "Investigate", type="primary", width="stretch"
        )
        if session.investigation is not None:
            cleared = right.form_submit_button("Clear")

    if cleared:
        session.clear_investigation()
        st.session_state[STATE_PENDING] = None
        st.session_state[STATE_QUESTION] = ""
        st.rerun()
    if submitted:
        _queue(typed)
        st.rerun()


def _render_starters(df: pd.DataFrame) -> None:
    starters = _starters(df)
    if not starters:
        st.caption(
            "Investigate needs a date column to compare two periods. This "
            "dataset has none, so only Ask Aurevia is available."
        )
        return

    st.caption("Investigations this dataset supports")
    for start in range(0, len(starters), 2):
        row = starters[start: start + 2]
        for container, question in zip(st.columns(len(row)), row):
            if container.button(
                question, key=f"inv_start_{start}_{question[:24]}",
                width="stretch",
            ):
                _queue(question)
                st.rerun()


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def _render_progress(outcome: InvestigationOutcome) -> None:
    """The sequence that actually ran, ticked off."""
    result = outcome.result
    if result is None:
        return
    with st.container(border=True):
        st.markdown("**Investigation**")
        for step in result.steps_completed:
            st.markdown(f"✓ {step}")
        if outcome.plan:
            planned = len(outcome.plan.describe_steps())
            st.caption(
                f"{len(result.steps_completed)} of {planned} planned steps "
                "completed · every figure computed in Python"
            )


def _render_overall(result: InvestigationResult) -> None:
    """The headline comparison."""
    st.markdown("#### Overall change")
    tiles = st.columns(4)
    label = lower_label(result.metric)
    tiles[0].metric(
        f"{result.baseline_label} {label}", format_value(result.baseline_value)
    )
    tiles[1].metric(
        f"{result.comparison_label} {label}",
        format_value(result.comparison_value),
    )
    tiles[2].metric("Absolute change", format_value(result.absolute_change))
    tiles[3].metric(
        "Change",
        f"{result.percentage_change:+.2f}%"
        if result.percentage_change is not None else "—",
        delta=str(result.direction),
        delta_color="off",
    )


def _render_premise(outcome: InvestigationOutcome) -> None:
    """The challenge shown when the question's premise does not hold."""
    result = outcome.result
    st.warning(f"**{result.premise_message}**")
    st.caption(
        "Aurevia checks a question's premise against the data before "
        "investigating it, so a question that assumes the wrong direction is "
        "answered with the real figures rather than a plausible story."
    )
    _render_overall(result)


def _render_breakdowns(result: InvestigationResult) -> None:
    """One table per dimension, with both readings of contribution."""
    if not result.breakdowns:
        return

    st.markdown("#### Contribution by dimension")
    tabs = st.tabs([b.dimension for b in result.breakdowns])
    for tab, breakdown in zip(tabs, result.breakdowns):
        with tab:
            rows = []
            for finding in breakdown.findings:
                rows.append(
                    {
                        "#": finding.rank,
                        breakdown.dimension: finding.category,
                        result.baseline_label: finding.baseline_value,
                        result.comparison_label: finding.comparison_value,
                        "Change": finding.absolute_change,
                        "Change %": finding.percentage_change,
                        "Share of net change %": finding.contribution,
                        "Share of movement %": finding.share_of_movement,
                        "Rows": finding.total_rows,
                    }
                )
            frame = pd.DataFrame(rows)
            if breakdown.offsetting:
                # The net-share column is all nulls here; drop it rather than
                # show a column of dashes.
                frame = frame.drop(columns=["Share of net change %"])

            st.dataframe(
                frame.style.format(
                    {
                        result.baseline_label: "{:,.2f}",
                        result.comparison_label: "{:,.2f}",
                        "Change": "{:+,.2f}",
                        "Change %": "{:+,.2f}",
                        "Share of net change %": "{:,.2f}",
                        "Share of movement %": "{:,.2f}",
                        "Rows": "{:,.0f}",
                    },
                    na_rep="—",
                ),
                width="stretch",
                hide_index=True,
            )
            for note in breakdown.notes:
                st.caption(f"ℹ {note}")


def _render_ranked(result: InvestigationResult) -> None:
    """The cross-dimension leaderboard, and anything moving the other way."""
    ranked = result.ranked_findings(limit=5)
    if not ranked:
        return

    direction = (
        "negative" if result.direction is Direction.DOWN else "positive"
    )
    left, right = st.columns(2)

    with left:
        st.markdown(f"#### Strongest {direction} contributors")
        for position, finding in enumerate(ranked, start=1):
            share = (
                f"{finding.contribution:.1f}% of the net change"
                if finding.contribution is not None
                else f"{finding.share_of_movement:.1f}% of total movement"
            )
            st.markdown(
                f"`{position}`  **{finding.category}** "
                f"({finding.dimension}) — {format_value(finding.absolute_change)}, "
                f"{share}"
            )
        st.caption(
            "Ranked across every dimension by how much each category moved. "
            "A category can appear alongside others it overlaps with."
        )

    with right:
        counter = result.counter_findings(limit=3)
        st.markdown("#### Moved against the trend")
        if not counter:
            st.caption("Every category moved in the same direction.")
        else:
            for finding in counter:
                st.markdown(
                    f"**{finding.category}** ({finding.dimension}) — "
                    f"{format_value(finding.absolute_change)}"
                )


def _render_timing(result: InvestigationResult, df: pd.DataFrame) -> None:
    """When, inside the compared window, the movement happened."""
    if not result.temporal_findings:
        return

    st.markdown("#### Timing")
    worst = result.worst_period
    if worst is not None and worst.change_vs_previous is not None:
        word = "deterioration" if result.direction is Direction.DOWN else "gain"
        st.markdown(
            f"The largest {word} occurred in **{worst.period}** "
            f"({format_value(worst.change_vs_previous)} versus the period "
            "before)."
        )

    frame = pd.DataFrame(
        [
            {
                "Period": f.period,
                lower_label(result.metric): f.value,
                "Change vs previous": f.change_vs_previous,
                "Change %": f.percentage_change,
                "Rows": f.rows,
            }
            for f in result.temporal_findings
        ]
    )
    left, right = st.columns([1.2, 1])
    with left:
        try:
            data = generate_chart_data(
                df, "line",
                x=result.time_column, y=result.metric,
                aggregation=result.aggregation,
                frequency=_finer(result.granularity),
            )
            render_chart(build_chart(data))
        except ToolError as exc:
            logger.info("Timing chart unavailable: %s", exc)
            st.caption(f"No timing chart could be drawn ({exc}).")
    with right:
        st.dataframe(
            frame.style.format(
                {
                    lower_label(result.metric): "{:,.2f}",
                    "Change vs previous": "{:+,.2f}",
                    "Change %": "{:+,.2f}",
                    "Rows": "{:,.0f}",
                },
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )


def _finer(granularity: str) -> str:
    return {
        "yearly": "quarterly", "quarterly": "monthly", "monthly": "weekly",
        "weekly": "daily", "daily": "daily",
    }.get(granularity, "monthly")


def _render_summary(outcome: InvestigationOutcome) -> None:
    """The written findings."""
    summary = outcome.summary
    if summary is None:
        return

    with st.container(border=True):
        st.markdown(f"### {summary.headline}")
        if not summary.is_grounded:
            st.warning(
                "Some figures above could not be matched to the computed "
                f"evidence ({', '.join(summary.ungrounded_numbers)}). "
                "Treat the tables below as authoritative."
            )
        for line in summary.contributors:
            st.markdown(f"- {line}")
        if summary.timing:
            st.markdown(f"**Timing.** {summary.timing}")
        if summary.caution:
            st.caption(f"⚠ {summary.caution}")


def _render_follow_ups(outcome: InvestigationOutcome) -> None:
    questions = outcome.follow_ups
    if not questions:
        return
    st.caption("Investigate next")
    for start in range(0, len(questions[:4]), 2):
        row = questions[start: start + 2]
        for container, question in zip(st.columns(len(row)), row):
            if container.button(
                question, key=f"inv_next_{start}_{question[:24]}",
                width="stretch",
            ):
                _queue(question)
                st.rerun()


def _render_outcome(
    outcome: InvestigationOutcome, df: pd.DataFrame
) -> None:
    st.markdown("---")
    st.caption(f"Investigating: *{outcome.question}*")

    if outcome.needs_clarification:
        st.info(f"**{outcome.clarification_question}**")
        return

    if outcome.failed:
        if outcome.stage is InvestigationStage.UNAVAILABLE:
            st.warning(outcome.error)
        else:
            st.error(outcome.error or "The investigation could not be completed.")
        if outcome.adjustments:
            with st.expander("What Aurevia tried"):
                for note in outcome.adjustments:
                    st.markdown(f"- {note}")
        return

    result = outcome.result

    if outcome.premise_rejected:
        _render_premise(outcome)
        _render_summary(outcome)
        render_cautions(outcome.warnings)
        render_record(result.evidence_summary, title="Evidence")
        _render_follow_ups(outcome)
        return

    left, right = st.columns([2, 1])
    with left:
        _render_summary(outcome)
    with right:
        _render_progress(outcome)

    render_cautions(outcome.warnings)
    _render_overall(result)
    _render_ranked(result)
    _render_breakdowns(result)
    _render_timing(result, df)

    if outcome.adjustments:
        with st.expander(f"Plan adjustments ({len(outcome.adjustments)})"):
            for note in outcome.adjustments:
                st.markdown(f"- {note}")

    render_record(result.evidence_summary, title="Evidence")
    with st.expander("How this investigation ran"):
        st.markdown(
            f"- **AI calls**: {outcome.llm_calls} "
            "(one to plan, one to write up — the analysis itself is Python)\n"
            f"- **Tokens**: {outcome.usage.total_tokens:,}\n"
            f"- **Time**: {outcome.elapsed_seconds:.1f}s\n"
            f"- **Dimensions decomposed**: "
            f"{', '.join(result.dimensions_inspected) or 'none'}\n"
            f"- **Figures verified**: {'yes' if outcome.is_grounded else 'no'}"
        )
        if result.skipped_dimensions:
            st.markdown("**Dimensions not used**")
            for column, reason in result.skipped_dimensions.items():
                st.caption(f"`{column}` — {reason}")

    _render_follow_ups(outcome)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def render_investigate(settings: Settings | None = None) -> None:
    """Render the Investigate page."""
    settings = settings or get_settings()
    init_state()
    session = get_session()

    st.title("Investigate")
    st.caption(
        "Ask why something changed. Aurevia verifies the premise, decomposes "
        "the change across several dimensions, and shows the evidence."
    )

    if not session.has_dataset:
        render_no_dataset_notice("Investigate")
        return

    df = session.dataframe
    service = get_llm_service(settings)
    if not service.available:
        st.warning(service.unavailable_reason)
        with st.container(border=True):
            st.markdown(
                "**What Investigate does once a key is configured**\n\n"
                "1. An AI planner picks the measure, the periods and the "
                "dimensions worth examining\n"
                "2. **Python** verifies the change actually happened\n"
                "3. **Python** decomposes it across each dimension and ranks "
                "the contributors\n"
                "4. **Python** checks when within the period the movement "
                "occurred\n"
                "5. The AI writes up the computed evidence — it never "
                "calculates, and never claims a cause"
            )
        return

    st.caption(
        f"**{session.dataset_name}** · {len(df):,} rows × {df.shape[1]} columns"
    )

    # Run any queued question first, so the controls reflect the result.
    empty = False
    pending = st.session_state.get(STATE_PENDING)
    if pending is not None:
        st.session_state[STATE_PENDING] = None
        question = str(pending).strip()
        if not question:
            empty = True
        else:
            with st.status("Investigating…", expanded=True) as status:
                st.write("Planning the investigation…")
                outcome = investigate(
                    question, df, session.profile,
                    dataset_name=session.dataset_name, llm=service,
                )
                if outcome.result is not None:
                    for step in outcome.result.steps_completed:
                        st.write(f"✓ {step}")
                status.update(
                    label=(
                        "Investigation complete" if outcome.investigated
                        else "Investigation stopped"
                    ),
                    state="complete" if outcome.investigated else "error",
                    expanded=False,
                )
            session.set_investigation(outcome)
            if outcome.investigated:
                session.add_history(_history_item(outcome))

    _render_form(session)
    if empty:
        st.warning("Please describe what you would like investigated.")

    outcome = session.investigation
    if outcome is None:
        _render_starters(df)
    else:
        _render_outcome(outcome, df)


def _history_item(outcome: InvestigationOutcome):
    """A compact session-history row for a completed investigation."""
    from app.models.context import AnalysisHistoryItem

    result = outcome.result
    return AnalysisHistoryItem(
        question=outcome.question,
        intent="investigation",
        title=f"{lower_label(result.metric)}: {result.comparison_label} vs "
              f"{result.baseline_label}",
        plan_summary=list(result.steps_completed[:6]),
        result_summary={
            "Baseline": result.baseline_value,
            "Comparison": result.comparison_value,
            "Change": result.absolute_change,
            "Change %": result.percentage_change,
            "Premise confirmed": result.premise_confirmed,
        },
        valid=result.premise_confirmed,
        strength="moderate" if result.warnings else "strong",
        source="investigate",
    )
