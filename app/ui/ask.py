"""The Ask Aurevia page.

An analytical command interface, not a chat window: one question, one answer,
shown as a computed result with its plan, figures, chart and table. There is no
transcript — Phase 3 answers each question independently, and conversational
memory is deliberately out of scope.

The page runs the pipeline only on an explicit submit and keeps the outcome in
session state, so Streamlit's reruns never re-issue an API call.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.config import Settings, get_settings
from app.models.plans import AnalysisPlan
from app.services.ask_pipeline import AskOutcome, Stage, answer_question
from app.services.llm_service import get_llm_service
from app.services.session_manager import get_session
from app.services.visualization import build_chart
from app.tools import ToolError, generate_chart_data
from app.agents.suggestions import suggest_questions
from app.ui.components import render_chart
from app.ui.evidence import (
    format_value,
    render_cautions,
    render_record,
    render_strength,
    render_validation,
)
from app.ui.state import render_no_dataset_notice

logger = logging.getLogger(__name__)

#: The text to show in the question box. A plain session key, deliberately
#: *not* the text area's widget key: Streamlit forbids writing to a widget's
#: own key once that widget has been instantiated, and the suggestion buttons
#: render after the box. The text area is therefore unkeyed and reads this.
STATE_QUESTION = "ask_question_text"
STATE_PENDING = "ask_pending_question"

#: Metric tiles per row in the headline figures.
TILES_PER_ROW = 4
#: Summary entries shown as tiles before the rest move into a list.
MAX_TILES = 8


def init_state() -> None:
    st.session_state.setdefault(STATE_QUESTION, "")
    st.session_state.setdefault(STATE_PENDING, None)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def _queue(question: str) -> None:
    """Queue a question to run on the next rerun, and show it in the box."""
    st.session_state[STATE_PENDING] = question
    st.session_state[STATE_QUESTION] = question


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #

def _render_question_form(df: pd.DataFrame) -> None:
    cleared = False
    with st.form("ask_form", clear_on_submit=False, border=False):
        typed = st.text_area(
            "Ask a question about this dataset",
            value=st.session_state.get(STATE_QUESTION, ""),
            height=88,
            placeholder="e.g. Which region generated the most revenue?",
            label_visibility="collapsed",
        )
        left, right = st.columns([1, 4])
        submitted = left.form_submit_button(
            "Analyze", type="primary", width="stretch"
        )
        if get_session().last_outcome is not None:
            cleared = right.form_submit_button("Clear")

    if cleared:
        get_session().clear_last_outcome()
        st.session_state[STATE_PENDING] = None
        st.session_state[STATE_QUESTION] = ""
        st.rerun()
    if submitted:
        _queue(typed)
        st.rerun()


def _render_suggestions(df: pd.DataFrame) -> None:
    questions = suggest_questions(df)
    if not questions:
        return

    st.caption("Suggested questions for this dataset")
    # Two rows of three keeps the labels readable at any window width.
    for start in range(0, len(questions), 3):
        row = questions[start: start + 3]
        for container, question in zip(st.columns(len(row)), row):
            if container.button(
                question, key=f"ask_suggest_{start}_{question[:24]}",
                width="stretch",
            ):
                _queue(question)
                st.rerun()


# --------------------------------------------------------------------------- #
# Output sections
# --------------------------------------------------------------------------- #

def _render_plan(plan: AnalysisPlan, outcome: AskOutcome) -> None:
    """The user-facing execution summary. Never model reasoning."""
    with st.container(border=True):
        st.markdown("**Analysis plan**")
        for index, step in enumerate(plan.describe_steps(), start=1):
            st.markdown(f"`{index:02d}`  {step}")

        facts = [f"Analysis: {plan.intent_label}"]
        if plan.metric:
            facts.append(f"Measure: `{plan.metric}`")
        if plan.dimensions:
            facts.append("Grouped by: " + ", ".join(f"`{d}`" for d in plan.dimensions))
        if plan.time_column:
            facts.append(
                f"Time: `{plan.time_column}` by {plan.time_granularity or 'month'}"
            )
        if plan.filters:
            facts.append(
                "Filtered: " + "; ".join(f.describe() for f in plan.filters)
            )
        st.caption(" · ".join(facts))


def _render_adjustments(outcome: AskOutcome) -> None:
    adjustments = outcome.validation.adjustments if outcome.validation else []
    if not adjustments:
        return
    with st.expander(f"Aurevia adjusted the plan ({len(adjustments)})"):
        for note in adjustments:
            st.markdown(f"- {note}")


def _render_answer(outcome: AskOutcome) -> None:
    insight = outcome.insight
    if insight is None:
        return

    with st.container(border=True):
        st.markdown(f"### {insight.answer}")
        if not insight.is_grounded:
            st.warning(
                "Some figures above could not be matched to the computed "
                f"results ({', '.join(insight.ungrounded_numbers)}). "
                "Treat the table and chart below as authoritative."
            )
        if insight.observations:
            for observation in insight.observations:
                st.markdown(f"- {observation}")
        if insight.caveat:
            st.caption(f"⚠ {insight.caveat}")


def _render_figures(outcome: AskOutcome) -> None:
    result = outcome.result
    if result is None or not result.summary_data:
        return

    items = list(result.summary_data.items())
    tiles, listed = items[:MAX_TILES], items[MAX_TILES:]

    for start in range(0, len(tiles), TILES_PER_ROW):
        row = tiles[start: start + TILES_PER_ROW]
        for container, (label, value) in zip(st.columns(len(row)), row):
            container.metric(label, format_value(value))

    if listed:
        with st.expander(f"{len(listed)} more computed figure(s)"):
            for label, value in listed:
                st.markdown(f"- **{label}**: {format_value(value)}")


def _render_chart(outcome: AskOutcome, df: pd.DataFrame) -> None:
    result = outcome.result
    if result is None or not result.has_chart:
        return

    try:
        data = generate_chart_data(df, **result.chart_spec)
    except ToolError as exc:
        logger.info("Chart preparation failed for an answer: %s", exc)
        st.caption(f"No chart could be drawn for this result ({exc}).")
        return
    except Exception:  # noqa: BLE001 - a missing chart must not break the page
        logger.exception("Unexpected chart failure on the Ask page")
        st.caption("No chart could be drawn for this result.")
        return

    if data.is_empty:
        return
    render_chart(build_chart(data))
    for note in data.notes:
        st.caption(note)


def _render_table(outcome: AskOutcome) -> None:
    result = outcome.result
    if result is None or not result.has_table:
        return
    frame = pd.DataFrame(result.table_data)
    if frame.empty:
        return

    label = f"Result table ({len(frame):,} row{'s' if len(frame) != 1 else ''})"
    with st.expander(label, expanded=len(frame) <= 12):
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Download as CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="aurevia_answer.csv",
            mime="text/csv",
            key="ask_download",
        )


def _render_follow_ups(outcome: AskOutcome) -> None:
    questions = outcome.follow_ups
    if not questions:
        return
    st.caption("Ask next")
    for start in range(0, len(questions[:4]), 2):
        row = questions[start: start + 2]
        for container, question in zip(st.columns(len(row)), row):
            if container.button(
                question, key=f"ask_followup_{start}_{question[:24]}",
                width="stretch",
            ):
                _queue(question)
                st.rerun()


def _render_provenance(outcome: AskOutcome) -> None:
    """How the answer was produced — the audit trail for one question."""
    result = outcome.result
    with st.expander("How this answer was produced"):
        rows = {
            "Analysis": outcome.plan.intent_label if outcome.plan else "—",
            "Python functions": ", ".join(result.tools_used) if result else "—",
            "Rows analysed": f"{result.metadata.get('rows_analysed', 0):,}"
            if result else "—",
            "Chart": (result.metadata.get("visualization") or "none")
            if result else "none",
            "Figures verified": "yes" if outcome.is_grounded else "no",
            "Model": outcome.usage.model or get_settings().openai_model,
            "Tokens used": f"{outcome.usage.total_tokens:,}",
            "Time": f"{outcome.elapsed_seconds:.1f}s",
        }
        rows["AI calls"] = str(outcome.llm_calls)
        if outcome.retries:
            rows["Retries"] = f"{outcome.retries} ({'; '.join(outcome.retry_log)})"
        if outcome.verification is not None:
            rows["Validation checks"] = str(outcome.verification.checks_run)
        if outcome.context is not None:
            rows["Dataset context sent"] = (
                f"~{outcome.context.approx_tokens():,} tokens (schema only, no rows)"
            )
        resolution = outcome.context_resolution
        if resolution is not None and resolution.is_continuation:
            rows["Read as a follow-up"] = ", ".join(resolution.signals)
        for label, value in rows.items():
            st.markdown(f"- **{label}**: {value}")

        st.caption(
            "The LLM planned the analysis and wrote the explanation. Every "
            "figure shown was computed by Python from the loaded dataset."
        )


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #

def _render_outcome(outcome: AskOutcome, df: pd.DataFrame) -> None:
    st.markdown("---")
    st.caption(f"Question: *{outcome.question}*")

    if outcome.needs_clarification:
        st.info(f"**{outcome.clarification_question}**")
        if outcome.error:
            st.caption(outcome.error)
        st.caption(
            "Rephrase your question with the measure or field you have in mind."
        )
        _render_adjustments(outcome)
        return

    if outcome.failed:
        if outcome.stage is Stage.UNAVAILABLE:
            st.warning(outcome.error)
        else:
            st.error(outcome.error or "Aurevia could not answer that question.")
        render_validation(outcome.verification)
        # A validation failure is not a lost computation: the figures Python
        # produced are still shown, clearly marked as unverified.
        if outcome.computed:
            st.caption(
                "The figures below were computed, but the answer did not pass "
                "validation. Read them with that in mind."
            )
            _render_figures(outcome)
            _render_table(outcome)
            render_record(outcome.evidence, title="Evidence")
        if outcome.plan is not None:
            _render_plan(outcome.plan, outcome)
        _render_adjustments(outcome)
        return

    # A successful answer.
    if outcome.plan is not None:
        left, right = st.columns([2, 1])
        with left:
            _render_answer(outcome)
            render_strength(outcome.verification)
        with right:
            _render_plan(outcome.plan, outcome)
    else:  # pragma: no cover - a result always has a plan
        _render_answer(outcome)

    render_cautions(outcome.context_notes, label="↩")
    render_cautions(outcome.warnings)

    st.markdown(f"#### {outcome.result.title}")
    _render_figures(outcome)
    _render_chart(outcome, df)
    _render_table(outcome)
    _render_adjustments(outcome)
    render_validation(outcome.verification)
    render_record(outcome.evidence, title="Evidence")
    _render_follow_ups(outcome)
    _render_provenance(outcome)


def _render_unavailable(settings: Settings) -> None:
    service = get_llm_service(settings)
    st.warning(service.unavailable_reason)
    with st.container(border=True):
        st.markdown(
            "**What Ask Aurevia does once a key is configured**\n\n"
            "1. An AI planner reads your question and maps it to this "
            "dataset's fields\n"
            "2. The plan is validated against the real columns and types\n"
            "3. **Python** computes the answer with the deterministic analysis "
            "tools\n"
            "4. A chart is chosen to match the result\n"
            "5. The AI explains the computed figures — it never calculates them"
        )
    st.caption(
        "Add `OPENAI_API_KEY=sk-…` to `.env` in the project root, then restart. "
        "See `.env.example` for the full list of settings."
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def render_ask(settings: Settings | None = None) -> None:
    """Render the Ask Aurevia page."""
    settings = settings or get_settings()
    init_state()

    st.title("Ask Aurevia")
    st.caption(
        "Ask a question in plain language. Aurevia plans the analysis, computes "
        "it in Python, and explains the result."
    )

    session = get_session()
    if not session.has_dataset:
        render_no_dataset_notice("Ask Aurevia")
        return

    df = session.dataframe
    profile = session.profile

    service = get_llm_service(settings)
    if not service.available:
        _render_unavailable(settings)
        return

    st.caption(
        f"**{session.dataset_name}** · {len(df):,} rows × {df.shape[1]} columns"
    )
    context = session.context
    if not context.is_empty:
        st.caption(
            f"↩ Following on from: **{context.describe()}** — a self-contained "
            "question starts fresh."
        )

    # Run any queued question *first*, so the controls below are rendered
    # against the resulting state: otherwise the Clear button and the
    # suggestion list would both reflect the previous run.
    empty_question = False
    # `None` means "nothing queued"; an empty string means the user pressed
    # Analyze with an empty box, which deserves a message.
    pending = st.session_state.get(STATE_PENDING)
    if pending is not None:
        st.session_state[STATE_PENDING] = None
        question = str(pending).strip()
        if not question:
            empty_question = True
        else:
            with st.spinner("Planning the analysis, computing it, and writing up…"):
                outcome = answer_question(
                    question,
                    df,
                    profile,
                    dataset_name=session.dataset_name,
                    llm=service,
                    analytical_context=context,
                    dataset_key=session.dataset_key,
                )
            session.set_last_outcome(outcome)
            # Only a computed answer updates the conversation; a refusal or a
            # clarification leaves the previous context in place.
            if outcome.updated_context is not None:
                session.set_context(outcome.updated_context)
            if outcome.answered:
                session.add_history(outcome.to_history())

    _render_question_form(df)
    if empty_question:
        st.warning("Please enter a question first.")

    last = session.last_outcome
    if last is None:
        _render_suggestions(df)
    else:
        _render_outcome(last, df)
