"""Dataset loading and the shared data-source sidebar.

The *state* itself lives in :mod:`app.services.session_manager`; this module is
the loading path and the sidebar that drives it. The accessors here are thin
wrappers kept for the pages that already use them.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.config import SAMPLE_DATASET_PATH, Settings, get_settings
from app.models.profile import DatasetProfile
from app.services.data_loader import DataLoadError, LoadResult, load_dataframe
from app.services.profiler import profile_dataframe
from app.services.session_manager import get_session

logger = logging.getLogger(__name__)

STATE_LOAD = "load_result"
STATE_PROFILE = "profile"
STATE_ERROR = "load_error"
STATE_PAGE = "active_page"

STATE_KEYS = (STATE_LOAD, STATE_PROFILE, STATE_ERROR)


def init_state() -> None:
    """Seed every session key. Safe to call on every rerun."""
    get_session().init()


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #

def get_load_result() -> LoadResult | None:
    return get_session().load_result


def get_profile() -> DatasetProfile | None:
    return get_session().profile


def get_dataframe() -> pd.DataFrame | None:
    return get_session().dataframe


def get_error() -> str | None:
    return get_session().error


def has_dataset() -> bool:
    return get_session().has_dataset


def get_dataset_key() -> str | None:
    """The content hash of the loaded dataset, scoping analytical context."""
    return get_session().dataset_key


def render_no_dataset_notice(page: str) -> None:
    """Consistent empty state for the pages that require a dataset."""
    st.info(
        f"Load a dataset to use {page}. Upload a CSV or Excel file from the "
        "sidebar, or load the bundled sample dataset."
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_and_profile(
    payload: bytes, name: str, max_rows: int
) -> tuple[LoadResult, DatasetProfile]:
    """Load + profile a file, memoised on its bytes so reruns are instant."""
    result = load_dataframe(payload, name=name, max_rows=max_rows)
    profile = profile_dataframe(
        result.dataframe,
        name=name,
        source=result.file_format,
        truncated=result.truncated,
        original_row_count=result.original_rows,
        notes=result.notes,
    )
    return result, profile


def _store(result: LoadResult, profile: DatasetProfile) -> None:
    """Install a dataset, resetting the analytical session if it changed.

    Delegated to the session manager so that context, history, the last
    answer and any investigation are cleared in exactly one place -- stale
    context from another dataset is the worst thing this app could carry.
    """
    session = get_session()
    changed = session.set_dataset(result, profile)
    if changed:
        logger.info("Dataset changed to %s; session reset", result.source_name)


def _fail(message: str) -> None:
    get_session().set_error(message)


def clear_dataset() -> None:
    get_session().clear_dataset()


def _handle_upload(uploaded, settings: Settings) -> None:
    try:
        payload = uploaded.getvalue()
    except Exception as exc:  # noqa: BLE001
        _fail(f"Could not read the uploaded file: {exc}")
        return
    try:
        result, profile = load_and_profile(payload, uploaded.name, settings.max_rows)
    except DataLoadError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - never crash the page
        logger.exception("Unexpected failure while profiling %s", uploaded.name)
        _fail(f"Unexpected error while reading '{uploaded.name}': {exc}")
    else:
        _store(result, profile)


def load_sample_dataset(settings: Settings) -> None:
    """Load the bundled sample dataset into session state."""
    if not SAMPLE_DATASET_PATH.exists():
        _fail(
            "The sample dataset is missing. Generate it with:\n\n"
            "`python datasets/generate_sample_data.py`"
        )
        return
    try:
        payload = SAMPLE_DATASET_PATH.read_bytes()
        result, profile = load_and_profile(
            payload, SAMPLE_DATASET_PATH.name, settings.max_rows
        )
    except DataLoadError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure while loading the sample dataset")
        _fail(f"Unexpected error while loading the sample dataset: {exc}")
    else:
        _store(result, profile)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_data_source_sidebar(settings: Settings | None = None) -> None:
    """The upload / sample-data / environment controls shared by every page."""
    settings = settings or get_settings()

    with st.sidebar:
        st.subheader("Data source")

        uploaded = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help=f"Maximum {settings.max_upload_mb} MB.",
        )
        if uploaded is not None:
            _handle_upload(uploaded, settings)

        st.caption("No file handy? Start from the bundled telecom dataset.")
        if st.button("Load sample dataset", width="stretch"):
            load_sample_dataset(settings)

        if has_dataset():
            result = get_load_result()
            st.success(f"**{result.source_name}**  \n{result.rows:,} rows x {result.columns} columns")
            if st.button("Clear dataset", width="stretch"):
                clear_dataset()
                st.rerun()

        st.markdown("---")
        st.subheader("Environment")
        if settings.has_api_key:
            st.success("OpenAI API key detected.")
        else:
            st.info(
                "No OpenAI API key configured. Overview, Explore and Visualize "
                "work without one; Ask Aurevia and Investigate need it."
            )
        st.caption(f"Model: `{settings.openai_model}`")
        st.caption(f"Row cap: {settings.max_rows:,} · Preview: {settings.preview_rows} rows")

        _render_session_panel()


def _render_session_panel() -> None:
    """The analytical session: what is remembered, and a way to forget it."""
    session = get_session()
    if not session.has_dataset:
        return

    context = session.context
    history = session.history
    if context.is_empty and not history:
        return

    st.markdown("---")
    st.subheader("Session")
    if not context.is_empty:
        st.caption(f"Following on from: **{context.describe()}**")
        st.caption(f"{context.turn_count} question(s) answered")
    if history:
        with st.expander(f"Recent analyses ({len(history)})"):
            for item in session.recent_history(limit=8):
                st.markdown(f"**{item.when}** · {item.question}")
                headline = item.headline()
                if headline:
                    st.caption(f"{item.title or item.intent} — {headline}")
                else:
                    st.caption(item.title or item.intent)
    if st.button("Reset conversation", width="stretch",
                 help="Forget the previous questions. The dataset stays loaded."):
        session.reset_context()
        session.clear_history()
        session.clear_last_outcome()
        session.clear_investigation()
        st.rerun()
