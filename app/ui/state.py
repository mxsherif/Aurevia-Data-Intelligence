"""Shared session state and the data-source sidebar.

Phase 2 added two more pages, and all three need the same loaded dataset. The
loading, caching, and error handling that used to live inside the Overview page
moved here so every page reads one source of truth.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.config import SAMPLE_DATASET_PATH, Settings, get_settings
from app.models.profile import DatasetProfile
from app.services.data_loader import DataLoadError, LoadResult, load_dataframe
from app.services.profiler import profile_dataframe

logger = logging.getLogger(__name__)

STATE_LOAD = "load_result"
STATE_PROFILE = "profile"
STATE_ERROR = "load_error"
STATE_PAGE = "active_page"

STATE_KEYS = (STATE_LOAD, STATE_PROFILE, STATE_ERROR)


def init_state() -> None:
    for key in STATE_KEYS:
        st.session_state.setdefault(key, None)


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #

def get_load_result() -> LoadResult | None:
    return st.session_state.get(STATE_LOAD)


def get_profile() -> DatasetProfile | None:
    return st.session_state.get(STATE_PROFILE)


def get_dataframe() -> pd.DataFrame | None:
    result = get_load_result()
    return result.dataframe if result is not None else None


def get_error() -> str | None:
    return st.session_state.get(STATE_ERROR)


def has_dataset() -> bool:
    return get_load_result() is not None and get_profile() is not None


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
    st.session_state[STATE_LOAD] = result
    st.session_state[STATE_PROFILE] = profile
    st.session_state[STATE_ERROR] = None


def _fail(message: str) -> None:
    st.session_state[STATE_LOAD] = None
    st.session_state[STATE_PROFILE] = None
    st.session_state[STATE_ERROR] = message


def clear_dataset() -> None:
    for key in STATE_KEYS:
        st.session_state[key] = None


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
                "No OpenAI API key configured. Everything in Phase 2 is "
                "deterministic; AI analysis arrives in a later phase."
            )
        st.caption(f"Model (reserved): `{settings.openai_model}`")
        st.caption(f"Row cap: {settings.max_rows:,} · Preview: {settings.preview_rows} rows")
