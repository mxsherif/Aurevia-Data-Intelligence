"""The Overview page: load a dataset, then see what DataPilot understood.

This is the only page in Phase 1. It owns the upload / sample-data controls and
renders the profile produced by :mod:`app.services.profiler`.
"""

from __future__ import annotations

import logging

import streamlit as st

from app.config import SAMPLE_DATASET_PATH, Settings, get_settings
from app.models.profile import DatasetProfile
from app.services.data_loader import DataLoadError, LoadResult, load_dataframe
from app.services.profiler import profile_dataframe
from app.ui.components import (
    categorical_table,
    correlation_table,
    field_type_table,
    human_bytes,
    missing_value_table,
    numeric_stats_table,
    render_metrics,
    render_warnings,
)

logger = logging.getLogger(__name__)

STATE_LOAD = "load_result"
STATE_PROFILE = "profile"
STATE_ERROR = "load_error"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _profile_cached(
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


def _handle_upload(uploaded, settings: Settings) -> None:
    try:
        payload = uploaded.getvalue()
    except Exception as exc:  # noqa: BLE001
        _fail(f"Could not read the uploaded file: {exc}")
        return
    try:
        result, profile = _profile_cached(payload, uploaded.name, settings.max_rows)
    except DataLoadError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - never crash the page
        logger.exception("Unexpected failure while profiling %s", uploaded.name)
        _fail(f"Unexpected error while reading '{uploaded.name}': {exc}")
    else:
        _store(result, profile)


def _handle_sample(settings: Settings) -> None:
    if not SAMPLE_DATASET_PATH.exists():
        _fail(
            "The sample dataset is missing. Generate it with:\n\n"
            "`python datasets/generate_sample_data.py`"
        )
        return
    try:
        payload = SAMPLE_DATASET_PATH.read_bytes()
        result, profile = _profile_cached(
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

def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.subheader("Data source")

        uploaded = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help=f"Maximum {settings.max_upload_mb} MB.",
        )
        if uploaded is not None:
            _handle_upload(uploaded, settings)

        st.markdown("---")
        st.caption("No file handy? Start from the bundled telecom dataset.")
        if st.button("Load sample dataset", width="stretch"):
            _handle_sample(settings)

        if st.session_state.get(STATE_LOAD) is not None:
            if st.button("Clear dataset", width="stretch"):
                for key in (STATE_LOAD, STATE_PROFILE, STATE_ERROR):
                    st.session_state[key] = None
                st.rerun()

        st.markdown("---")
        st.subheader("Environment")
        if settings.has_api_key:
            st.success("OpenAI API key detected.")
        else:
            st.info(
                "No OpenAI API key configured. Profiling works without one; "
                "AI analysis arrives in a later phase."
            )
        st.caption(f"Model (reserved): `{settings.openai_model}`")
        st.caption(f"Row cap: {settings.max_rows:,} · Preview: {settings.preview_rows} rows")


# --------------------------------------------------------------------------- #
# Main body
# --------------------------------------------------------------------------- #

def _render_empty_state() -> None:
    st.info("Upload a CSV or Excel file from the sidebar, or load the sample dataset.")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "#### What this page does\n"
            "- Reads CSV, XLSX and XLS files\n"
            "- Detects each field's semantic type\n"
            "- Measures missing values, duplicates and outliers\n"
            "- Surfaces data-quality warnings before you analyse anything"
        )
    with right:
        st.markdown(
            "#### Coming in later phases\n"
            "- Natural-language questions about your data\n"
            "- Automatic visualisations\n"
            "- Anomaly detection and forecasting\n"
            "- A guided ML lab"
        )


def _render_source_summary(result: LoadResult) -> None:
    parts = [
        f"**{result.source_name}**",
        f"{result.rows:,} rows x {result.columns} columns",
        f"{result.file_format.upper()}",
        human_bytes(result.size_bytes),
    ]
    if result.sheet_name:
        parts.append(f"sheet `{result.sheet_name}`")
    if result.encoding:
        parts.append(f"encoding `{result.encoding}`")
    st.caption(" · ".join(parts))

    for note in result.notes:
        st.warning(note)
    if result.renamed_columns:
        with st.expander(f"{len(result.renamed_columns)} column(s) were renamed"):
            st.dataframe(
                {
                    "Original": list(result.renamed_columns),
                    "Renamed to": list(result.renamed_columns.values()),
                },
                width="stretch",
                hide_index=True,
            )


def render_overview(settings: Settings | None = None) -> None:
    """Render the whole Overview page."""
    settings = settings or get_settings()

    st.title("Dataset Overview")
    st.caption("Load a dataset and DataPilot will profile its structure and quality.")

    error = st.session_state.get(STATE_ERROR)
    if error:
        st.error(error)

    result: LoadResult | None = st.session_state.get(STATE_LOAD)
    profile: DatasetProfile | None = st.session_state.get(STATE_PROFILE)

    if result is None or profile is None:
        _render_empty_state()
        return

    _render_source_summary(result)
    st.markdown("---")
    render_metrics(profile)
    st.markdown("---")

    preview_tab, types_tab, quality_tab, stats_tab = st.tabs(
        ["Preview", "Field types", "Data quality", "Statistics"]
    )

    with preview_tab:
        st.subheader("Data preview")
        rows = min(settings.preview_rows, profile.row_count)
        st.caption(f"First {rows:,} of {profile.row_count:,} rows.")
        st.dataframe(result.dataframe.head(rows), width="stretch")

    with types_tab:
        st.subheader("Detected field types")
        st.dataframe(field_type_table(profile), width="stretch", hide_index=True)

        groups = {
            "Numeric": profile.numeric_columns,
            "Categorical": profile.categorical_columns,
            "Boolean": profile.boolean_columns,
            "Date / time": profile.date_columns,
            "Likely identifiers": profile.id_columns,
            "Low cardinality": profile.low_cardinality_columns,
        }
        st.markdown("##### Grouped by role")
        for label, columns in groups.items():
            if columns:
                st.markdown(f"**{label}** ({len(columns)}): " + ", ".join(f"`{c}`" for c in columns))
            else:
                st.markdown(f"**{label}**: _none detected_")

    with quality_tab:
        st.subheader("Data-quality warnings")
        render_warnings(profile)

        st.markdown("##### Missing values by column")
        missing = missing_value_table(profile)
        if missing.empty:
            st.success("No missing values.")
        else:
            st.dataframe(missing, width="stretch", hide_index=True)

        st.markdown("##### Duplicates")
        if profile.duplicate_row_count:
            st.warning(
                f"{profile.duplicate_row_count:,} duplicate row(s) "
                f"({profile.duplicate_row_pct:.2f}%)."
            )
            duplicated = result.dataframe[result.dataframe.duplicated(keep=False)]
            st.dataframe(duplicated.head(50), width="stretch")
        else:
            st.success("No duplicate rows.")

    with stats_tab:
        st.subheader("Numeric statistics")
        numeric = numeric_stats_table(profile)
        if numeric.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(numeric, width="stretch", hide_index=True)

        st.subheader("Categorical summary")
        categorical = categorical_table(profile)
        if categorical.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(categorical, width="stretch", hide_index=True)

        st.subheader("Date ranges")
        date_columns = [profile.column(c) for c in profile.date_columns]
        if not date_columns:
            st.info("No date columns detected.")
        else:
            st.dataframe(
                [
                    {
                        "Column": col.name,
                        "From": col.min_date,
                        "To": col.max_date,
                        "Span (days)": col.date_range_days,
                    }
                    for col in date_columns
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Strong correlations")
        correlations = correlation_table(profile)
        if correlations.empty:
            st.info("No strongly correlated numeric pairs found.")
        else:
            st.dataframe(correlations, width="stretch", hide_index=True)
