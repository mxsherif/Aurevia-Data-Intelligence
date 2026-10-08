"""The Overview page: load a dataset, then see what Aurevia understood.

The dataset loading and the data-source sidebar live in :mod:`app.ui.state`,
shared with the Explore and Visualize pages; this module is purely the Overview
rendering.
"""

from __future__ import annotations

import logging

import streamlit as st

from app.config import Settings, get_settings
from app.models.profile import DatasetProfile
from app.services.data_loader import LoadResult
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
from app.ui.state import (
    get_error,
    get_load_result,
    get_profile,
    render_data_source_sidebar,
)

logger = logging.getLogger(__name__)


def render_sidebar(settings: Settings | None = None) -> None:
    """Backwards-compatible alias for the shared data-source sidebar."""
    render_data_source_sidebar(settings)


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
            "#### Where to go next\n"
            "- **Explore** — distributions, correlations, outliers\n"
            "- **Visualize** — build a chart from any columns\n"
            "- _Later phases_: natural-language analysis, "
            "anomaly detection, forecasting, ML Lab"
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
    st.caption("Load a dataset and Aurevia will profile its structure and quality.")

    error = get_error()
    if error:
        st.error(error)

    result: LoadResult | None = get_load_result()
    profile: DatasetProfile | None = get_profile()

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
