"""Small presentation helpers shared by DataPilot's pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.models.profile import DatasetProfile, FieldType, WarningSeverity

#: Emoji + label used when rendering an inferred field type.
TYPE_BADGES: dict[FieldType, str] = {
    FieldType.NUMERIC: "numeric",
    FieldType.INTEGER: "integer",
    FieldType.CATEGORICAL: "categorical",
    FieldType.BOOLEAN: "boolean",
    FieldType.DATETIME: "date / time",
    FieldType.TEXT: "free text",
    FieldType.IDENTIFIER: "identifier",
    FieldType.EMPTY: "empty",
    FieldType.UNKNOWN: "unknown",
}

_SEVERITY_RENDERERS = {
    WarningSeverity.CRITICAL: ("error", "Critical"),
    WarningSeverity.WARNING: ("warning", "Warning"),
    WarningSeverity.INFO: ("info", "Info"),
}


def human_bytes(size: float) -> str:
    """Render a byte count as a short human-readable string."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


def render_metrics(profile: DatasetProfile) -> None:
    """The headline metric row."""
    row_one = st.columns(4)
    row_one[0].metric("Rows", f"{profile.row_count:,}")
    row_one[1].metric("Columns", f"{profile.column_count:,}")
    row_one[2].metric("Memory", human_bytes(profile.memory_usage_bytes))
    row_one[3].metric(
        "Missing cells",
        f"{profile.missing_cells_pct:.2f}%",
        delta=f"{profile.missing_cells:,} cells" if profile.missing_cells else None,
        delta_color="inverse",
    )

    row_two = st.columns(4)
    row_two[0].metric(
        "Duplicate rows",
        f"{profile.duplicate_row_count:,}",
        delta=f"{profile.duplicate_row_pct:.2f}%" if profile.duplicate_row_count else None,
        delta_color="inverse",
    )
    row_two[1].metric("Numeric fields", len(profile.numeric_columns))
    row_two[2].metric("Categorical fields", len(profile.categorical_columns))
    row_two[3].metric("Date fields", len(profile.date_columns))


def render_warnings(profile: DatasetProfile) -> None:
    """Data-quality findings, most severe first."""
    if not profile.warnings:
        st.success("No data-quality issues detected.")
        return

    counts = {
        severity: len(profile.warnings_by_severity(severity))
        for severity in WarningSeverity
    }
    summary = " · ".join(
        f"{count} {_SEVERITY_RENDERERS[severity][1].lower()}"
        for severity, count in counts.items()
        if count
    )
    st.caption(summary)

    for warning in profile.warnings:
        renderer, label = _SEVERITY_RENDERERS[warning.severity]
        getattr(st, renderer)(f"**{label} — {warning.code}**  \n{warning.message}")
        if warning.columns:
            shown = ", ".join(f"`{c}`" for c in warning.columns[:12])
            more = f" … and {len(warning.columns) - 12} more" if len(warning.columns) > 12 else ""
            st.caption(f"Affected columns: {shown}{more}")


def field_type_table(profile: DatasetProfile) -> pd.DataFrame:
    """One row per column: detected type, nulls, cardinality, range."""
    rows = []
    for col in profile.columns:
        if col.is_numeric and col.min is not None:
            value_range = f"{col.min:,.2f} … {col.max:,.2f}"
        elif col.is_datetime and col.min_date:
            value_range = f"{col.min_date} … {col.max_date}"
        elif col.top_values:
            value_range = ", ".join(list(col.top_values)[:3])
        else:
            value_range = ", ".join(col.sample_values[:3])

        flags = []
        if col.is_likely_id:
            flags.append("ID")
        if col.is_constant:
            flags.append("constant")
        if col.is_low_cardinality:
            flags.append("low-cardinality")
        if col.outlier_count:
            flags.append(f"{col.outlier_count} outliers")

        rows.append(
            {
                "Column": col.name,
                "Detected type": TYPE_BADGES[col.inferred_type],
                "Pandas dtype": col.dtype,
                "Non-null": col.count,
                "Missing": col.missing_count,
                "Missing %": round(col.missing_pct, 2),
                "Unique": col.unique_count,
                "Range / top values": value_range,
                "Flags": ", ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def missing_value_table(profile: DatasetProfile) -> pd.DataFrame:
    rows = [
        {
            "Column": col.name,
            "Missing": col.missing_count,
            "Missing %": round(col.missing_pct, 2),
            "Non-null": col.count,
        }
        for col in profile.columns
        if col.missing_count
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values("Missing", ascending=False) if not frame.empty else frame


def numeric_stats_table(profile: DatasetProfile) -> pd.DataFrame:
    rows = [
        {
            "Column": col.name,
            "Min": col.min,
            "Q1": col.q1,
            "Median": col.median,
            "Mean": col.mean,
            "Q3": col.q3,
            "Max": col.max,
            "Std": col.std,
            "Skew": col.skew,
            "Zeros": col.zeros,
            "Negatives": col.negatives,
            "Outliers": col.outlier_count,
        }
        for col in profile.columns
        if col.is_numeric
    ]
    return pd.DataFrame(rows)


def categorical_table(profile: DatasetProfile) -> pd.DataFrame:
    rows = [
        {
            "Column": col.name,
            "Unique": col.unique_count,
            "Top value": next(iter(col.top_values), ""),
            "Top count": next(iter(col.top_values.values()), 0),
            "Most frequent": ", ".join(
                f"{k} ({v:,})" for k, v in list(col.top_values.items())[:5]
            ),
        }
        for col in profile.columns
        if col.is_categorical or col.is_boolean
    ]
    return pd.DataFrame(rows)


def correlation_table(profile: DatasetProfile) -> pd.DataFrame:
    if not profile.correlations:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Column A": pair["left"],
                "Column B": pair["right"],
                "Correlation": pair["correlation"],
                "Direction": pair["direction"],
            }
            for pair in profile.correlations
        ]
    )
