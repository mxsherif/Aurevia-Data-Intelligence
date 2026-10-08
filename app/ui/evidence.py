"""The Evidence view.

A transparent record of what was executed: which fields, which filters, which
aggregation, which periods, how many rows, and the figures that came out. This
is not hidden reasoning — it is the audit trail for an answer, so a reader can
check the work rather than take it on trust.

Shared by Ask Aurevia and Investigate, which need the same thing in slightly
different shapes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.models.validation import EvidenceStrength, Severity, ValidationResult

#: How a strength label is rendered.
_STRENGTH_ICON = {
    EvidenceStrength.STRONG: "●●●",
    EvidenceStrength.MODERATE: "●●○",
    EvidenceStrength.LIMITED: "●○○",
}

#: Why each label was assigned, shown so the label is not a bare assertion.
STRENGTH_EXPLANATION = {
    EvidenceStrength.STRONG: (
        "Every validation check passed and the sample is large enough to rely "
        "on."
    ),
    EvidenceStrength.MODERATE: (
        "The analysis ran correctly, but a check raised a caution or the "
        "sample is small."
    ),
    EvidenceStrength.LIMITED: (
        "A validation check failed, or there are too few records to rely on "
        "this result."
    ),
}


def format_value(value: Any) -> str:
    """Render a computed figure compactly, without losing its magnitude."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:,.2f}M"
        if abs(value) >= 10_000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, dict):
        return ", ".join(f"{k}: {format_value(v)}" for k, v in list(value.items())[:4])
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in list(value)[:6]) or "—"
    text = str(value)
    return text if len(text) <= 60 else text[:59] + "…"


def render_strength(validation: ValidationResult | None) -> None:
    """The evidence-strength label, with the reason it was assigned."""
    if validation is None:
        return
    strength = validation.strength
    st.caption(
        f"{_STRENGTH_ICON[strength]}  **{validation.strength_label}** · "
        f"{STRENGTH_EXPLANATION[strength]} "
        f"({validation.checks_run} checks run)"
    )


def render_validation(validation: ValidationResult | None) -> None:
    """The validation findings, when there are any worth showing."""
    if validation is None:
        return
    shown = [f for f in validation.findings if f.severity is not Severity.NOTE]
    if not shown and validation.valid:
        return

    label = (
        f"Validation: {len(shown)} finding(s)" if validation.valid
        else "Validation failed"
    )
    with st.expander(label, expanded=not validation.valid):
        for finding in validation.findings:
            marker = {
                Severity.ERROR: "✕", Severity.WARNING: "!", Severity.NOTE: "·",
            }[finding.severity]
            st.markdown(f"{marker}  {finding.message}")
        st.caption(
            f"{validation.checks_run} deterministic checks ran. "
            f"Confidence {validation.confidence:.0%}."
        )


def render_record(record: dict[str, Any], *, title: str = "Evidence") -> None:
    """A labelled evidence record as a two-column table."""
    if not record:
        return

    with st.expander(title, expanded=False):
        simple = {k: v for k, v in record.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in record.items() if isinstance(v, dict)}

        if simple:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Field": list(simple),
                        "Value": [format_value(v) for v in simple.values()],
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        for label, values in nested.items():
            if not values:
                continue
            st.markdown(f"**{label}**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Name": list(values),
                        "Value": [format_value(v) for v in values.values()],
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "Every figure above was computed by Python from the loaded "
            "dataset. This is a record of what ran, not model reasoning."
        )


def render_cautions(warnings: list[str], *, label: str = "ℹ") -> None:
    """Data-sufficiency and adjustment notes, as captions rather than alarms."""
    for warning in warnings:
        st.caption(f"{label} {warning}")


__all__ = [
    "STRENGTH_EXPLANATION",
    "format_value",
    "render_cautions",
    "render_record",
    "render_strength",
    "render_validation",
]
