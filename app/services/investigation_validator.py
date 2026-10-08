"""Validation of an investigation plan against the dataset.

The same discipline as :mod:`app.services.plan_validator`, applied to the
investigation planner: resolve every column it named, repair what is safe,
refuse what the dataset cannot support, and never silently investigate the
wrong field.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from app.models.investigation import InvestigationPlan
from app.services.investigation_engine import MAX_DIMENSIONS, discover_dimensions
from app.tools.aggregations import resolve_aggregation, supported_aggregations
from app.tools.exceptions import ToolError
from app.tools.resolution import resolve_column
from app.tools.timeseries import supported_frequencies
from app.tools.validation import (
    datetime_columns,
    numeric_columns,
    require_dataframe,
)

logger = logging.getLogger(__name__)

#: Aggregations an investigation can decompose meaningfully.
INVESTIGABLE_AGGREGATIONS = ("sum", "count", "mean")

#: Granularities offered, coarsest last; a dataset with few periods gets a
#: finer one so there is something to compare.
GRANULARITY_ORDER = ("daily", "weekly", "monthly", "quarterly", "yearly")


@dataclass
class InvestigationValidation:
    """The outcome of checking an investigation plan."""

    ok: bool
    plan: InvestigationPlan | None = None
    message: str | None = None
    clarification_question: str | None = None
    adjustments: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.clarification_question is not None


def validate_investigation_plan(
    df: pd.DataFrame, plan: InvestigationPlan
) -> InvestigationValidation:
    """Check and repair an investigation plan. Never raises."""
    require_dataframe(df)

    if df.empty:
        return InvestigationValidation(
            ok=False, message="The loaded dataset has no rows to investigate."
        )

    if plan.requires_clarification and plan.clarification_question:
        return InvestigationValidation(
            ok=False, plan=plan,
            clarification_question=plan.clarification_question,
        )

    working = plan.model_copy(deep=True)
    adjustments: list[str] = []
    columns = [str(c) for c in df.columns]
    numeric = numeric_columns(df)
    dates = datetime_columns(df)

    # -- metric ------------------------------------------------------------ #
    if not numeric:
        return InvestigationValidation(
            ok=False,
            message=(
                "Aurevia understood the request, but this dataset has no "
                "numeric column whose change could be investigated."
            ),
        )

    match = resolve_column(working.metric, columns, allowed=numeric)
    if match.is_trusted:
        if match.column != working.metric:
            adjustments.append(
                f"Read the measure '{working.metric}' as `{match.column}`."
            )
        working.metric = match.column
    else:
        anywhere = resolve_column(working.metric, columns)
        if anywhere.is_trusted and anywhere.column not in numeric:
            return InvestigationValidation(
                ok=False,
                message=(
                    f"Aurevia understood the request, but `{anywhere.column}` "
                    "is not a numeric column, so its change cannot be "
                    "measured. Try "
                    + ", ".join(f"`{c}`" for c in numeric[:3])
                    + "."
                ),
            )
        return InvestigationValidation(
            ok=False,
            clarification_question=(
                f"I could not find a measure matching '{working.metric}'. "
                "Which should I investigate? Available: "
                + ", ".join(numeric[:5]) + "."
            ),
        )

    # -- date column ------------------------------------------------------- #
    if not dates:
        return InvestigationValidation(
            ok=False,
            message=(
                "Aurevia understood the request, but this dataset does not "
                "contain a valid date field, which an investigation needs to "
                "compare two periods."
            ),
        )

    time_match = resolve_column(working.time_column, columns, allowed=dates)
    if time_match.is_trusted:
        if time_match.column != working.time_column:
            adjustments.append(
                f"Used `{time_match.column}` as the date field."
            )
        working.time_column = time_match.column
    else:
        working.time_column = dates[0]
        adjustments.append(f"Used `{working.time_column}` as the date field.")

    # -- aggregation ------------------------------------------------------- #
    try:
        aggregator = resolve_aggregation(working.aggregation)
    except ToolError:
        adjustments.append(
            f"'{working.aggregation}' is not a supported aggregation; used sum. "
            "Supported: " + ", ".join(supported_aggregations()) + "."
        )
        working.aggregation = "sum"
    else:
        if aggregator.name not in INVESTIGABLE_AGGREGATIONS:
            adjustments.append(
                f"{aggregator.name} cannot be decomposed into contributions; "
                "used sum instead."
            )
            working.aggregation = "sum"
        else:
            working.aggregation = aggregator.name

    # -- granularity ------------------------------------------------------- #
    granularity = (working.granularity or "").strip().lower()
    if granularity not in supported_frequencies():
        if granularity:
            adjustments.append(
                f"'{working.granularity}' is not a supported time grouping; "
                "used quarterly."
            )
        granularity = "quarterly"

    granularity, note = _ensure_two_periods(df, working.time_column, granularity)
    if granularity is None:
        return InvestigationValidation(
            ok=False,
            message=(
                "Aurevia understood the request, but this dataset covers only "
                "one period at every available time grouping, so there is "
                "nothing to compare against."
            ),
        )
    if note:
        adjustments.append(note)
    working.granularity = granularity

    # -- dimensions -------------------------------------------------------- #
    resolved: list[str] = []
    for candidate in working.candidate_dimensions:
        dimension_match = resolve_column(str(candidate), columns)
        if dimension_match.is_trusted and dimension_match.column not in resolved:
            resolved.append(dimension_match.column)

    chosen, skipped = discover_dimensions(
        df, resolved,
        exclude={working.metric, working.time_column},
        limit=MAX_DIMENSIONS,
    )
    if not chosen:
        adjustments.append(
            "No categorical column in this dataset can be used to break the "
            "change down; only the overall movement will be reported."
        )
    working.candidate_dimensions = chosen

    for column, reason in skipped.items():
        adjustments.append(f"Skipped `{column}`: {reason}.")

    return InvestigationValidation(ok=True, plan=working, adjustments=adjustments)


def _ensure_two_periods(
    df: pd.DataFrame, time_column: str, granularity: str
) -> tuple[str | None, str | None]:
    """Pick a granularity the data actually spans at least twice.

    A quarterly comparison on three weeks of data has nothing to compare, so
    step to a finer grouping rather than failing.
    """
    alias = {
        "daily": "D", "weekly": "W", "monthly": "M",
        "quarterly": "Q", "yearly": "Y",
    }

    try:
        parsed = pd.to_datetime(df[time_column], errors="coerce", format="mixed")
    except Exception:  # noqa: BLE001
        return granularity, None

    parsed = parsed.dropna()
    if parsed.empty:
        return granularity, None

    def period_count(name: str) -> int:
        try:
            return int(parsed.dt.to_period(alias[name]).nunique())
        except Exception:  # noqa: BLE001
            return 0

    if period_count(granularity) >= 2:
        return granularity, None

    # Try finer groupings, in order.
    start = GRANULARITY_ORDER.index(granularity)
    for finer in reversed(GRANULARITY_ORDER[:start]):
        if period_count(finer) >= 2:
            return finer, (
                f"The data covers only one {granularity} period, so the "
                f"comparison was made {finer} instead."
            )
    return None, None


__all__ = [
    "GRANULARITY_ORDER",
    "INVESTIGABLE_AGGREGATIONS",
    "InvestigationValidation",
    "validate_investigation_plan",
]
