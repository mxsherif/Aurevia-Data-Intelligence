"""The deterministic investigation engine.

Runs a bounded sequence against the dataframe: confirm the premise, quantify
the change, decompose it across a few dimensions, inspect the timing, rank what
moved. No LLM is involved — the plan arrives already validated, and everything
here is pandas.

Two things in this module deserve care.

**Premise verification.** "Why did revenue decline last quarter?" asserts a
decline. If revenue rose, answering the question as asked would be fabrication.
So the direction is computed first and checked against the claim, and a
mismatch stops the investigation with the real figures instead.

**Contribution arithmetic.** A category's share of a *net* change is the
intuitive reading and the easy one to get wrong: when some categories rise
while others fall, shares of the net blow up past 100% and read as nonsense.
See :func:`_build_breakdown` for exactly how that is handled.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from app.models.investigation import (
    DIRECTION_CLAIMS,
    DimensionBreakdown,
    Direction,
    InvestigationFinding,
    InvestigationPlan,
    InvestigationResult,
    TemporalFinding,
)
from app.services.plan_validator import ID_LIKE_MIN_DISTINCT, ID_LIKE_RATIO
from app.tools.aggregations import resolve_aggregation
from app.tools.exceptions import ToolError
from app.tools.timeseries import calculate_time_trend, resolve_frequency
from app.tools.validation import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
    require_dataframe,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Bounds -- an investigation must not run away
# --------------------------------------------------------------------------- #

#: Dimensions decomposed per investigation.
MAX_DIMENSIONS = 5
MIN_DIMENSIONS = 1
#: Categories reported per dimension.
MAX_CATEGORIES = 8
#: Distinct values above which a column is too granular to decompose.
MAX_CARDINALITY = 30
#: Sub-periods examined inside the compared window.
MAX_TEMPORAL_PERIODS = 12

#: A change this small, relative to the metric's own scale, is "no change".
FLAT_THRESHOLD_PCT = 0.5
#: When the net change is this small a share of gross movement, the categories
#: are offsetting each other and the net-share reading is suppressed.
OFFSETTING_RATIO = 0.25

#: Row counts below which a finding is flagged as thin.
MIN_CATEGORY_ROWS = 15
MIN_PERIOD_ROWS = 30

#: Aggregations whose per-category changes sum to the overall change. Only
#: these support a contribution decomposition.
ADDITIVE_AGGREGATIONS = frozenset({"sum", "count"})


# --------------------------------------------------------------------------- #
# Dimension discovery
# --------------------------------------------------------------------------- #

def discover_dimensions(
    df: pd.DataFrame,
    requested: list[str] | None = None,
    *,
    exclude: set[str] | None = None,
    limit: int = MAX_DIMENSIONS,
) -> tuple[list[str], dict[str, str]]:
    """Choose which columns are worth decomposing a change across.

    Deterministic rules run *before* the planner's preferences are honoured:
    a column has to be categorical and of usable cardinality to be decomposed
    at all. Identifiers, free text, raw timestamps and numeric measures are
    rejected with a reason, which the UI shows so the exclusion is visible
    rather than silent.

    Returns ``(chosen, skipped)`` where `skipped` maps a column to why.
    """
    require_dataframe(df)
    excluded = exclude or set()
    skipped: dict[str, str] = {}

    eligible = set(categorical_columns(df, max_unique=MAX_CARDINALITY * 10))
    numeric = set(numeric_columns(df))
    dates = set(datetime_columns(df))
    rows = max(len(df), 1)

    def assess(column: str) -> str | None:
        """Why `column` cannot be used, or None when it can."""
        if column not in df.columns:
            return "not a column in this dataset"
        if column in excluded:
            return "already used as the metric or date field"
        if column in dates:
            return "a date column, inspected as timing instead"
        if column in numeric:
            return "a numeric measure, not a category"
        if column not in eligible:
            return "not a categorical column"
        distinct = int(df[column].nunique(dropna=True))
        if distinct < 2:
            return "has only one value, so it cannot explain a difference"
        if distinct > MAX_CARDINALITY:
            return f"has {distinct:,} distinct values, too granular to decompose"
        # The ratio alone misfires on a small frame -- 3 regions in 6 rows is
        # not an identifier -- so a floor applies first. Same rule, and the
        # same constants, as the Phase 3 plan validator.
        if distinct >= max(ID_LIKE_MIN_DISTINCT, ID_LIKE_RATIO * rows):
            return "nearly unique per row, so grouping explains nothing"
        return None

    chosen: list[str] = []

    # The planner's suggestions first, in its order, each still screened.
    for column in requested or []:
        name = str(column)
        if name in chosen:
            continue
        reason = assess(name)
        if reason:
            skipped[name] = reason
            continue
        chosen.append(name)
        if len(chosen) >= limit:
            return chosen, skipped

    # Then fill from the dataset itself, coarsest first: a 3-value column
    # yields a clearer decomposition than a 25-value one.
    remaining = [
        column
        for column in (str(c) for c in df.columns)
        if column not in chosen and column not in skipped and assess(column) is None
    ]
    remaining.sort(key=lambda c: int(df[c].nunique(dropna=True)))
    for column in remaining:
        chosen.append(column)
        if len(chosen) >= limit:
            break

    return chosen, skipped


# --------------------------------------------------------------------------- #
# Premise
# --------------------------------------------------------------------------- #

def claimed_direction(question: str) -> Direction | None:
    """The direction of travel a question asserts, if it asserts one.

    Deterministic keyword matching rather than an LLM call: the planner also
    reports a claim, and these two agreeing is worth more than either alone.
    """
    words = set(
        word.strip(".,!?;:'\"()")
        for word in (question or "").lower().split()
    )
    for direction, markers in DIRECTION_CLAIMS.items():
        if words & set(markers):
            return direction
    return None


def _direction_of(change: float, reference: float) -> Direction:
    """Which way a change went, with a dead band for noise."""
    if reference:
        if abs(change) / abs(reference) * 100 < FLAT_THRESHOLD_PCT:
            return Direction.FLAT
    elif change == 0:
        return Direction.FLAT
    return Direction.UP if change > 0 else Direction.DOWN


# --------------------------------------------------------------------------- #
# Period slicing
# --------------------------------------------------------------------------- #

def _period_series(
    df: pd.DataFrame, plan: InvestigationPlan
) -> tuple[pd.Series, pd.DataFrame]:
    """A period label per row, and the rows that have one."""
    frequency = resolve_frequency(plan.granularity)
    alias = {
        "daily": "D", "weekly": "W", "monthly": "M",
        "quarterly": "Q", "yearly": "Y",
    }[frequency.name]

    parsed = pd.to_datetime(df[plan.time_column], errors="coerce", format="mixed")
    usable = df.loc[parsed.notna()].copy()
    periods = parsed.loc[parsed.notna()].dt.to_period(alias)
    return periods, usable


def _aggregate(values: pd.Series, aggregation: str) -> float:
    """Reduce a slice to one number, with NaN for nothing."""
    aggregator = resolve_aggregation(aggregation)
    if values.empty:
        return float("nan")
    try:
        return float(aggregator.apply(values))
    except (TypeError, ValueError):
        return float("nan")


def _safe_pct(change: float, base: float) -> float | None:
    if not base or not math.isfinite(base) or not math.isfinite(change):
        return None
    return round(change / abs(base) * 100, 2)


# --------------------------------------------------------------------------- #
# Contribution
# --------------------------------------------------------------------------- #

def _build_breakdown(
    dimension: str,
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    plan: InvestigationPlan,
    overall_change: float,
) -> DimensionBreakdown:
    """Decompose the overall change across one dimension's categories.

    Each category gets its baseline value, comparison value and absolute
    change. Two shares are then computed, because one number cannot honestly
    carry both meanings:

    ``contribution``
        ``category_change / overall_change``, as a percentage — the share of
        the *net* change. Intuitive when categories move together, and
        actively misleading when they do not: if one category falls 500 while
        another rises 400, the net is -100 and the faller "contributed 500%".
        So it is computed only when the net change is a meaningful share of
        total movement (see :data:`OFFSETTING_RATIO`); otherwise it is ``None``
        and the breakdown is marked ``offsetting``.

    ``share_of_movement``
        ``|category_change| / sum(|category_change|)``, as a percentage — the
        share of *gross* movement. Always defined, always 0-100, and the
        answer to "where did the movement happen" regardless of offsetting.

    Ranking is by absolute change, so it reflects what actually moved.
    """
    breakdown = DimensionBreakdown(dimension=dimension)
    metric, aggregation = plan.metric, plan.aggregation

    categories = sorted(
        set(baseline[dimension].dropna().astype(str))
        | set(comparison[dimension].dropna().astype(str))
    )
    if not categories:
        breakdown.notes.append("No categories with data in both periods.")
        return breakdown

    breakdown.categories_examined = len(categories)
    rows: list[dict[str, Any]] = []

    for category in categories:
        base_rows = baseline[baseline[dimension].astype(str) == category]
        comp_rows = comparison[comparison[dimension].astype(str) == category]

        base_value = _aggregate(base_rows[metric], aggregation)
        comp_value = _aggregate(comp_rows[metric], aggregation)
        # A category absent from one period contributes its whole value.
        base_value = 0.0 if math.isnan(base_value) else base_value
        comp_value = 0.0 if math.isnan(comp_value) else comp_value

        rows.append(
            {
                "category": category,
                "baseline": base_value,
                "comparison": comp_value,
                "change": comp_value - base_value,
                "baseline_rows": int(len(base_rows)),
                "comparison_rows": int(len(comp_rows)),
            }
        )

    gross = sum(abs(row["change"]) for row in rows)
    breakdown.gross_movement = gross

    additive = aggregation in ADDITIVE_AGGREGATIONS
    net_is_meaningful = (
        additive
        and gross > 0
        and abs(overall_change) >= OFFSETTING_RATIO * gross
    )
    breakdown.offsetting = additive and gross > 0 and not net_is_meaningful

    if not additive:
        breakdown.notes.append(
            f"`{aggregation}` does not decompose additively, so each category "
            "shows its own change rather than a share of the total."
        )
    elif breakdown.offsetting:
        breakdown.notes.append(
            "Increases and decreases largely cancel out across these "
            "categories, so a share of the net change would be misleading. "
            "Shares of total movement are shown instead."
        )

    rows.sort(key=lambda row: abs(row["change"]), reverse=True)

    for rank, row in enumerate(rows[:MAX_CATEGORIES], start=1):
        change = row["change"]
        breakdown.findings.append(
            InvestigationFinding(
                dimension=dimension,
                category=row["category"],
                baseline_value=round(row["baseline"], 4),
                comparison_value=round(row["comparison"], 4),
                absolute_change=round(change, 4),
                percentage_change=_safe_pct(change, row["baseline"]),
                contribution=(
                    # `+ 0.0` normalises -0.0, which renders as "-0.0%".
                    round(change / overall_change * 100, 2) + 0.0
                    if net_is_meaningful and overall_change
                    else None
                ),
                share_of_movement=round(abs(change) / gross * 100, 2) if gross else 0.0,
                rank=rank,
                direction=_direction_of(change, row["baseline"]),
                baseline_rows=row["baseline_rows"],
                comparison_rows=row["comparison_rows"],
            )
        )

    if len(rows) > MAX_CATEGORIES:
        breakdown.notes.append(
            f"Showing the {MAX_CATEGORIES} largest movers of {len(rows)} "
            f"{dimension} values."
        )
    return breakdown


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

def run_investigation(
    df: pd.DataFrame,
    plan: InvestigationPlan,
    *,
    question: str = "",
    max_dimensions: int = MAX_DIMENSIONS,
) -> InvestigationResult:
    """Execute a validated investigation plan. Never raises.

    The plan is assumed already checked against the dataset by
    :func:`app.services.investigation_validator.validate_investigation_plan`.
    """
    require_dataframe(df)
    result = InvestigationResult(
        metric=plan.metric,
        aggregation=plan.aggregation,
        time_column=plan.time_column,
        granularity=plan.granularity,
    )

    # -- Step 1: slice into periods ---------------------------------------- #
    try:
        periods, usable = _period_series(df, plan)
    except ToolError as exc:
        return InvestigationResult.failure(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not read the date column")
        return InvestigationResult.failure(
            f"The date field `{plan.time_column}` could not be read as dates "
            f"({type(exc).__name__})."
        )

    excluded_rows = int(len(df) - len(usable))
    distinct = periods.sort_values().unique()
    if len(distinct) < 2:
        return InvestigationResult.failure(
            f"The data covers only {len(distinct)} {plan.granularity} period, "
            "so there is nothing to compare against. Try a finer time grouping."
        )

    comparison_period = distinct[-1]
    baseline_period = distinct[-2]
    result.baseline_label = str(baseline_period)
    result.comparison_label = str(comparison_period)

    baseline = usable.loc[(periods == baseline_period).to_numpy()]
    comparison = usable.loc[(periods == comparison_period).to_numpy()]
    result.steps_completed.append(
        f"Compared {result.comparison_label} with {result.baseline_label}"
    )

    # -- Step 2: quantify the overall change ------------------------------- #
    baseline_value = _aggregate(baseline[plan.metric], plan.aggregation)
    comparison_value = _aggregate(comparison[plan.metric], plan.aggregation)
    if math.isnan(baseline_value) or math.isnan(comparison_value):
        return InvestigationResult.failure(
            f"`{plan.metric}` has no usable values in one of the two periods."
        )

    result.baseline_value = round(baseline_value, 4)
    result.comparison_value = round(comparison_value, 4)
    result.absolute_change = round(comparison_value - baseline_value, 4)
    result.percentage_change = _safe_pct(result.absolute_change, baseline_value)
    result.direction = _direction_of(result.absolute_change, baseline_value)
    result.steps_completed.append("Quantified the overall change")

    # -- Step 3: verify the premise ---------------------------------------- #
    claim = claimed_direction(question) or _plan_claim(plan)
    result.claimed_direction = claim
    if claim is not None and claim is not result.direction:
        result.premise_confirmed = False
        result.premise_message = _premise_message(result, claim)
        result.steps_completed.append("Checked the premise — it does not hold")
        result.evidence_summary = _evidence(
            result, baseline, comparison, excluded_rows, len(distinct)
        )
        result.warnings.extend(
            _sufficiency_warnings(result, baseline, comparison, excluded_rows)
        )
        logger.info(
            "Premise rejected: claimed %s, actual %s", claim, result.direction
        )
        return result
    result.steps_completed.append("Confirmed the premise")

    # -- Step 4: decompose across dimensions -------------------------------- #
    dimensions, skipped = discover_dimensions(
        df,
        plan.candidate_dimensions,
        exclude={plan.metric, plan.time_column},
        limit=max(MIN_DIMENSIONS, min(max_dimensions, MAX_DIMENSIONS)),
    )
    result.skipped_dimensions = skipped

    for dimension in dimensions:
        try:
            breakdown = _build_breakdown(
                dimension, baseline, comparison, plan, result.absolute_change
            )
        except Exception:  # noqa: BLE001 - one bad column must not stop the rest
            logger.exception("Could not decompose by %s", dimension)
            result.skipped_dimensions[dimension] = "could not be decomposed"
            continue
        if breakdown.findings:
            result.breakdowns.append(breakdown)
            result.steps_completed.append(f"Measured contribution by {dimension}")

    if not result.breakdowns:
        result.warnings.append(
            "No categorical column in this dataset could be used to break the "
            "change down, so only the overall movement is available."
        )

    # -- Step 5: timing ---------------------------------------------------- #
    window = _window_bounds(baseline, comparison, plan.time_column)
    result.temporal_findings = _temporal(df, plan, window)
    if result.temporal_findings:
        result.steps_completed.append("Inspected the timing within the period")

    # -- Step 6: evidence and cautions -------------------------------------- #
    result.evidence_summary = _evidence(
        result, baseline, comparison, excluded_rows, len(distinct)
    )
    result.warnings.extend(
        _sufficiency_warnings(result, baseline, comparison, excluded_rows)
    )
    result.steps_completed.append("Ranked the evidence")
    return result


def _plan_claim(plan: InvestigationPlan) -> Direction | None:
    """The direction the planner read in the question."""
    claimed = (plan.claimed_direction or "").strip().lower()
    if claimed in ("decrease", "decline", "down"):
        return Direction.DOWN
    if claimed in ("increase", "growth", "up", "rise"):
        return Direction.UP
    return None


def _premise_message(result: InvestigationResult, claim: Direction) -> str:
    """The sentence shown when the data contradicts the question."""
    metric = result.metric
    actual = result.direction
    change = result.percentage_change

    if actual is Direction.FLAT:
        movement = (
            f"it was essentially unchanged ({change:+.2f}%)"
            if change is not None
            else "it was essentially unchanged"
        )
    else:
        verb = "increased" if actual is Direction.UP else "decreased"
        movement = (
            f"it {verb} by {abs(change):.2f}%" if change is not None
            else f"it {verb}"
        )

    claimed_word = "decline" if claim is Direction.DOWN else "increase"
    offer = (
        "I can investigate what contributed to the increase instead."
        if actual is Direction.UP
        else "I can investigate what contributed to the decrease instead."
        if actual is Direction.DOWN
        else "There is no material change to investigate in this period."
    )
    return (
        f"`{metric}` did not {claimed_word} in {result.comparison_label}: "
        f"{movement} compared with {result.baseline_label}. {offer}"
    )


def _temporal(
    df: pd.DataFrame,
    plan: InvestigationPlan,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> list[TemporalFinding]:
    """A finer-grained series, so "when did it happen" has an answer.

    Scoped to `window` -- the two periods being compared -- because timing is
    a question about *this* movement. A twelve-month tail would invite the
    write-up to cite a month outside the comparison, which reads as an answer
    to a different question.
    """
    finer = {
        "yearly": "quarterly",
        "quarterly": "monthly",
        "monthly": "weekly",
        "weekly": "daily",
        "daily": "daily",
    }[resolve_frequency(plan.granularity).name]

    try:
        trend = calculate_time_trend(
            df, plan.time_column, plan.metric,
            aggregation=plan.aggregation, frequency=finer,
        )
    except ToolError as exc:
        logger.info("Temporal breakdown unavailable: %s", exc)
        return []

    frame = trend.dataframe
    if window is not None and not frame.empty:
        start, end = window
        # One period before the window, so the first in-window period has
        # something to be measured against.
        inside = frame["period"].between(start, end)
        if inside.any():
            first = int(inside.to_numpy().argmax())
            frame = frame.iloc[max(first - 1, 0):]
            frame = frame[frame["period"] <= end]
    frame = frame.tail(MAX_TEMPORAL_PERIODS)

    findings: list[TemporalFinding] = []
    for _, row in frame.iterrows():
        change = row.get("pct_change")
        value = float(row["value"])
        findings.append(
            TemporalFinding(
                period=str(row["period_label"]),
                value=round(value, 4),
                percentage_change=(
                    round(float(change), 2)
                    if change is not None and not pd.isna(change) else None
                ),
                rows=int(row.get("row_count", 0)),
            )
        )

    # Absolute period-over-period movement, which "when was it worst" needs.
    for previous, current in zip(findings, findings[1:]):
        current.change_vs_previous = round(current.value - previous.value, 4)
    return findings


def _window_bounds(
    baseline: pd.DataFrame, comparison: pd.DataFrame, time_column: str
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The date span covered by the two compared periods."""
    stamps = []
    for frame in (baseline, comparison):
        if time_column not in frame.columns or frame.empty:
            continue
        parsed = pd.to_datetime(
            frame[time_column], errors="coerce", format="mixed"
        ).dropna()
        if not parsed.empty:
            stamps.extend([parsed.min(), parsed.max()])
    if not stamps:
        return None
    return min(stamps), max(stamps)


def _evidence(
    result: InvestigationResult,
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    excluded_rows: int,
    periods_available: int,
) -> dict[str, Any]:
    """The provenance record shown in the Evidence panel."""
    aggregator = resolve_aggregation(result.aggregation)
    return {
        "metric": result.metric,
        "primary_computation": (
            f"{aggregator.label.upper()}({result.metric})"
            if result.aggregation != "count"
            else "COUNT(rows)"
        ),
        "time_column": result.time_column,
        "granularity": result.granularity,
        "periods_compared": f"{result.baseline_label} vs {result.comparison_label}",
        "periods_available": periods_available,
        "baseline_rows": int(len(baseline)),
        "comparison_rows": int(len(comparison)),
        "rows_included": int(len(baseline) + len(comparison)),
        "rows_excluded_no_date": excluded_rows,
        "dimensions_inspected": result.dimensions_inspected,
        "dimensions_skipped": dict(result.skipped_dimensions),
        "contribution_method": (
            "share of net change"
            if any(
                f.contribution is not None for f in result.all_findings
            )
            else "share of total movement"
        ),
    }


def _sufficiency_warnings(
    result: InvestigationResult,
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    excluded_rows: int,
) -> list[str]:
    """Deterministic cautions about how much the evidence can bear."""
    warnings: list[str] = []
    base_rows, comp_rows = len(baseline), len(comparison)

    if min(base_rows, comp_rows) < MIN_PERIOD_ROWS:
        warnings.append(
            f"One period has only {min(base_rows, comp_rows)} record(s) "
            f"({result.baseline_label}: {base_rows:,}, "
            f"{result.comparison_label}: {comp_rows:,}), so this comparison "
            "should be interpreted cautiously."
        )
    elif base_rows and comp_rows:
        ratio = max(base_rows, comp_rows) / min(base_rows, comp_rows)
        if ratio >= 2:
            warnings.append(
                f"The two periods cover very different volumes "
                f"({base_rows:,} vs {comp_rows:,} records), so totals are not "
                "directly comparable."
            )

    if excluded_rows:
        share = excluded_rows / max(len(baseline) + len(comparison) + excluded_rows, 1)
        if share >= 0.05:
            warnings.append(
                f"{excluded_rows:,} row(s) have no usable date and were "
                "excluded from every period."
            )

    missing = 0
    for frame in (baseline, comparison):
        if result.metric in frame.columns:
            missing += int(frame[result.metric].isna().sum())
    if missing:
        total = base_rows + comp_rows
        if total and missing / total >= 0.1:
            warnings.append(
                f"`{result.metric}` is missing in {missing:,} of {total:,} rows "
                "across the two periods."
            )

    thin = [
        f for f in result.ranked_findings(limit=3)
        if f.total_rows < MIN_CATEGORY_ROWS
    ]
    for finding in thin:
        warnings.append(
            f"The {finding.dimension} finding for \"{finding.category}\" is "
            f"based on only {finding.total_rows} record(s) and should be "
            "interpreted cautiously."
        )

    if any(b.offsetting for b in result.breakdowns):
        warnings.append(
            "In at least one dimension, gains and losses largely offset each "
            "other; shares of total movement are shown rather than shares of "
            "the net change."
        )
    return warnings


__all__ = [
    "ADDITIVE_AGGREGATIONS",
    "MAX_CARDINALITY",
    "MAX_CATEGORIES",
    "MAX_DIMENSIONS",
    "OFFSETTING_RATIO",
    "claimed_direction",
    "discover_dimensions",
    "run_investigation",
]
