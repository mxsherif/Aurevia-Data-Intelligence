"""Validation of a completed analysis.

Runs after the executor and before the answer is shown. Compares the *plan*
against the *result it produced* and asks whether the question was actually
answered: the right columns, the grouping that was asked for, the filters that
were requested, a usable shape, finite numbers, a chart that matches, and prose
that quotes only computed figures.

Everything here is deterministic except one optional step. The semantic check
(does the prose address the question?) is the single judgement code cannot
make, and it only runs when every cheap check has already passed — so a
malformed result costs no extra API call.

Each finding names the stage that could fix it, which is what makes the retry
in :mod:`app.services.ask_pipeline` targeted rather than a blind re-run.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import pandas as pd

from app.models.plans import AnalysisPlan, Intent, SortDirection
from app.models.results import AnalysisResult, Insight
from app.models.validation import (
    EvidenceStrength,
    RetryStage,
    SemanticCheck,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from app.services.grounding import check_texts
from app.services.llm_service import LLMError, LLMService

logger = logging.getLogger(__name__)

#: Below this many analysed rows, a result is thin enough to caution about.
MIN_ROWS_FOR_CONFIDENCE = 30
#: Below this, say so plainly.
MIN_ROWS_SUFFICIENT = 10
#: Result shapes that legitimately carry a single row.
SINGLE_ROW_SHAPES = frozenset({"scalar", "distribution", "outliers", "none"})

#: Confidence lost per finding severity.
_PENALTY = {
    Severity.ERROR: 0.45,
    Severity.WARNING: 0.12,
    Severity.NOTE: 0.03,
}

SEMANTIC_PROMPT = """\
You are a validator inside Aurevia, a data-analysis platform. You are given a \
user's question, the figures Python computed, and the explanation that was \
written about them.

Answer only two things:
1. Does the explanation address the question that was asked?
2. Does any statement in the explanation disagree with the supplied figures?

Judge only what is in front of you. An explanation that is correct but terse \
still addresses the question. An explanation that answers a different question, \
or states the opposite of a supplied figure, does not.

Do not rewrite the explanation. Do not comment on style. Do not perform \
analysis of your own.
"""


class _Checker:
    """Accumulates findings for one validation pass."""

    def __init__(self) -> None:
        self.findings: list[ValidationIssue] = []
        self.checks = 0

    def check(self) -> None:
        self.checks += 1

    def fail(
        self,
        code: str,
        message: str,
        stage: RetryStage = RetryStage.NONE,
        **details: Any,
    ) -> None:
        self.findings.append(
            ValidationIssue(
                code=code, severity=Severity.ERROR, message=message,
                stage=stage, details=details,
            )
        )

    def warn(
        self,
        code: str,
        message: str,
        stage: RetryStage = RetryStage.NONE,
        **details: Any,
    ) -> None:
        self.findings.append(
            ValidationIssue(
                code=code, severity=Severity.WARNING, message=message,
                stage=stage, details=details,
            )
        )

    def note(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(
            ValidationIssue(
                code=code, severity=Severity.NOTE, message=message,
                stage=RetryStage.NONE, details=details,
            )
        )


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #

def _check_execution(checker: _Checker, result: AnalysisResult) -> bool:
    """Did the analysis run at all? Returns False to stop further checks."""
    checker.check()
    if not result.success:
        checker.fail(
            "execution_failed",
            result.error or "The analysis did not complete.",
            RetryStage.EXECUTION,
        )
        return False
    return True


def _check_columns_exist(
    checker: _Checker, df: pd.DataFrame, plan: AnalysisPlan
) -> None:
    """Every column the plan named must be in the dataframe."""
    checker.check()
    columns = {str(c) for c in df.columns}
    referenced = [
        ("metric", plan.metric),
        ("date field", plan.time_column),
        *[("grouping field", d) for d in plan.dimensions],
        *[("filter field", f.column) for f in plan.filters],
    ]
    for role, name in referenced:
        if name and name not in columns:
            checker.fail(
                "column_missing",
                f"The {role} `{name}` is not a column in this dataset.",
                RetryStage.PLANNING,
                column=name, role=role,
            )


def _check_not_empty(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """A result with no figures and no rows answers nothing."""
    checker.check()
    shape = str(result.metadata.get("result_shape", "none"))

    if not result.summary_data and not result.table_data:
        checker.fail(
            "empty_result",
            "The analysis produced no figures.",
            RetryStage.PLANNING,
        )
        return

    if result.table_data is not None and not result.table_data:
        if shape not in SINGLE_ROW_SHAPES:
            checker.warn(
                "empty_table",
                "The analysis produced headline figures but no result rows.",
                RetryStage.PLANNING,
            )


def _check_metric_present(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """The metric the plan asked for must appear in what was computed."""
    checker.check()
    if plan.metric is None:
        return
    if result.metadata.get("metric") == plan.metric:
        return

    haystack = " ".join(
        [*result.summary_data, *(result.table_data[0] if result.table_data else [])]
    ).lower()
    if plan.metric.lower().replace("_", " ") in haystack or plan.metric in haystack:
        return

    checker.warn(
        "metric_not_evident",
        f"The result does not obviously reflect the measure `{plan.metric}`.",
        RetryStage.PLANNING,
        metric=plan.metric,
    )


def _check_grouping_applied(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """A plan that asked to group by a field must have grouped by it."""
    checker.check()
    dimension = plan.primary_dimension
    if dimension is None:
        return
    if plan.intent in (Intent.SUMMARY, Intent.CORRELATION, Intent.DATASET_QUESTION):
        return  # these legitimately ignore a dimension

    used = result.metadata.get("dimension")
    if used == dimension:
        return
    if result.table_data and dimension in result.table_data[0]:
        return

    checker.fail(
        "grouping_not_applied",
        f"The analysis was asked to break results down by `{dimension}`, "
        "but the result is not grouped by it.",
        RetryStage.PLANNING,
        requested=dimension, used=used,
    )


def _check_filters_applied(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """Every filter in the plan must have been applied, and have kept rows."""
    checker.check()
    if not plan.filters:
        return

    applied = result.metadata.get("filters")
    if not applied:
        checker.fail(
            "filters_not_applied",
            f"{len(plan.filters)} filter(s) were requested but the result "
            "shows no filtering.",
            RetryStage.EXECUTION,
            requested=[f.describe() for f in plan.filters],
        )
        return

    if len(applied) != len(plan.filters):
        checker.warn(
            "filters_partially_applied",
            f"{len(plan.filters)} filter(s) were requested but "
            f"{len(applied)} were applied.",
            RetryStage.PLANNING,
        )

    before = result.metadata.get("rows_before_filter")
    after = result.metadata.get("rows_after_filter")
    if isinstance(before, int) and isinstance(after, int) and before and not after:
        checker.fail(
            "filter_removed_everything",
            "The filters removed every row, so there was nothing to analyse.",
            RetryStage.PLANNING,
        )


def _check_shape(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """The result's shape must match the kind of analysis that was asked for."""
    checker.check()
    shape = str(result.metadata.get("result_shape", "none"))
    expected: dict[Intent, set[str]] = {
        Intent.RANKING: {"ranked"},
        Intent.COMPARISON: {"ranked"},
        Intent.SEGMENTATION: {"ranked"},
        Intent.COUNT: {"ranked", "scalar"},
        Intent.TREND: {"time_series"},
        Intent.TIME_COMPARISON: {"time_series"},
        Intent.PERCENTAGE_CHANGE: {"time_series", "ranked"},
        Intent.CORRELATION: {"correlation_matrix"},
        Intent.DISTRIBUTION: {"distribution", "distribution_by_group"},
        Intent.ANOMALY: {"outliers", "distribution"},
        Intent.SUMMARY: {"distribution", "none"},
        Intent.DATASET_QUESTION: {"schema"},
    }
    allowed = expected.get(plan.intent)
    if allowed and shape not in allowed:
        checker.warn(
            "shape_mismatch",
            f"A {plan.intent_label.lower()} was planned but the result has a "
            f"'{shape}' shape.",
            RetryStage.PLANNING,
            intent=str(plan.intent), shape=shape,
        )


def _check_ranking_direction(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """A descending ranking must actually be in descending order."""
    checker.check()
    if plan.intent not in (Intent.RANKING, Intent.PERCENTAGE_CHANGE):
        return
    if not result.table_data or len(result.table_data) < 2:
        return

    rows = result.table_data
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and key.lower() not in {"rank", "rows", "row_count"}
    ]
    if not numeric_keys:
        return

    key = numeric_keys[0]
    values = [
        row.get(key) for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]
    if len(values) < 2:
        return

    ascending = plan.sort_direction is SortDirection.ASCENDING
    ordered = values == sorted(values, reverse=not ascending)
    if not ordered:
        checker.warn(
            "ranking_order_wrong",
            f"The results were meant to be sorted "
            f"{'ascending' if ascending else 'descending'} by {key}, but they "
            "are not in that order.",
            RetryStage.EXECUTION,
            column=key,
        )


def _check_time_window(
    checker: _Checker, result: AnalysisResult, plan: AnalysisPlan
) -> None:
    """A requested date column and granularity must be the ones used."""
    checker.check()
    if plan.time_column is None:
        return

    used_column = result.metadata.get("time_column")
    if used_column and used_column != plan.time_column:
        checker.fail(
            "wrong_time_column",
            f"The analysis was planned on `{plan.time_column}` but ran on "
            f"`{used_column}`.",
            RetryStage.PLANNING,
        )

    used_granularity = result.metadata.get("granularity")
    if (
        plan.time_granularity
        and used_granularity
        and used_granularity != plan.time_granularity
    ):
        checker.warn(
            "wrong_granularity",
            f"The analysis was planned {plan.time_granularity} but computed "
            f"{used_granularity}.",
            RetryStage.PLANNING,
        )

    if plan.periods:
        periods = result.metadata.get("periods")
        if isinstance(periods, int) and periods > plan.periods:
            checker.warn(
                "time_window_not_applied",
                f"The question scoped {plan.periods} period(s) but "
                f"{periods} were analysed.",
                RetryStage.EXECUTION,
            )


def _check_numbers_finite(checker: _Checker, result: AnalysisResult) -> None:
    """NaN and infinity in a headline figure mean the computation broke."""
    checker.check()
    broken: list[str] = []
    for label, value in result.summary_data.items():
        if isinstance(value, float) and not math.isfinite(value):
            broken.append(label)
    if broken:
        checker.fail(
            "non_finite_values",
            "Some computed figures are not valid numbers: "
            + ", ".join(broken[:4])
            + ".",
            RetryStage.EXECUTION,
            fields=broken,
        )

    if result.table_data:
        bad_cells = sum(
            1
            for row in result.table_data
            for value in row.values()
            if isinstance(value, float) and not math.isfinite(value)
        )
        if bad_cells:
            checker.warn(
                "non_finite_cells",
                f"{bad_cells} cell(s) in the result table are not valid numbers.",
                RetryStage.NONE,
            )


def _check_chart(
    checker: _Checker, df: pd.DataFrame, result: AnalysisResult
) -> None:
    """The chart spec must reference real columns and actually be drawable."""
    checker.check()
    spec = result.chart_spec
    if not spec:
        return

    columns = {str(c) for c in df.columns}
    for role in ("x", "y", "group_by"):
        name = spec.get(role)
        if name and name not in columns:
            checker.warn(
                "chart_column_missing",
                f"The chart refers to `{name}`, which is not a column in this "
                "dataset.",
                RetryStage.CHARTING,
                role=role, column=name,
            )
            return

    # Prove it renders rather than assuming it does.
    try:
        from app.tools import generate_chart_data

        data = generate_chart_data(df, **spec)
    except Exception as exc:  # noqa: BLE001 - any failure here is a chart problem
        checker.warn(
            "chart_unbuildable",
            f"The suggested chart could not be prepared ({exc}).",
            RetryStage.CHARTING,
        )
        return

    if data.is_empty:
        checker.warn(
            "chart_empty",
            "The suggested chart would have no data to show.",
            RetryStage.CHARTING,
        )


def _check_explanation(
    checker: _Checker, result: AnalysisResult, insight: Insight | None
) -> None:
    """The prose must quote only figures the analysis computed."""
    checker.check()
    if insight is None:
        return

    report = check_texts(insight.texts(), result.allowed_numbers())
    insight.ungrounded_numbers = list(report.ungrounded)

    if report.ungrounded:
        checker.fail(
            "ungrounded_numbers",
            "The explanation contains "
            f"{len(report.ungrounded)} figure(s) that were not computed: "
            + ", ".join(report.ungrounded[:4])
            + ".",
            RetryStage.INTERPRETATION,
            numbers=report.ungrounded,
        )

    checker.check()
    if not insight.answer.strip():
        checker.fail(
            "empty_explanation",
            "The explanation was blank.",
            RetryStage.INTERPRETATION,
        )


def _check_sufficiency(checker: _Checker, result: AnalysisResult) -> None:
    """Thin data is not an error, but the user should be told."""
    checker.check()
    rows = result.metadata.get("rows_analysed")
    if not isinstance(rows, int):
        return
    if rows < MIN_ROWS_SUFFICIENT:
        checker.warn(
            "very_few_rows",
            f"This finding is based on only {rows} record(s) and should be "
            "interpreted cautiously.",
            RetryStage.NONE,
            rows=rows,
        )
    elif rows < MIN_ROWS_FOR_CONFIDENCE:
        checker.note(
            "few_rows",
            f"Based on {rows} records, which is a small sample.",
            rows=rows,
        )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _score(findings: list[ValidationIssue], checks: int) -> float:
    """Confidence from the checks that ran, not from a model's opinion."""
    if not checks:
        return 0.0
    confidence = 1.0
    for finding in findings:
        confidence -= _PENALTY[finding.severity]
    return max(0.0, min(1.0, confidence))


def _strength(
    confidence: float, findings: list[ValidationIssue], result: AnalysisResult
) -> EvidenceStrength:
    """A qualitative label, from concrete facts only.

    Limited when a blocking check failed, the sample is tiny, or confidence
    has fallen below two thirds; moderate when there are warnings or the
    sample is small; strong otherwise.
    """
    if any(f.blocking for f in findings) or confidence < 0.67:
        return EvidenceStrength.LIMITED
    if any(f.code == "very_few_rows" for f in findings):
        return EvidenceStrength.LIMITED
    rows = result.metadata.get("rows_analysed")
    thin = isinstance(rows, int) and rows < MIN_ROWS_FOR_CONFIDENCE
    if thin or any(f.severity is Severity.WARNING for f in findings):
        return EvidenceStrength.MODERATE
    return EvidenceStrength.STRONG


def _worst_stage(findings: list[ValidationIssue]) -> RetryStage:
    """The stage to retry: the cheapest that could fix a blocking finding.

    Order matters. Interpretation and charting are cheap and local; planning
    costs a call and may change the answer; execution is usually not
    retryable at all.
    """
    blocking = [f for f in findings if f.blocking and f.stage is not RetryStage.NONE]
    if not blocking:
        return RetryStage.NONE
    for stage in (
        RetryStage.CHARTING,
        RetryStage.INTERPRETATION,
        RetryStage.PLANNING,
        RetryStage.EXECUTION,
    ):
        if any(f.stage is stage for f in blocking):
            return stage
    return RetryStage.NONE


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def validate_result(
    df: pd.DataFrame,
    plan: AnalysisPlan,
    result: AnalysisResult,
    insight: Insight | None = None,
) -> ValidationResult:
    """Check a completed analysis. Entirely deterministic; never raises."""
    checker = _Checker()

    if _check_execution(checker, result):
        _check_columns_exist(checker, df, plan)
        _check_not_empty(checker, result, plan)
        _check_metric_present(checker, result, plan)
        _check_grouping_applied(checker, result, plan)
        _check_filters_applied(checker, result, plan)
        _check_shape(checker, result, plan)
        _check_ranking_direction(checker, result, plan)
        _check_time_window(checker, result, plan)
        _check_numbers_finite(checker, result)
        _check_chart(checker, df, result)
        _check_sufficiency(checker, result)
        _check_explanation(checker, result, insight)

    findings = checker.findings
    confidence = _score(findings, checker.checks)
    blocking = [f for f in findings if f.blocking]
    stage = _worst_stage(findings)

    outcome = ValidationResult(
        valid=not blocking,
        confidence=confidence,
        issues=[f.message for f in blocking],
        warnings=[f.message for f in findings if f.severity is Severity.WARNING],
        retry_recommended=bool(blocking) and stage is not RetryStage.NONE,
        retry_stage=stage,
        findings=findings,
        checks_run=checker.checks,
        strength=_strength(confidence, findings, result),
    )

    if not outcome.valid:
        logger.info(
            "Validation failed (%s): %s",
            stage, [f.code for f in blocking],
        )
    return outcome


def check_semantics(
    llm: LLMService,
    question: str,
    result: AnalysisResult,
    insight: Insight,
) -> ValidationIssue | None:
    """Ask the model the one question code cannot answer.

    Returns a finding when the prose answers a different question or
    contradicts the figures, and ``None`` otherwise -- including when the call
    fails, because a validator that blocks answers on its own unavailability
    would be worse than no validator.
    """
    payload = json.dumps(
        {
            "question": question,
            "computed_figures": result.grounding_payload(max_rows=10),
            "explanation": {
                "answer": insight.answer,
                "observations": insight.observations,
            },
        },
        separators=(",", ":"),
        default=str,
    )

    try:
        response = llm.complete_structured(
            messages=[
                {"role": "system", "content": SEMANTIC_PROMPT},
                {"role": "user", "content": payload},
            ],
            schema=SemanticCheck,
            max_output_tokens=200,
        )
    except LLMError as exc:
        logger.info("Semantic validation unavailable: %s", exc)
        return None
    except Exception:  # noqa: BLE001 - never block an answer on the validator
        logger.exception("Semantic validation failed unexpectedly")
        return None

    check = response.data
    if not isinstance(check, SemanticCheck):  # pragma: no cover - defensive
        return None

    if check.contradicts_data:
        return ValidationIssue(
            code="explanation_contradicts_data",
            severity=Severity.ERROR,
            message=(
                "The explanation disagrees with the computed figures"
                + (f": {check.reason}" if check.reason else ".")
            ),
            stage=RetryStage.INTERPRETATION,
        )
    if not check.addresses_question:
        return ValidationIssue(
            code="explanation_off_topic",
            severity=Severity.ERROR,
            message=(
                "The explanation does not answer the question that was asked"
                + (f": {check.reason}" if check.reason else ".")
            ),
            stage=RetryStage.INTERPRETATION,
        )
    return None


def merge_finding(
    validation: ValidationResult, finding: ValidationIssue | None
) -> ValidationResult:
    """Fold a late finding (such as the semantic check) into a verdict."""
    if finding is None:
        return validation

    findings = [*validation.findings, finding]
    blocking = [f for f in findings if f.blocking]
    stage = _worst_stage(findings)
    confidence = max(0.0, validation.confidence - _PENALTY[finding.severity])

    return validation.model_copy(
        update={
            "findings": findings,
            "valid": not blocking,
            "confidence": confidence,
            "issues": [f.message for f in blocking],
            "warnings": [
                f.message for f in findings if f.severity is Severity.WARNING
            ],
            "retry_recommended": bool(blocking) and stage is not RetryStage.NONE,
            "retry_stage": stage,
            "checks_run": validation.checks_run + 1,
            "strength": (
                EvidenceStrength.LIMITED if blocking else validation.strength
            ),
        }
    )


__all__ = [
    "MIN_ROWS_FOR_CONFIDENCE",
    "SEMANTIC_PROMPT",
    "check_semantics",
    "merge_finding",
    "validate_result",
]
