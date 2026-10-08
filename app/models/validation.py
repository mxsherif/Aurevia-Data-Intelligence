"""Validation of a completed analysis, and what to do when it fails.

Phase 3 trusted the executor's own success flag. Phase 4 adds a stage that asks
a harder question: *did we actually answer what was asked?* A result can
execute cleanly and still be wrong — grouped by the wrong field, missing the
filter the user asked for, ranked the wrong way round, or explained with a
figure nobody computed.

Almost every check is deterministic: it compares the plan against the result it
produced. An LLM is used only for the one thing code cannot judge — whether the
prose actually addresses the question — and even then only when the cheap
checks have already passed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    """How badly a check failed."""

    #: The answer is wrong or unusable. Blocks the result.
    ERROR = "error"
    #: The answer stands but something is off. Shown to the user.
    WARNING = "warning"
    #: Worth recording, not worth interrupting for.
    NOTE = "note"

    def __str__(self) -> str:
        return self.value


class RetryStage(str, Enum):
    """Which stage to re-run when a check fails.

    Retrying the whole pipeline for a bad chart choice wastes a planning call
    and risks a different answer; this lets the failure point to the cheapest
    stage that could fix it.
    """

    #: Re-plan: the wrong columns or analysis were chosen.
    PLANNING = "planning"
    #: Re-execute: the computation itself went wrong (rarely retryable).
    EXECUTION = "execution"
    #: Regenerate the explanation only; the numbers are fine.
    INTERPRETATION = "interpretation"
    #: Replace or drop the chart; no analysis or LLM call needed.
    CHARTING = "charting"
    #: Nothing to retry.
    NONE = "none"

    def __str__(self) -> str:
        return self.value


class ValidationIssue(BaseModel):
    """One thing a check found."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    message: str
    #: Which stage could fix it.
    stage: RetryStage = RetryStage.NONE
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "stage": self.stage.value,
            "details": self.details,
        }


class EvidenceStrength(str, Enum):
    """A qualitative read on how much the result can bear.

    Deliberately qualitative. An invented numeric "AI confidence" would imply
    a precision we do not have; these three labels are assigned by counting
    concrete facts (rows analysed, periods available, validation outcome).
    """

    STRONG = "strong"
    MODERATE = "moderate"
    LIMITED = "limited"

    def __str__(self) -> str:
        return self.value


#: What each label means, shown in the UI next to the label itself.
STRENGTH_LABELS: dict[EvidenceStrength, str] = {
    EvidenceStrength.STRONG: "Strong evidence",
    EvidenceStrength.MODERATE: "Moderate evidence",
    EvidenceStrength.LIMITED: "Limited evidence",
}


class ValidationResult(BaseModel):
    """The verdict on one completed analysis."""

    model_config = ConfigDict(extra="forbid")

    valid: bool = True
    #: 0-1, computed from the checks that ran -- not a model's self-report.
    confidence: float = 1.0
    #: Blocking problems, as plain sentences.
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retry_recommended: bool = False
    #: The cheapest stage that could fix the worst problem.
    retry_stage: RetryStage = RetryStage.NONE
    #: Every finding, with its code and severity, for logging and the UI.
    findings: list[ValidationIssue] = Field(default_factory=list)
    #: How many checks ran, so "no issues" can be distinguished from
    #: "nothing was checked".
    checks_run: int = 0
    #: Qualitative strength of the evidence behind the answer.
    strength: EvidenceStrength = EvidenceStrength.STRONG

    # -- accessors --------------------------------------------------------- #

    @property
    def blocking_findings(self) -> list[ValidationIssue]:
        return [f for f in self.findings if f.blocking]

    @property
    def strength_label(self) -> str:
        return STRENGTH_LABELS[self.strength]

    def codes(self) -> list[str]:
        return [f.code for f in self.findings]

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "confidence": round(self.confidence, 3),
            "strength": self.strength.value,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "retry_recommended": self.retry_recommended,
            "retry_stage": self.retry_stage.value,
            "checks_run": self.checks_run,
            "findings": [f.to_dict() for f in self.findings],
        }


class SemanticCheck(BaseModel):
    """The LLM's verdict on whether the prose answers the question.

    The one judgement code cannot make. Kept to two booleans and a sentence so
    the model has no room to editorialise, and consulted only after the
    deterministic checks pass.
    """

    model_config = ConfigDict(extra="forbid")

    addresses_question: bool = Field(
        description=(
            "True if the explanation answers the user's actual question, "
            "rather than a different one."
        )
    )
    contradicts_data: bool = Field(
        description=(
            "True if any statement in the explanation disagrees with the "
            "computed figures supplied alongside it."
        )
    )
    reason: str = Field(
        default="",
        description=(
            "One short sentence naming the problem, when either flag is true. "
            "Empty otherwise."
        ),
    )


__all__ = [
    "STRENGTH_LABELS",
    "EvidenceStrength",
    "RetryStage",
    "SemanticCheck",
    "Severity",
    "ValidationIssue",
    "ValidationResult",
]
