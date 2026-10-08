"""Models for Investigate mode.

An investigation is not one analysis but a bounded sequence: confirm the
premise, quantify the change, decompose it across a few dimensions, check the
timing, rank what moved. Every figure in an
:class:`InvestigationResult` is computed by Python; the LLM contributes the
plan that selects *what* to look at, and the prose at the end.

The contribution arithmetic is the part most easily got wrong, so the models
here carry both readings of it and a flag for when the net-share reading would
mislead. See :class:`InvestigationFinding`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Direction(str, Enum):
    """Which way a figure moved."""

    UP = "increase"
    DOWN = "decrease"
    FLAT = "no change"

    def __str__(self) -> str:
        return self.value


#: Words in a question that assert a direction of travel. Used to check the
#: user's premise against what the data actually did.
DIRECTION_CLAIMS: dict[Direction, tuple[str, ...]] = {
    Direction.DOWN: (
        "decline", "declined", "declining", "drop", "dropped", "dropping",
        "fall", "fell", "falling", "decrease", "decreased", "decreasing",
        "down", "lower", "worse", "worsen", "worsened", "deteriorate",
        "deteriorated", "shrink", "shrank", "shrunk", "lose", "lost", "losing",
        "slump", "slumped", "plunge", "plunged", "weaken", "weakened",
        "reduction", "reduced", "contraction",
    ),
    Direction.UP: (
        "increase", "increased", "increasing", "rise", "rose", "rising",
        "grow", "grew", "growing", "growth", "up", "higher", "better",
        "improve", "improved", "improving", "surge", "surged", "jump",
        "jumped", "climb", "climbed", "gain", "gained", "spike", "spiked",
        "expansion", "expanded",
    ),
}


class InvestigationPlan(BaseModel):
    """What the LLM proposes to investigate.

    Structured output, validated against the dataset before anything runs.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(
        description="The numeric column whose change is being investigated."
    )
    time_column: str = Field(
        description="The date column that defines the periods."
    )
    aggregation: str = Field(
        default="sum",
        description="How to reduce the metric per period: sum, mean or count.",
    )
    granularity: str = Field(
        default="quarterly",
        description="Period size: daily, weekly, monthly, quarterly or yearly.",
    )
    baseline_period: str = Field(
        default="previous",
        description=(
            "Which period is the baseline. Use 'previous' for the period "
            "immediately before the comparison period."
        ),
    )
    comparison_period: str = Field(
        default="latest",
        description="Which period to compare. Use 'latest' for the most recent.",
    )
    candidate_dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Categorical columns worth decomposing the change across, best "
            "first. Three to five."
        ),
    )
    claimed_direction: str = Field(
        default="unknown",
        description=(
            "What the question asserts happened: 'decrease', 'increase', or "
            "'unknown' when the question makes no claim."
        ),
    )
    steps: list[str] = Field(
        default_factory=list,
        description=(
            "User-facing plan, one short line per step. No reasoning, no "
            "numbers."
        ),
    )
    requires_clarification: bool = False
    clarification_question: str | None = None

    def describe_steps(self) -> list[str]:
        if self.steps:
            return list(self.steps)
        steps = [
            f"Confirm the change in {self.metric}",
            f"Quantify the {self.granularity} change",
        ]
        steps += [
            f"Measure the contribution of each {d}" for d in self.candidate_dimensions
        ]
        steps += ["Inspect the timing within the period", "Rank the strongest movers"]
        return steps


class InvestigationFinding(BaseModel):
    """One category's contribution to the overall change.

    Two readings of "contribution" are carried deliberately:

    - ``contribution_pct`` is the category's share of the **net** change
      (``category_change / overall_change``). It is the intuitive reading, and
      it is only meaningful when categories move mostly together. With
      offsetting movement it produces absurdities like "contributed 340% of the
      decline", so it is set to ``None`` when the net change is small relative
      to total movement.
    - ``share_of_movement_pct`` is the category's share of **gross** movement
      (``|category_change| / sum(|category_change|)``). Always defined, always
      0-100, and the honest answer to "where did the movement happen".
    """

    model_config = ConfigDict(extra="forbid")

    dimension: str
    category: str
    baseline_value: float
    comparison_value: float
    absolute_change: float
    percentage_change: float | None = None
    #: Share of the net change; ``None`` when that reading would mislead.
    contribution: float | None = None
    #: Share of gross movement across all categories of this dimension.
    share_of_movement: float = 0.0
    rank: int = 0
    direction: Direction = Direction.FLAT
    #: Rows behind this category, for the sufficiency warnings.
    baseline_rows: int = 0
    comparison_rows: int = 0

    @property
    def total_rows(self) -> int:
        return self.baseline_rows + self.comparison_rows

    @property
    def moved_against_the_trend(self) -> bool:
        """True when this category moved opposite to the overall change."""
        return self.contribution is not None and self.contribution < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "category": self.category,
            "baseline_value": self.baseline_value,
            "comparison_value": self.comparison_value,
            "absolute_change": self.absolute_change,
            "percentage_change": self.percentage_change,
            "contribution": self.contribution,
            "share_of_movement": self.share_of_movement,
            "rank": self.rank,
            "direction": self.direction.value,
            "rows": self.total_rows,
        }


class DimensionBreakdown(BaseModel):
    """Everything computed for one dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: str
    findings: list[InvestigationFinding] = Field(default_factory=list)
    #: True when positive and negative movements largely cancel out, so the
    #: net-share reading of contribution was suppressed.
    offsetting: bool = False
    #: Total gross movement across the categories, for context.
    gross_movement: float = 0.0
    categories_examined: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def strongest(self) -> InvestigationFinding | None:
        return self.findings[0] if self.findings else None

    def top(self, direction: Direction, limit: int = 3) -> list[InvestigationFinding]:
        return [f for f in self.findings if f.direction is direction][:limit]


class TemporalFinding(BaseModel):
    """One sub-period inside the compared window."""

    model_config = ConfigDict(extra="forbid")

    period: str
    value: float
    change_vs_previous: float | None = None
    percentage_change: float | None = None
    rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "value": self.value,
            "change_vs_previous": self.change_vs_previous,
            "percentage_change": self.percentage_change,
            "rows": self.rows,
        }


class InvestigationResult(BaseModel):
    """The complete computed evidence for one investigation."""

    model_config = ConfigDict(extra="forbid")

    success: bool = True
    metric: str = ""
    aggregation: str = "sum"
    time_column: str = ""
    granularity: str = "quarterly"

    baseline_label: str = ""
    comparison_label: str = ""
    baseline_value: float = 0.0
    comparison_value: float = 0.0
    absolute_change: float = 0.0
    percentage_change: float | None = None
    direction: Direction = Direction.FLAT

    #: Whether the question's premise matched what the data shows.
    premise_confirmed: bool = True
    claimed_direction: Direction | None = None
    premise_message: str | None = None

    breakdowns: list[DimensionBreakdown] = Field(default_factory=list)
    temporal_findings: list[TemporalFinding] = Field(default_factory=list)

    #: Dimensions considered but not analysed, with the reason.
    skipped_dimensions: dict[str, str] = Field(default_factory=dict)
    #: Row counts, periods available, and other provenance.
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    #: Deterministic data-sufficiency cautions.
    warnings: list[str] = Field(default_factory=list)
    steps_completed: list[str] = Field(default_factory=list)
    error: str | None = None

    # -- accessors --------------------------------------------------------- #

    @property
    def dimensions_inspected(self) -> list[str]:
        return [b.dimension for b in self.breakdowns]

    @property
    def all_findings(self) -> list[InvestigationFinding]:
        return [f for b in self.breakdowns for f in b.findings]

    def ranked_findings(self, limit: int = 5) -> list[InvestigationFinding]:
        """The strongest movers in the overall direction, across dimensions.

        Ranked by gross movement so the list is dominated by what actually
        moved, not by whichever dimension happened to be examined first.
        """
        aligned = [
            f for f in self.all_findings
            if self.direction is Direction.FLAT or f.direction is self.direction
        ]
        aligned.sort(key=lambda f: abs(f.absolute_change), reverse=True)
        return aligned[:limit]

    def counter_findings(self, limit: int = 3) -> list[InvestigationFinding]:
        """Categories that moved *against* the overall direction."""
        if self.direction is Direction.FLAT:
            return []
        opposite = Direction.UP if self.direction is Direction.DOWN else Direction.DOWN
        against = [f for f in self.all_findings if f.direction is opposite]
        against.sort(key=lambda f: abs(f.absolute_change), reverse=True)
        return against[:limit]

    @property
    def worst_period(self) -> TemporalFinding | None:
        """The sub-period that moved furthest in the overall direction."""
        scored = [f for f in self.temporal_findings if f.change_vs_previous is not None]
        if not scored:
            return None
        if self.direction is Direction.UP:
            return max(scored, key=lambda f: f.change_vs_previous)
        return min(scored, key=lambda f: f.change_vs_previous)

    def to_evidence(self, *, max_findings: int = 12) -> dict[str, Any]:
        """The compact, computed-values-only payload for the LLM summary."""
        return {
            "metric": self.metric,
            "aggregation": self.aggregation,
            "periods_compared": {
                "baseline": {
                    "label": self.baseline_label, "value": self.baseline_value
                },
                "comparison": {
                    "label": self.comparison_label, "value": self.comparison_value
                },
            },
            "overall_change": {
                "absolute": self.absolute_change,
                "percentage": self.percentage_change,
                "direction": self.direction.value,
            },
            "premise_confirmed": self.premise_confirmed,
            "contributors": [
                f.to_dict() for f in self.ranked_findings(limit=max_findings)
            ],
            "moved_against_the_trend": [
                f.to_dict() for f in self.counter_findings()
            ],
            "timing": [f.to_dict() for f in self.temporal_findings],
            "dimensions_inspected": self.dimensions_inspected,
            "evidence": self.evidence_summary,
            "data_cautions": list(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "metric": self.metric,
            "time_column": self.time_column,
            "granularity": self.granularity,
            "baseline": {"label": self.baseline_label, "value": self.baseline_value},
            "comparison": {
                "label": self.comparison_label, "value": self.comparison_value
            },
            "absolute_change": self.absolute_change,
            "percentage_change": self.percentage_change,
            "direction": self.direction.value,
            "premise_confirmed": self.premise_confirmed,
            "premise_message": self.premise_message,
            "breakdowns": [
                {
                    "dimension": b.dimension,
                    "offsetting": b.offsetting,
                    "findings": [f.to_dict() for f in b.findings],
                    "notes": b.notes,
                }
                for b in self.breakdowns
            ],
            "temporal_findings": [f.to_dict() for f in self.temporal_findings],
            "skipped_dimensions": dict(self.skipped_dimensions),
            "evidence_summary": self.evidence_summary,
            "warnings": list(self.warnings),
            "steps_completed": list(self.steps_completed),
            "error": self.error,
        }

    @classmethod
    def failure(cls, message: str, **fields: Any) -> InvestigationResult:
        return cls(success=False, error=message, **fields)


class InvestigationSummary(BaseModel):
    """The LLM's write-up of a completed investigation.

    The field descriptions carry the wording discipline: these are findings
    about association and concentration, not claims about cause.
    """

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(
        description=(
            "One or two sentences stating what changed and where it was "
            "concentrated, quoting only supplied figures."
        )
    )
    contributors: list[str] = Field(
        default_factory=list,
        description=(
            "Two to four lines on the strongest contributors, each naming a "
            "category and its supplied figure."
        ),
    )
    timing: str | None = Field(
        default=None,
        description=(
            "One sentence on when within the period the movement was largest, "
            "if the timing data shows one."
        ),
    )
    caution: str | None = Field(
        default=None,
        description=(
            "One short caution: offsetting movement, small samples, missing "
            "data, or that this shows association rather than cause."
        ),
    )
    next_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Two to four follow-up questions this dataset could answer, each "
            "specific to what was found."
        ),
    )

    #: Set by the grounding check, not by the model.
    ungrounded_numbers: list[str] = Field(default_factory=list, exclude=True)

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded_numbers

    def texts(self) -> list[str]:
        parts = [self.headline, *self.contributors]
        if self.timing:
            parts.append(self.timing)
        if self.caution:
            parts.append(self.caution)
        return [p for p in parts if p]


__all__ = [
    "DIRECTION_CLAIMS",
    "DimensionBreakdown",
    "Direction",
    "InvestigationFinding",
    "InvestigationPlan",
    "InvestigationResult",
    "InvestigationSummary",
    "TemporalFinding",
]
