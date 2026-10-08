"""Computed analysis results and the LLM's interpretation of them.

:class:`AnalysisResult` is produced **only** by Python (the executor running
Phase 2 tools). :class:`Insight` is produced by the LLM, which is given the
result and forbidden from introducing numbers of its own.

The split matters: `AnalysisResult.allowed_numbers()` is the exact set of
figures the model is permitted to quote, and
:mod:`app.services.grounding` checks its answer against that set.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisResult(BaseModel):
    """The output of one deterministic analysis run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    success: bool
    title: str = ""

    #: Headline figures: ``{"Total revenue": 45207665.94, ...}``. Rendered as
    #: metric tiles and handed to the insight agent as ground truth.
    summary_data: dict[str, Any] = Field(default_factory=dict)

    #: The full result table, one dict per row.
    table_data: list[dict[str, Any]] | None = None

    #: Keyword arguments for :func:`app.tools.generate_chart_data`, or ``None``
    #: when no chart would help. The UI, not the model, renders it.
    chart_spec: dict[str, Any] | None = None

    #: How the answer was computed: tools used, columns, row counts, notes.
    metadata: dict[str, Any] = Field(default_factory=dict)

    #: A user-facing message when ``success`` is False. Never a stack trace.
    error: str | None = None

    # -- helpers ----------------------------------------------------------- #

    @property
    def has_table(self) -> bool:
        return bool(self.table_data)

    @property
    def has_chart(self) -> bool:
        return self.chart_spec is not None

    @property
    def row_count(self) -> int:
        return len(self.table_data or [])

    @property
    def notes(self) -> list[str]:
        notes = self.metadata.get("notes", [])
        return list(notes) if isinstance(notes, (list, tuple)) else [str(notes)]

    @property
    def tools_used(self) -> list[str]:
        tools = self.metadata.get("tools", [])
        return list(tools) if isinstance(tools, (list, tuple)) else [str(tools)]

    def allowed_numbers(self) -> set[float]:
        """Every number Python actually computed, for grounding the LLM.

        Includes rounded variants, because an answer that says "36%" about a
        computed ``36.0421`` is quoting the result, not inventing a figure.
        """
        collected: set[float] = set()

        def collect(value: Any) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                number = float(value)
                if not math.isfinite(number):
                    return
                collected.add(number)
                for digits in (0, 1, 2, 3):
                    collected.add(round(number, digits))
                # A magnitude-scaled reading ("1.24M" of 1_240_000).
                for scale in (1_000, 1_000_000, 1_000_000_000):
                    if abs(number) >= scale:
                        for digits in (0, 1, 2):
                            collected.add(round(number / scale, digits))
                return
            if isinstance(value, dict):
                for label, item in value.items():
                    # Labels are part of what the model was shown, and they
                    # carry figures: "25th percentile", "Revenue (Q3 2024)".
                    # Without this, quoting the quartile it was given reads as
                    # an invented number.
                    if isinstance(label, str):
                        _collect_from_text(label, collected)
                    collect(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)
                return
            if isinstance(value, str):
                # Strings can legitimately carry figures (period labels such as
                # "2024-Q3"); treat their numerals as computed too.
                _collect_from_text(value, collected)

        collect(self.summary_data)
        collect(self.table_data)
        collect(self.metadata)

        # Rank references ("the top 5") are part of the request, not a claim.
        limit = self.metadata.get("limit")
        upper = int(limit) if isinstance(limit, (int, float)) else self.row_count
        collected.update(float(n) for n in range(0, max(upper, 0) + 1))
        return collected

    def grounding_payload(self, *, max_rows: int = 25) -> dict[str, Any]:
        """The compact, numbers-only view handed to the insight agent."""
        table = self.table_data or []
        payload: dict[str, Any] = {
            "title": self.title,
            "summary": self.summary_data,
            "metadata": {
                key: value
                for key, value in self.metadata.items()
                # Chart plumbing and raw indices are noise to the model.
                if key not in {"chart_spec", "indices"}
            },
        }
        if table:
            payload["table"] = table[:max_rows]
            payload["table_row_count"] = len(table)
            if len(table) > max_rows:
                # Be explicit that this is a prompt-size cap, not a gap in the
                # analysis -- otherwise the model reports it as a data caveat.
                payload["table_note"] = (
                    f"The analysis covered all {len(table)} rows; only the "
                    f"first {max_rows} are listed here to keep this message "
                    "small. Do not describe this as missing data."
                )
        return payload

    @classmethod
    def failure(cls, message: str, **metadata: Any) -> AnalysisResult:
        """A failed result carrying a message safe to show the user."""
        return cls(success=False, title="Analysis could not be completed",
                   error=message, metadata=metadata)


def _collect_from_text(text: str, collected: set[float]) -> None:
    """Pull numerals out of a string so period labels count as computed."""
    import re

    for match in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            collected.add(float(match))
        except ValueError:
            continue


class Insight(BaseModel):
    """The LLM's interpretation of an :class:`AnalysisResult`.

    Every numeric claim here must already appear in the result; the field
    descriptions say so, and :mod:`app.services.grounding` verifies it.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description=(
            "A direct, one-or-two-sentence answer to the user's question. "
            "Quote only figures present in the computed results."
        )
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "One to three additional observations drawn strictly from the "
            "computed results. No causal explanation."
        ),
    )
    caveat: str | None = Field(
        default=None,
        description=(
            "One short caution about reading the result, when warranted "
            "(small samples, missing values, correlation not causation)."
        ),
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Two to four natural-language questions this dataset could "
            "answer next."
        ),
    )

    #: Set by the grounding check, not by the model.
    ungrounded_numbers: list[str] = Field(default_factory=list, exclude=True)

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded_numbers

    def texts(self) -> list[str]:
        """Every user-visible string, for the grounding check."""
        parts = [self.answer, *self.observations]
        if self.caveat:
            parts.append(self.caveat)
        return [p for p in parts if p]
