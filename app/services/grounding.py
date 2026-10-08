"""Numerical grounding.

The prompt tells the model to quote only computed figures. This module checks
whether it actually did: every number in the model's prose is matched against
the set of values Python produced, and anything left over is reported.

This is a *detector*, not a rewriter. Phase 4 adds the full validate-and-retry
loop; here the job is to notice an ungrounded figure and let the UI mark the
answer as unverified rather than present an invented number as fact.

Matching is deliberately generous about **formatting** and strict about
**value**: "1.24M", "1,240,000" and "1240000.0" all match a computed
1_240_000, while 1.25M does not match anything.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Numbers, with optional thousands separators, decimals, sign and magnitude
#: suffix. Also catches percentages because the '%' is consumed separately.
_NUMBER = re.compile(
    r"""
    (?<![\w.])                     # not mid-identifier
    (?P<sign>[-+]?)
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,240,000.5
              |\d+(?:\.\d+)?)                  # 1240000.5
    \s*
    (?P<suffix>[kKmMbB](?![\w])|%)?
    """,
    re.VERBOSE,
)

_SUFFIX_SCALE = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}

#: Relative tolerance when comparing a quoted figure with a computed one.
RELATIVE_TOLERANCE = 0.005
#: Absolute tolerance, so small values still compare sensibly.
ABSOLUTE_TOLERANCE = 0.01

#: Numbers that are never a claim about the data: years, ordinals, "top 3".
#: Kept small on purpose -- a wide allowance would defeat the check.
_ALWAYS_ALLOWED = {0.0, 1.0, 2.0, 3.0}


@dataclass
class GroundingReport:
    """What the check found."""

    #: Every numeric token seen in the text, as written.
    found: list[str] = field(default_factory=list)
    #: Tokens with no matching computed value.
    ungrounded: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded

    @property
    def checked_count(self) -> int:
        return len(self.found)

    def to_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked_count,
            "ungrounded": list(self.ungrounded),
            "is_grounded": self.is_grounded,
        }


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Find the numbers in `text` as ``(as_written, value)`` pairs.

    A magnitude suffix is expanded (``1.2M`` -> 1_200_000) and a percent sign is
    kept as the bare number, since a computed percentage is stored as 36.04,
    not 0.3604.
    """
    results: list[tuple[str, float]] = []
    for match in _NUMBER.finditer(text or ""):
        raw = match.group(0).strip()
        digits = match.group("digits").replace(",", "")
        try:
            value = float(digits)
        except ValueError:  # pragma: no cover - the regex guarantees digits
            continue
        if match.group("sign") == "-":
            value = -value

        suffix = (match.group("suffix") or "").lower()
        scaled = [value]
        if suffix in _SUFFIX_SCALE:
            # Accept both readings: the model may write "1.24M" about a
            # computed 1_240_000 or about a computed 1.24 in a millions column.
            scaled.append(value * _SUFFIX_SCALE[suffix])
        for candidate in scaled:
            results.append((raw, candidate))
    return results


def _matches(value: float, allowed: set[float]) -> bool:
    if not math.isfinite(value):
        return True  # nothing to verify
    if abs(value) in _ALWAYS_ALLOWED or value in _ALWAYS_ALLOWED:
        return True
    if value in allowed:
        return True
    for candidate in allowed:
        if math.isclose(
            value, candidate,
            rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE,
        ):
            return True
        # A sign-flipped reading of a computed decline is still that figure.
        if math.isclose(
            abs(value), abs(candidate),
            rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE,
        ):
            return True
    return False


def check_text(text: str, allowed: set[float]) -> GroundingReport:
    """Check one string against the set of computed values."""
    report = GroundingReport()
    # One token can yield several readings (plain and scaled); it is grounded
    # if *any* reading matches.
    by_token: dict[str, list[float]] = {}
    for raw, value in extract_numbers(text):
        by_token.setdefault(raw, []).append(value)

    for raw, values in by_token.items():
        report.found.append(raw)
        if not any(_matches(value, allowed) for value in values):
            report.ungrounded.append(raw)
    return report


def check_texts(texts: list[str], allowed: set[float]) -> GroundingReport:
    """Check several strings, merging the findings."""
    combined = GroundingReport()
    for text in texts:
        report = check_text(text, allowed)
        for token in report.found:
            if token not in combined.found:
                combined.found.append(token)
        for token in report.ungrounded:
            if token not in combined.ungrounded:
                combined.ungrounded.append(token)
    return combined


def verify_insight(insight, result) -> GroundingReport:
    """Check an :class:`~app.models.results.Insight` against its result.

    Records the findings on the insight itself, so the UI can show the answer
    with a clear caution instead of silently trusting it.
    """
    allowed = result.allowed_numbers()
    report = check_texts(insight.texts(), allowed)
    insight.ungrounded_numbers = list(report.ungrounded)

    if report.ungrounded:
        logger.warning(
            "Insight contains %d ungrounded figure(s): %s",
            len(report.ungrounded), report.ungrounded,
        )
    return report


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "GroundingReport",
    "RELATIVE_TOLERANCE",
    "check_text",
    "check_texts",
    "extract_numbers",
    "verify_insight",
]
