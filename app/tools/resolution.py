"""Safe column resolution.

Users — and the planner, reading their words — say "monthly bill" when the
column is `monthly_charge`, or "sales" when it is `revenue`. This module maps
such a term onto a real column through a fixed ladder of increasingly loose
strategies, and reports **how** it matched so the caller can refuse a guess it
does not trust.

The ladder, most to least certain:

1. exact name
2. normalised name (case, separators, pluralisation)
3. a curated alias table of business synonyms
4. token overlap (every word of the term appears in the column, or vice versa)
5. fuzzy string similarity, above a deliberately high cut-off

Anything weaker is *not* a match. Silently analysing the wrong column is the
worst failure mode available to a tool like this, so an unresolved term comes
back as ``None`` with candidate suggestions instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Iterable

import pandas as pd

from app.tools.validation import require_dataframe

#: Similarity above which a fuzzy match is accepted (0-1). High on purpose.
FUZZY_THRESHOLD = 0.86
#: Similarity above which a term is offered as a *suggestion* but not accepted.
SUGGESTION_THRESHOLD = 0.55
#: Words that carry no identifying information in a column reference.
STOP_WORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "per", "by", "in", "on", "at", "to",
        "and", "or", "total", "value", "values", "amount", "number", "num",
        "column", "field", "data", "customer", "customers", "each", "all",
        "average", "avg", "mean", "my",
    }
)

#: Business synonyms -> canonical column tokens. Keys are normalised.
#: Curated rather than inferred: a wrong entry here is a silent mis-analysis,
#: so each one is a deliberate choice.
ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "sales": ("revenue", "sales", "total_revenue", "amount"),
    "turnover": ("revenue", "sales"),
    "income": ("revenue", "income"),
    "earnings": ("revenue", "income"),
    "spend": ("revenue", "monthly_charge", "charge"),
    "bill": ("monthly_charge", "charge", "bill"),
    "monthly_bill": ("monthly_charge", "charge"),
    "price": ("monthly_charge", "price", "charge"),
    "cost": ("monthly_charge", "cost", "charge"),
    "fee": ("monthly_charge", "fee", "charge"),
    "arpu": ("monthly_charge", "revenue"),
    "satisfaction": ("satisfaction_score", "satisfaction", "csat", "nps"),
    "happiness": ("satisfaction_score", "satisfaction"),
    "rating": ("satisfaction_score", "rating", "score"),
    "score": ("satisfaction_score", "score"),
    "churn": ("churn", "churned", "is_churn", "attrition"),
    "attrition": ("churn", "churned"),
    "retention": ("churn", "retained"),
    "cancelled": ("churn", "churned"),
    "usage": ("data_usage_gb", "usage", "data_usage"),
    "data": ("data_usage_gb", "data_usage"),
    "data_usage": ("data_usage_gb", "data_usage"),
    "calls": ("support_calls", "calls", "call_count"),
    "support": ("support_calls", "support_tickets"),
    "tickets": ("support_calls", "support_tickets", "tickets"),
    "complaints": ("support_calls", "complaints"),
    "tenure": ("tenure_months", "tenure", "months_active"),
    "age": ("tenure_months", "age"),
    "loyalty": ("tenure_months", "tenure"),
    "contract": ("contract_type", "contract"),
    "plan": ("contract_type", "plan", "plan_type", "product_type"),
    "subscription": ("contract_type", "subscription_type", "plan"),
    "product": ("product_type", "product"),
    "service": ("product_type", "service_type", "product"),
    "network": ("network_type", "network"),
    "technology": ("network_type", "technology"),
    "payment": ("payment_method", "payment"),
    "payment_type": ("payment_method",),
    "region": ("region", "area", "zone", "territory"),
    "area": ("region", "area"),
    "territory": ("region", "territory"),
    "market": ("region", "market"),
    "location": ("city", "region", "location"),
    "city": ("city", "town", "location"),
    "geography": ("region", "city"),
    "date": ("signup_date", "date", "created_at", "timestamp"),
    "signup": ("signup_date", "signup"),
    "joined": ("signup_date", "join_date"),
    "time": ("signup_date", "date", "timestamp"),
    "month": ("signup_date", "date", "month"),
    "id": ("customer_id", "id"),
    "identifier": ("customer_id", "id"),
    "account": ("customer_id", "account_id"),
}

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _singularize(token: str) -> str:
    """Fold a plural onto its stem.

    Consistency matters more than linguistic accuracy here: both sides of a
    comparison go through this, so "status" -> "statu" is harmless as long as
    it happens every time. What is *not* harmless is "charges" -> "charg"
    while "charge" stays put, which is why the "es" ending is not stripped
    wholesale.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalized_alias_map() -> dict[str, tuple[str, ...]]:
    """ALIAS_MAP keyed by normalised term, so lookups match normalised input.

    Without this, the entry for "sales" is unreachable: the term normalises to
    "sale" before it ever reaches the table.
    """
    merged: dict[str, tuple[str, ...]] = {}
    for term, targets in ALIAS_MAP.items():
        key = normalize(term)
        if not key:
            continue
        existing = merged.get(key, ())
        merged[key] = existing + tuple(t for t in targets if t not in existing)
    return merged


class MatchKind(str, Enum):
    """How a term was matched, in descending order of confidence."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    ALIAS = "alias"
    TOKEN = "token_overlap"
    FUZZY = "fuzzy"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


#: Match kinds we are willing to act on without asking the user.
TRUSTED_KINDS = frozenset(
    {MatchKind.EXACT, MatchKind.NORMALIZED, MatchKind.ALIAS, MatchKind.TOKEN}
)

#: Confidence attached to each kind, for display and thresholding.
CONFIDENCE: dict[MatchKind, float] = {
    MatchKind.EXACT: 1.0,
    MatchKind.NORMALIZED: 0.97,
    MatchKind.ALIAS: 0.9,
    MatchKind.TOKEN: 0.8,
    MatchKind.FUZZY: 0.7,
    MatchKind.NONE: 0.0,
}


@dataclass
class ColumnMatch:
    """The outcome of resolving one term."""

    term: str
    column: str | None
    kind: MatchKind = MatchKind.NONE
    confidence: float = 0.0
    #: Plausible alternatives, for a clarification message.
    candidates: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.column is not None

    @property
    def is_trusted(self) -> bool:
        """True when the match is strong enough to use without asking."""
        return self.found and self.kind in TRUSTED_KINDS

    @property
    def was_renamed(self) -> bool:
        """True when the resolved column differs from what was asked for."""
        return self.found and self.column != self.term

    def describe(self) -> str:
        if not self.found:
            suggestion = (
                " Closest columns: " + ", ".join(self.candidates[:3])
                if self.candidates
                else ""
            )
            return f"No column matches '{self.term}'.{suggestion}"
        if self.kind is MatchKind.EXACT:
            return f"'{self.column}'"
        return f"'{self.term}' -> '{self.column}' ({self.kind} match)"


def normalize(name: str) -> str:
    """Reduce a name to comparable form: lowercase, de-punctuated, singular."""
    text = _NON_WORD.sub("_", str(name).strip().lower()).strip("_")
    if not text:
        return ""
    return "_".join(_singularize(token) for token in text.split("_") if token)


def tokens(name: str) -> set[str]:
    """Meaningful tokens of a name, stop words removed."""
    raw = {t for t in normalize(name).split("_") if t}
    meaningful = {t for t in raw if t not in STOP_WORDS and len(t) > 1}
    # Never return nothing: a term made only of stop words still has to match.
    return meaningful or raw


#: ALIAS_MAP keyed by normalised term; built once, after normalize() exists.
NORMALIZED_ALIASES: dict[str, tuple[str, ...]] = _normalized_alias_map()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def resolve_column(
    term: str,
    columns: Iterable[str],
    *,
    allowed: Iterable[str] | None = None,
) -> ColumnMatch:
    """Resolve `term` to one of `columns`.

    `allowed` optionally narrows the search to a subset (the numeric columns,
    say), which both improves accuracy and prevents a metric from resolving
    onto a categorical field.
    """
    available = [str(c) for c in columns]
    pool = [c for c in (list(allowed) if allowed is not None else available)
            if c in available] if allowed is not None else available

    if not isinstance(term, str) or not term.strip():
        return ColumnMatch(term=str(term), column=None, candidates=pool[:3])
    if not pool:
        return ColumnMatch(term=term, column=None)

    # 1. Exact.
    if term in pool:
        return ColumnMatch(term, term, MatchKind.EXACT, CONFIDENCE[MatchKind.EXACT])

    normalized_term = normalize(term)
    normalized_pool = {column: normalize(column) for column in pool}

    # 2. Normalised.
    for column, normalized in normalized_pool.items():
        if normalized and normalized == normalized_term:
            return ColumnMatch(
                term, column, MatchKind.NORMALIZED, CONFIDENCE[MatchKind.NORMALIZED]
            )

    # 3. Curated aliases, tried on the whole term then on its tokens.
    alias_targets: list[str] = []
    for key in (normalized_term, *sorted(tokens(term))):
        alias_targets.extend(NORMALIZED_ALIASES.get(key, ()))
    for target in alias_targets:
        normalized_target = normalize(target)
        for column, normalized in normalized_pool.items():
            if normalized == normalized_target:
                return ColumnMatch(
                    term, column, MatchKind.ALIAS, CONFIDENCE[MatchKind.ALIAS]
                )

    # 4. Token containment, in either direction, when unambiguous.
    term_tokens = tokens(term)
    overlaps = [
        column
        for column, _ in normalized_pool.items()
        if term_tokens and (
            term_tokens <= tokens(column) or tokens(column) <= term_tokens
        )
    ]
    if len(overlaps) == 1:
        return ColumnMatch(term, overlaps[0], MatchKind.TOKEN, CONFIDENCE[MatchKind.TOKEN])

    # 5. Fuzzy, above a high threshold and only when one candidate stands out.
    scored = sorted(
        ((column, _similarity(normalized_term, normalized))
         for column, normalized in normalized_pool.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if scored and scored[0][1] >= FUZZY_THRESHOLD:
        runner_up = scored[1][1] if len(scored) > 1 else 0.0
        if scored[0][1] - runner_up > 0.05 or len(scored) == 1:
            return ColumnMatch(
                term, scored[0][0], MatchKind.FUZZY, round(scored[0][1], 3)
            )

    candidates = [column for column, score in scored if score >= SUGGESTION_THRESHOLD]
    if overlaps:
        # Ambiguous token overlap is still the best hint we have.
        candidates = list(dict.fromkeys(overlaps + candidates))
    return ColumnMatch(
        term, None, MatchKind.NONE, 0.0, candidates=candidates[:5] or [c for c, _ in scored[:3]]
    )


def resolve_columns(
    terms: Iterable[str],
    columns: Iterable[str],
    *,
    allowed: Iterable[str] | None = None,
) -> list[ColumnMatch]:
    """Resolve several terms, preserving order and skipping blanks."""
    return [
        resolve_column(term, columns, allowed=allowed)
        for term in terms
        if term is not None and str(term).strip()
    ]


def resolve_in_dataframe(
    df: pd.DataFrame,
    term: str,
    *,
    allowed: Iterable[str] | None = None,
) -> ColumnMatch:
    """:func:`resolve_column` against a dataframe's own columns."""
    require_dataframe(df)
    return resolve_column(term, [str(c) for c in df.columns], allowed=allowed)


__all__ = [
    "ALIAS_MAP",
    "NORMALIZED_ALIASES",
    "CONFIDENCE",
    "FUZZY_THRESHOLD",
    "ColumnMatch",
    "MatchKind",
    "TRUSTED_KINDS",
    "normalize",
    "resolve_column",
    "resolve_columns",
    "resolve_in_dataframe",
    "tokens",
]
