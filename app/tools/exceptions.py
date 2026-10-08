"""Domain-specific exceptions for the deterministic analysis tools.

Every tool in :mod:`app.tools` fails with one of these -- never a bare
``KeyError`` or ``TypeError`` from pandas. The messages are written to be shown
to a user, and (in a later phase) handed back to an LLM agent as a correction
signal, so they name the offending column and list the valid alternatives.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable


class ToolError(Exception):
    """Base class for every analysis-tool failure."""


class InvalidDataError(ToolError):
    """The dataframe itself is unusable (wrong type, no columns)."""


class ColumnNotFoundError(ToolError):
    """A requested column does not exist in the dataframe."""

    def __init__(self, column: str, available: Iterable[str]) -> None:
        self.column = column
        self.available = list(available)

        suggestions = get_close_matches(str(column), self.available, n=3, cutoff=0.6)
        message = f"Column '{column}' does not exist."
        if suggestions:
            message += " Did you mean " + " or ".join(f"'{s}'" for s in suggestions) + "?"
        elif self.available:
            shown = ", ".join(f"'{c}'" for c in self.available[:10])
            more = f" (+{len(self.available) - 10} more)" if len(self.available) > 10 else ""
            message += f" Available columns: {shown}{more}."
        super().__init__(message)


class InvalidColumnTypeError(ToolError):
    """A column exists but holds the wrong kind of data for the operation."""

    def __init__(self, column: str, expected: str, actual: str) -> None:
        self.column = column
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Column '{column}' must be {expected}, but it is {actual}."
        )


class InvalidParameterError(ToolError):
    """A parameter is missing, malformed, or out of range."""


class UnsupportedOperationError(InvalidParameterError):
    """An unknown aggregation, filter operator, chart type, or frequency."""

    def __init__(self, kind: str, requested: object, supported: Iterable[str]) -> None:
        self.kind = kind
        self.requested = requested
        self.supported = list(supported)
        super().__init__(
            f"Unsupported {kind}: {requested!r}. "
            f"Supported: {', '.join(sorted(self.supported))}."
        )
