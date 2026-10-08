"""Centralised session state.

Phase 3 reached into ``st.session_state`` from several modules with
string keys. That works until two pages disagree about a key's lifetime — and
the lifetime that matters most here is "until a different dataset is loaded",
because analytical context from one dataset applied to another produces
confident nonsense.

So all session state lives behind one object. It operates on any mutable
mapping, which means the whole thing is testable with a plain dict and no
Streamlit runtime; the UI passes ``st.session_state`` in.

The dataset key is a content hash. Re-uploading the same file keeps the
conversation; loading a different one clears context, history, the last result
and any investigation, every time, without the caller having to remember to.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, MutableMapping

import pandas as pd

from app.models.context import AnalysisHistoryItem, AnalyticalContext
from app.models.profile import DatasetProfile

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #

#: Dataset and profile.
KEY_LOAD_RESULT = "load_result"
KEY_PROFILE = "profile"
KEY_ERROR = "load_error"
KEY_DATASET_KEY = "dataset_key"

#: Analytical session.
KEY_CONTEXT = "analytical_context"
KEY_HISTORY = "analysis_history"
KEY_LAST_OUTCOME = "ask_outcome"
KEY_INVESTIGATION = "investigation_outcome"

#: UI preferences that survive a dataset change.
KEY_PAGE = "active_page"

#: Cleared whenever the dataset changes.
DATASET_SCOPED_KEYS = (
    KEY_CONTEXT,
    KEY_HISTORY,
    KEY_LAST_OUTCOME,
    KEY_INVESTIGATION,
)

#: How many analyses to keep. Session-only; nothing is written to disk.
MAX_HISTORY = 25


def dataset_key(df: pd.DataFrame, name: str = "") -> str:
    """A stable identity for a loaded dataset.

    Hashes the shape, column names and dtypes rather than the full contents:
    cheap on a large frame, and specific enough that two different files do not
    collide while the same file reloaded keeps its identity.
    """
    try:
        signature = "|".join(
            [
                name,
                str(df.shape),
                ",".join(str(c) for c in df.columns),
                ",".join(str(t) for t in df.dtypes),
            ]
        )
    except Exception:  # noqa: BLE001 - identity must never raise
        signature = f"{name}:{id(df)}"
    return hashlib.sha1(signature.encode("utf-8", "replace")).hexdigest()[:16]


class SessionManager:
    """Typed access to one user's session."""

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state

    # -- plumbing ---------------------------------------------------------- #

    @property
    def state(self) -> MutableMapping[str, Any]:
        return self._state

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = self._state[key]
        except KeyError:
            return default
        return default if value is None else value

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def init(self) -> None:
        """Seed every key, so no caller has to guard for absence."""
        defaults: dict[str, Any] = {
            KEY_LOAD_RESULT: None,
            KEY_PROFILE: None,
            KEY_ERROR: None,
            KEY_DATASET_KEY: None,
            KEY_CONTEXT: None,
            KEY_HISTORY: [],
            KEY_LAST_OUTCOME: None,
            KEY_INVESTIGATION: None,
        }
        for key, value in defaults.items():
            if key not in self._state:
                self._state[key] = value

    # -- dataset ----------------------------------------------------------- #

    @property
    def load_result(self) -> Any:
        return self.get(KEY_LOAD_RESULT)

    @property
    def dataframe(self) -> pd.DataFrame | None:
        result = self.load_result
        return result.dataframe if result is not None else None

    @property
    def profile(self) -> DatasetProfile | None:
        return self.get(KEY_PROFILE)

    @property
    def dataset_key(self) -> str | None:
        return self.get(KEY_DATASET_KEY)

    @property
    def dataset_name(self) -> str:
        result = self.load_result
        return getattr(result, "source_name", "dataset") if result else "dataset"

    @property
    def error(self) -> str | None:
        return self.get(KEY_ERROR)

    @property
    def has_dataset(self) -> bool:
        return self.load_result is not None and self.profile is not None

    def set_dataset(self, load_result: Any, profile: DatasetProfile) -> bool:
        """Install a dataset, clearing session state if it is a different one.

        Returns True when the analytical session was reset.
        """
        key = dataset_key(load_result.dataframe, getattr(load_result, "source_name", ""))
        changed = key != self.dataset_key

        self.set(KEY_LOAD_RESULT, load_result)
        self.set(KEY_PROFILE, profile)
        self.set(KEY_DATASET_KEY, key)
        self.set(KEY_ERROR, None)

        if changed:
            self._clear_dataset_scoped()
            self.set(KEY_CONTEXT, AnalyticalContext(dataset_key=key))
            logger.info("New dataset %s; analytical session reset", key)
        return changed

    def set_error(self, message: str) -> None:
        """Record a load failure and drop whatever was loaded before."""
        self.set(KEY_LOAD_RESULT, None)
        self.set(KEY_PROFILE, None)
        self.set(KEY_DATASET_KEY, None)
        self.set(KEY_ERROR, message)
        self._clear_dataset_scoped()

    def clear_dataset(self) -> None:
        """Unload the dataset and everything derived from it."""
        self.set(KEY_LOAD_RESULT, None)
        self.set(KEY_PROFILE, None)
        self.set(KEY_DATASET_KEY, None)
        self.set(KEY_ERROR, None)
        self._clear_dataset_scoped()

    def _clear_dataset_scoped(self) -> None:
        for key in DATASET_SCOPED_KEYS:
            self._state[key] = [] if key == KEY_HISTORY else None

    # -- analytical context ------------------------------------------------ #

    @property
    def context(self) -> AnalyticalContext:
        """The current context, always matching the loaded dataset."""
        existing = self.get(KEY_CONTEXT)
        key = self.dataset_key
        if isinstance(existing, AnalyticalContext) and existing.belongs_to(key):
            return existing
        fresh = AnalyticalContext(dataset_key=key)
        self.set(KEY_CONTEXT, fresh)
        return fresh

    def set_context(self, context: AnalyticalContext) -> None:
        self.set(KEY_CONTEXT, context)

    def reset_context(self) -> AnalyticalContext:
        fresh = AnalyticalContext(dataset_key=self.dataset_key)
        self.set(KEY_CONTEXT, fresh)
        return fresh

    # -- history ----------------------------------------------------------- #

    @property
    def history(self) -> list[AnalysisHistoryItem]:
        items = self.get(KEY_HISTORY, [])
        return list(items) if isinstance(items, list) else []

    def add_history(self, item: AnalysisHistoryItem) -> None:
        """Append to the session history, newest last, oldest dropped."""
        items = self.history
        items.append(item)
        self.set(KEY_HISTORY, items[-MAX_HISTORY:])

    def recent_history(self, limit: int = 5) -> list[AnalysisHistoryItem]:
        """The most recent analyses, newest first."""
        return list(reversed(self.history))[:limit]

    def clear_history(self) -> None:
        self.set(KEY_HISTORY, [])

    # -- results ----------------------------------------------------------- #

    @property
    def last_outcome(self) -> Any:
        return self.get(KEY_LAST_OUTCOME)

    def set_last_outcome(self, outcome: Any) -> None:
        self.set(KEY_LAST_OUTCOME, outcome)

    def clear_last_outcome(self) -> None:
        self.set(KEY_LAST_OUTCOME, None)

    @property
    def investigation(self) -> Any:
        return self.get(KEY_INVESTIGATION)

    def set_investigation(self, outcome: Any) -> None:
        self.set(KEY_INVESTIGATION, outcome)

    def clear_investigation(self) -> None:
        self.set(KEY_INVESTIGATION, None)

    # -- diagnostics ------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name if self.has_dataset else None,
            "dataset_key": self.dataset_key,
            "has_dataset": self.has_dataset,
            "context": self.context.to_dict() if self.has_dataset else None,
            "history_length": len(self.history),
            "has_last_outcome": self.last_outcome is not None,
            "has_investigation": self.investigation is not None,
        }


def get_session() -> SessionManager:
    """The manager for the live Streamlit session.

    Imported lazily so that nothing outside the UI depends on Streamlit.
    """
    import streamlit as st

    manager = SessionManager(st.session_state)
    manager.init()
    return manager


__all__ = [
    "DATASET_SCOPED_KEYS",
    "KEY_CONTEXT",
    "KEY_DATASET_KEY",
    "KEY_ERROR",
    "KEY_HISTORY",
    "KEY_INVESTIGATION",
    "KEY_LAST_OUTCOME",
    "KEY_LOAD_RESULT",
    "KEY_PAGE",
    "KEY_PROFILE",
    "MAX_HISTORY",
    "SessionManager",
    "dataset_key",
    "get_session",
]
