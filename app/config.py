"""Central configuration for Aurevia.

Values are read from the process environment, which is seeded from a `.env`
file at the project root (see `.env.example`).  Nothing here raises: a missing
or malformed value falls back to a sane default so the application always
starts, even with no API key configured.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SAMPLE_DATASET_PATH = DATASETS_DIR / "sample_telecom_customers.csv"

APP_NAME = "Aurevia"
APP_TAGLINE = "Agentic data intelligence"

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx", ".xls")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: App-specific settings accept either prefix; `AUREVIA_` wins when both are
#: set. `DATAPILOT_` predates the rename and is kept so existing `.env` files
#: keep working.
_ENV_PREFIXES = ("AUREVIA_", "DATAPILOT_")


def _env_keys(key: str) -> tuple[str, ...]:
    """The environment names to try for `key`, in priority order."""
    for prefix in _ENV_PREFIXES:
        if key.startswith(prefix):
            bare = key[len(prefix):]
            return tuple(f"{p}{bare}" for p in _ENV_PREFIXES)
    return (key,)


def _env_str(key: str, default: str = "") -> str:
    for name in _env_keys(key):
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return default


def _env_int(key: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparsable."""
    raw = _env_str(key)
    if not raw:
        return default
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Ignoring invalid value for %s=%r; using default %s", key, raw, default
        )
        return default
    return parsed if parsed > 0 else default


def _env_float(key: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a float from the environment, falling back on anything unparsable."""
    raw = _env_str(key)
    if not raw:
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Ignoring invalid value for %s=%r; using default %s", key, raw, default
        )
        return default
    return parsed if parsed >= minimum else default


# Placeholder values that should be treated as "not configured".
_PLACEHOLDER_KEYS = {
    "",
    "your_api_key_here",
    "your-api-key-here",
    "changeme",
    "none",
    "null",
}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

class Settings(BaseModel):
    """Immutable snapshot of the runtime configuration."""

    model_config = {"frozen": True}

    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-4.1-mini"

    max_upload_mb: int = 200
    max_rows: int = 500_000
    preview_rows: int = 100
    log_level: str = "INFO"

    # -- LLM layer (Phase 3) ---------------------------------------------- #
    #: Seconds before an LLM request is abandoned.
    llm_timeout_seconds: float = 45.0
    #: Retries the SDK performs for transient failures.
    llm_max_retries: int = 2
    #: Low temperature: planning is a structured task, not a creative one.
    llm_temperature: float = 0.1
    #: Cap on the response size, which also caps cost per question.
    llm_max_output_tokens: int = 1_200

    # -- derived ----------------------------------------------------------- #

    @property
    def has_api_key(self) -> bool:
        """True when a usable (non-placeholder) OpenAI key is configured."""
        return self.openai_api_key.strip().lower() not in _PLACEHOLDER_KEYS

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def describe(self) -> dict[str, object]:
        """Redacted view of the configuration, safe to render in the UI."""
        return {
            "openai_model": self.openai_model,
            "api_key_configured": self.has_api_key,
            "max_upload_mb": self.max_upload_mb,
            "max_rows": self.max_rows,
            "preview_rows": self.preview_rows,
            "log_level": self.log_level,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_retries": self.llm_max_retries,
            "llm_temperature": self.llm_temperature,
            "llm_max_output_tokens": self.llm_max_output_tokens,
        }


def load_settings(env_file: str | Path | None = None, *, override: bool = False) -> Settings:
    """Build a :class:`Settings` from the environment (and a `.env` file).

    Never raises -- any unreadable `.env` or malformed value degrades to the
    documented default.
    """
    path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
    try:
        if path.exists():
            load_dotenv(path, override=override)
    except OSError as exc:  # unreadable / locked file
        logging.getLogger(__name__).warning("Could not read %s: %s", path, exc)

    return Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_model=_env_str("OPENAI_MODEL", "gpt-4.1-mini"),
        max_upload_mb=_env_int("DATAPILOT_MAX_UPLOAD_MB", 200),
        max_rows=_env_int("DATAPILOT_MAX_ROWS", 500_000),
        preview_rows=_env_int("DATAPILOT_PREVIEW_ROWS", 100),
        log_level=_env_str("DATAPILOT_LOG_LEVEL", "INFO").upper(),
        llm_timeout_seconds=_env_float("AUREVIA_LLM_TIMEOUT", 45.0, minimum=1.0),
        llm_max_retries=_env_int("AUREVIA_LLM_MAX_RETRIES", 2),
        llm_temperature=_env_float("AUREVIA_LLM_TEMPERATURE", 0.1),
        llm_max_output_tokens=_env_int("AUREVIA_LLM_MAX_OUTPUT_TOKENS", 1_200),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the lifetime of the process."""
    return load_settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests)."""
    get_settings.cache_clear()


def configure_logging(settings: Settings | None = None) -> None:
    """Install a single, idempotent root logging handler."""
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        )
    root.setLevel(level)
