"""The OpenAI service layer.

One place that talks to the model. Everything above it (the planner and the
insight agent) asks for a *typed* result and receives either a validated
Pydantic object or an :class:`LLMError` subclass with a message fit for a user.

Three properties matter here:

- **It never crashes the app.** A missing key is a state (`available is False`),
  not an exception at import or construction time. Overview, Explore and
  Visualize must keep working with no key configured.
- **Structured output only.** `complete_structured()` uses the SDK's schema
  parsing, so there is no free-form text to parse and no prompt-injection path
  into our control flow.
- **Technical detail is logged, not shown.** Every failure logs the real cause
  and raises a sentence the UI can print verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

Message = dict[str, str]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class LLMError(Exception):
    """Base class for every LLM failure. The message is user-facing."""

    #: Whether retrying the same request might succeed.
    transient = False


class LLMUnavailableError(LLMError):
    """No usable API key, or the SDK is not installed."""


class LLMTimeoutError(LLMError):
    """The model did not respond in time."""

    transient = True


class LLMRateLimitError(LLMError):
    """The account hit a rate or quota limit."""

    transient = True


class LLMAuthError(LLMError):
    """The key was rejected."""


class LLMResponseError(LLMError):
    """The model replied, but not with something we can use."""


class LLMConnectionError(LLMError):
    """The request never reached OpenAI."""

    transient = True


# --------------------------------------------------------------------------- #
# Usage accounting
# --------------------------------------------------------------------------- #

@dataclass
class LLMUsage:
    """Token accounting for one call, surfaced in the UI for transparency."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class StructuredResponse:
    """A parsed model response plus what it cost."""

    data: Any
    usage: LLMUsage = field(default_factory=LLMUsage)


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #

#: Shown wherever AI features are unavailable.
NO_KEY_MESSAGE = (
    "AI analysis requires an OpenAI API key. Add `OPENAI_API_KEY` to your "
    "`.env` file and restart Aurevia. Everything else — Overview, Explore and "
    "Visualize — works without one."
)


class LLMService:
    """A thin, typed wrapper over the OpenAI chat completions API."""

    def __init__(self, settings: Settings | None = None, client: Any = None) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._client_error: str | None = None
        #: Set when a caller injects a client, so tests bypass the key check.
        self._injected = client is not None

    # -- availability ------------------------------------------------------ #

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def model(self) -> str:
        return self._settings.openai_model

    @property
    def available(self) -> bool:
        """True when a request could actually be made."""
        if self._injected:
            return True
        return self._settings.has_api_key and self._import_openai() is not None

    @property
    def unavailable_reason(self) -> str | None:
        """Why AI features are off, or ``None`` when they are on."""
        if self._injected:
            return None
        if not self._settings.has_api_key:
            return NO_KEY_MESSAGE
        if self._import_openai() is None:
            return (
                "The `openai` package is not installed. Run "
                "`pip install -r requirements.txt` and restart Aurevia."
            )
        return None

    @staticmethod
    def _import_openai():
        try:
            import openai
        except ImportError:  # pragma: no cover - environment-dependent
            return None
        return openai

    def _ensure_client(self) -> Any:
        """Build the client on first use, so import time stays side-effect free."""
        if self._client is not None:
            return self._client

        if not self._settings.has_api_key:
            raise LLMUnavailableError(NO_KEY_MESSAGE)

        openai = self._import_openai()
        if openai is None:
            raise LLMUnavailableError(self.unavailable_reason or NO_KEY_MESSAGE)

        try:
            self._client = openai.OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - SDK construction is opaque
            logger.exception("Could not construct the OpenAI client")
            raise LLMUnavailableError(
                f"Could not initialise the OpenAI client: {exc}"
            ) from exc
        return self._client

    # -- the one call that matters ----------------------------------------- #

    def complete_structured(
        self,
        messages: Sequence[Message],
        schema: type[ModelT],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredResponse:
        """Ask the model to fill `schema` and return the validated instance.

        Raises an :class:`LLMError` subclass -- and nothing else -- on failure.
        """
        if not messages:
            raise LLMResponseError("No prompt was supplied to the model.")

        client = self._ensure_client()
        settings = self._settings

        try:
            completion = client.chat.completions.parse(
                model=settings.openai_model,
                messages=list(messages),
                response_format=schema,
                temperature=(
                    settings.llm_temperature if temperature is None else temperature
                ),
                max_completion_tokens=(
                    settings.llm_max_output_tokens
                    if max_output_tokens is None
                    else max_output_tokens
                ),
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            raise self._translate(exc) from exc

        return StructuredResponse(
            data=self._extract(completion, schema),
            usage=self._usage(completion),
        )

    # -- response handling ------------------------------------------------- #

    def _extract(self, completion: Any, schema: type[ModelT]) -> ModelT:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMResponseError("The model returned an empty response.")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)

        if finish_reason == "length":
            raise LLMResponseError(
                "The model's response was cut short. Try a more specific question."
            )
        if finish_reason == "content_filter":
            raise LLMResponseError(
                "The request was blocked by OpenAI's content filter."
            )
        if message is None:
            raise LLMResponseError("The model returned no message content.")

        refusal = getattr(message, "refusal", None)
        if refusal:
            logger.warning("Model refused the request: %s", refusal)
            raise LLMResponseError(f"The model declined this request: {refusal}")

        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, schema):
            return parsed

        # Fall back to validating the raw JSON ourselves. Some SDK/model
        # combinations populate `content` but not `parsed`.
        content = getattr(message, "content", None)
        if parsed is not None or content:
            try:
                if parsed is not None:
                    return schema.model_validate(parsed)
                return schema.model_validate_json(content)
            except ValidationError as exc:
                logger.warning("Structured output failed validation: %s", exc)
                raise LLMResponseError(
                    "The model's reply did not match the expected structure."
                ) from exc
            except Exception as exc:  # noqa: BLE001 - malformed JSON
                logger.warning("Could not read the model's reply: %s", exc)
                raise LLMResponseError(
                    "The model's reply could not be read as structured data."
                ) from exc

        raise LLMResponseError("The model returned no usable content.")

    def _usage(self, completion: Any) -> LLMUsage:
        usage = getattr(completion, "usage", None)
        return LLMUsage(
            model=str(getattr(completion, "model", self.model)),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    # -- error translation -------------------------------------------------- #

    def _translate(self, exc: Exception) -> LLMError:
        """Turn an SDK exception into a domain error with a usable message."""
        if isinstance(exc, LLMError):
            return exc

        openai = self._import_openai()
        name = type(exc).__name__

        if openai is not None:
            if isinstance(exc, openai.APITimeoutError):
                logger.warning("OpenAI request timed out after %ss",
                               self._settings.llm_timeout_seconds)
                return LLMTimeoutError(
                    f"The AI request timed out after "
                    f"{self._settings.llm_timeout_seconds:.0f}s. Try again, or "
                    "ask a narrower question."
                )
            if isinstance(exc, openai.RateLimitError):
                logger.warning("OpenAI rate limit: %s", exc)
                return LLMRateLimitError(
                    "The OpenAI rate limit or quota was reached. Wait a moment "
                    "and try again, or check your account's usage limits."
                )
            if isinstance(exc, openai.AuthenticationError):
                logger.error("OpenAI rejected the API key")
                return LLMAuthError(
                    "OpenAI rejected the API key. Check `OPENAI_API_KEY` in "
                    "your `.env` file."
                )
            if isinstance(exc, openai.PermissionDeniedError):
                logger.error("OpenAI permission denied: %s", exc)
                return LLMAuthError(
                    f"This API key cannot use the model "
                    f"'{self._settings.openai_model}'."
                )
            if isinstance(exc, openai.NotFoundError):
                logger.error("Model not found: %s", exc)
                return LLMResponseError(
                    f"The model '{self._settings.openai_model}' was not found. "
                    "Check `OPENAI_MODEL` in your `.env` file."
                )
            if isinstance(exc, openai.BadRequestError):
                logger.error("OpenAI rejected the request: %s", exc)
                return LLMResponseError(
                    "OpenAI rejected the request. The configured model may not "
                    "support structured output."
                )
            if isinstance(exc, openai.APIConnectionError):
                logger.warning("Could not reach OpenAI: %s", exc)
                return LLMConnectionError(
                    "Could not reach OpenAI. Check your network connection."
                )
            if isinstance(exc, openai.InternalServerError):
                logger.warning("OpenAI server error: %s", exc)
                return LLMResponseError(
                    "OpenAI reported a server error. Try again shortly."
                )
            if isinstance(exc, openai.APIStatusError):
                logger.error("OpenAI API error (%s): %s",
                             getattr(exc, "status_code", "?"), exc)
                return LLMResponseError(
                    f"OpenAI returned an error "
                    f"({getattr(exc, 'status_code', 'unknown')})."
                )

        logger.exception("Unexpected LLM failure (%s)", name)
        return LLMError(f"The AI request failed unexpectedly ({name}).")


# --------------------------------------------------------------------------- #
# Shared instance
# --------------------------------------------------------------------------- #

_service: LLMService | None = None


def get_llm_service(settings: Settings | None = None) -> LLMService:
    """The process-wide service, created on first use."""
    global _service
    if _service is None or settings is not None:
        _service = LLMService(settings)
    return _service


def reset_llm_service() -> None:
    """Drop the shared instance (used by tests)."""
    global _service
    _service = None


__all__ = [
    "LLMAuthError",
    "LLMConnectionError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMService",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "LLMUsage",
    "NO_KEY_MESSAGE",
    "StructuredResponse",
    "get_llm_service",
    "reset_llm_service",
]
