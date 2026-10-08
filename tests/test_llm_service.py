"""Tests for the OpenAI service layer.

No test here touches the network. The SDK client is replaced with a stub that
either returns a canned completion or raises the SDK's own exception types, so
error translation is exercised against the real classes.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.services.llm_service import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMService,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUsage,
    NO_KEY_MESSAGE,
    get_llm_service,
    reset_llm_service,
)


class Answer(BaseModel):
    value: str
    count: int = 0


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

def _completion(
    parsed=None,
    *,
    content: str | None = None,
    refusal: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
    model: str = "gpt-4.1-mini",
):
    message = SimpleNamespace(parsed=parsed, content=content, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
        model=model,
    )


class StubClient:
    """Stands in for `openai.OpenAI`, recording the call it received."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=self._parse)
        )

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


def _sdk_error(cls, status: int = 400):
    """Build a real SDK exception without performing a request."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    if cls is openai.APITimeoutError:
        return cls(request=request)
    if cls is openai.APIConnectionError:
        return cls(message="connection failed", request=request)
    response = httpx.Response(status, request=request, json={"error": {}})
    return cls(message="boom", response=response, body=None)


@pytest.fixture
def keyed_settings() -> Settings:
    return Settings(openai_api_key="sk-test-key", openai_model="gpt-4.1-mini")


@pytest.fixture
def unkeyed_settings() -> Settings:
    return Settings(openai_api_key="")


@pytest.fixture(autouse=True)
def _clean_shared_service():
    reset_llm_service()
    yield
    reset_llm_service()


# --------------------------------------------------------------------------- #
# Availability -- the app must work without a key
# --------------------------------------------------------------------------- #

def test_service_is_unavailable_without_a_key(unkeyed_settings: Settings):
    service = LLMService(unkeyed_settings)

    assert service.available is False
    assert service.unavailable_reason == NO_KEY_MESSAGE


def test_placeholder_key_counts_as_missing():
    service = LLMService(Settings(openai_api_key="your_api_key_here"))
    assert service.available is False


def test_constructing_the_service_without_a_key_does_not_raise(
    unkeyed_settings: Settings,
):
    # Importing and constructing must never break app startup.
    LLMService(unkeyed_settings)


def test_calling_without_a_key_raises_a_clear_error(unkeyed_settings: Settings):
    service = LLMService(unkeyed_settings)

    with pytest.raises(LLMUnavailableError, match="requires an OpenAI API key"):
        service.complete_structured(
            [{"role": "user", "content": "hi"}], Answer
        )


def test_service_is_available_with_a_key(keyed_settings: Settings):
    assert LLMService(keyed_settings).available is True
    assert LLMService(keyed_settings).unavailable_reason is None


def test_injected_client_bypasses_the_key_check(unkeyed_settings: Settings):
    service = LLMService(unkeyed_settings, client=StubClient())
    assert service.available is True
    assert service.unavailable_reason is None


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #

def test_parsed_response_is_returned(keyed_settings: Settings):
    client = StubClient(_completion(parsed=Answer(value="ok", count=3)))
    service = LLMService(keyed_settings, client=client)

    response = service.complete_structured(
        [{"role": "user", "content": "q"}], Answer
    )

    assert isinstance(response.data, Answer)
    assert response.data.value == "ok"
    assert response.usage.total_tokens == 150


def test_request_uses_the_configured_model_and_limits(keyed_settings: Settings):
    client = StubClient(_completion(parsed=Answer(value="ok")))
    service = LLMService(keyed_settings, client=client)

    service.complete_structured([{"role": "user", "content": "q"}], Answer)
    call = client.calls[0]

    assert call["model"] == "gpt-4.1-mini"
    assert call["response_format"] is Answer
    assert call["temperature"] == keyed_settings.llm_temperature
    assert call["max_completion_tokens"] == keyed_settings.llm_max_output_tokens


def test_temperature_can_be_overridden(keyed_settings: Settings):
    client = StubClient(_completion(parsed=Answer(value="ok")))
    LLMService(keyed_settings, client=client).complete_structured(
        [{"role": "user", "content": "q"}], Answer, temperature=0.9
    )
    assert client.calls[0]["temperature"] == 0.9


def test_raw_json_is_validated_when_parsed_is_absent(keyed_settings: Settings):
    client = StubClient(_completion(content='{"value": "from json", "count": 7}'))
    service = LLMService(keyed_settings, client=client)

    response = service.complete_structured(
        [{"role": "user", "content": "q"}], Answer
    )

    assert response.data.value == "from json"
    assert response.data.count == 7


def test_malformed_json_raises_a_response_error(keyed_settings: Settings):
    client = StubClient(_completion(content="not json at all"))
    service = LLMService(keyed_settings, client=client)

    # Pydantic reports unparseable JSON as a validation failure, so this and a
    # schema mismatch surface the same user-facing message.
    with pytest.raises(LLMResponseError, match="did not match the expected structure"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_json_failing_the_schema_raises_a_response_error(keyed_settings: Settings):
    client = StubClient(_completion(content='{"count": "not a number"}'))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMResponseError, match="did not match the expected structure"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_refusal_raises_a_response_error(keyed_settings: Settings):
    client = StubClient(_completion(refusal="I cannot help with that."))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMResponseError, match="declined this request"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_truncated_response_raises_a_response_error(keyed_settings: Settings):
    client = StubClient(_completion(parsed=None, finish_reason="length"))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMResponseError, match="cut short"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_content_filter_raises_a_response_error(keyed_settings: Settings):
    client = StubClient(_completion(finish_reason="content_filter"))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMResponseError, match="content filter"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_empty_choices_raises_a_response_error(keyed_settings: Settings):
    client = StubClient(SimpleNamespace(choices=[], usage=None, model="m"))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMResponseError, match="empty response"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_no_messages_raises(keyed_settings: Settings):
    service = LLMService(keyed_settings, client=StubClient())

    with pytest.raises(LLMResponseError, match="No prompt"):
        service.complete_structured([], Answer)


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("sdk_error", "expected", "fragment"),
    [
        (openai.APITimeoutError, LLMTimeoutError, "timed out"),
        (openai.RateLimitError, LLMRateLimitError, "rate limit"),
        (openai.AuthenticationError, LLMAuthError, "rejected the API key"),
        (openai.PermissionDeniedError, LLMAuthError, "cannot use the model"),
        (openai.NotFoundError, LLMResponseError, "was not found"),
        (openai.BadRequestError, LLMResponseError, "rejected the request"),
        (openai.APIConnectionError, LLMConnectionError, "Could not reach OpenAI"),
        (openai.InternalServerError, LLMResponseError, "server error"),
    ],
)
def test_sdk_errors_become_domain_errors(
    keyed_settings: Settings, sdk_error, expected, fragment
):
    client = StubClient(error=_sdk_error(sdk_error))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(expected, match=fragment):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_timeout_and_rate_limit_are_marked_transient(keyed_settings: Settings):
    assert LLMTimeoutError.transient is True
    assert LLMRateLimitError.transient is True
    assert LLMAuthError.transient is False


def test_unknown_exceptions_become_a_generic_llm_error(keyed_settings: Settings):
    client = StubClient(error=RuntimeError("something odd"))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMError, match="failed unexpectedly"):
        service.complete_structured([{"role": "user", "content": "q"}], Answer)


def test_errors_never_leak_a_stack_trace(keyed_settings: Settings):
    client = StubClient(error=_sdk_error(openai.RateLimitError))
    service = LLMService(keyed_settings, client=client)

    with pytest.raises(LLMError) as caught:
        service.complete_structured([{"role": "user", "content": "q"}], Answer)

    message = str(caught.value)
    # Naming OpenAI is useful to the user; leaking internals is not.
    assert "Traceback" not in message
    assert "RateLimitError" not in message
    assert "httpx" not in message
    assert ".py" not in message


# --------------------------------------------------------------------------- #
# Usage and the shared instance
# --------------------------------------------------------------------------- #

def test_usage_totals():
    usage = LLMUsage(model="m", prompt_tokens=100, completion_tokens=25)
    assert usage.total_tokens == 125
    assert usage.to_dict()["total_tokens"] == 125


def test_missing_usage_block_is_tolerated(keyed_settings: Settings):
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=Answer(value="x"), content=None,
                                        refusal=None),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="gpt-4.1-mini",
    )
    service = LLMService(keyed_settings, client=StubClient(completion))

    response = service.complete_structured([{"role": "user", "content": "q"}], Answer)
    assert response.usage.total_tokens == 0


def test_shared_service_is_cached():
    first = get_llm_service()
    assert get_llm_service() is first
    reset_llm_service()
    assert get_llm_service() is not first


def test_shared_service_is_replaced_when_settings_are_passed(
    keyed_settings: Settings,
):
    first = get_llm_service()
    second = get_llm_service(keyed_settings)
    assert second is not first
    assert second.settings is keyed_settings
