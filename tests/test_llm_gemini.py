from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from fuel_agent.audit import AuditTrail
from fuel_agent.llm.gemini import GeminiProvider


class Overloaded:
    """Responde 503 nas primeiras `failures` chamadas e depois funciona."""

    def __init__(self, failures: int):
        self.failures = failures
        self.models_called = []

    def generate_content(self, *, model, contents, config):
        self.models_called.append(model)
        if len(self.models_called) <= self.failures:
            raise genai_errors.ServerError(503, {"error": {"message": "high demand"}})
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(role="model", parts=[types.Part.from_text(text="ok")]),
                    finish_reason=SimpleNamespace(name="STOP"),
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            model_version=model,
            response_id="r",
        )


def _provider(tmp_path, models):
    audit = AuditTrail(tmp_path)
    waits = []
    provider = GeminiProvider(
        "main",
        audit,
        api_key=None,
        max_tokens=100,
        fallback_model="lite",
        client=SimpleNamespace(models=models),
        retry_delays=(1.0, 2.0),
        sleep=waits.append,
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="oi")])]
    return provider, audit, contents, waits


def test_retries_same_model_before_falling_back(tmp_path):
    models = Overloaded(failures=2)
    provider, audit, contents, waits = _provider(tmp_path, models)
    provider.generate("teste", "sistema", contents, {})
    assert models.models_called == ["main", "main", "main"]
    assert waits == [1.0, 2.0]
    assert not [e for e in audit.read() if e["event"] == "llm_fallback"]


def test_falls_back_after_retries_and_keeps_fallback(tmp_path):
    models = Overloaded(failures=3)
    provider, audit, contents, _ = _provider(tmp_path, models)
    provider.generate("teste", "sistema", contents, {})
    provider.generate("teste", "sistema", contents, {})
    assert models.models_called == ["main", "main", "main", "lite", "lite"]
    assert [e["to_model"] for e in audit.read() if e["event"] == "llm_fallback"] == ["lite"]


def test_raises_when_everything_is_overloaded(tmp_path):
    provider, _, contents, _ = _provider(tmp_path, Overloaded(failures=10))
    with pytest.raises(genai_errors.ServerError):
        provider.generate("teste", "sistema", contents, {})
