"""Tests for the provider fallback in llm/client.py.

This file exists because of a real incident and a real near-miss.

The incident: the provider retired the model being called, every request 404'd,
and the client fell through to the backup without anything visible changing.
The smoke test still reported success, because it asked for so few tokens that
a reasoning model spent the budget thinking and returned an empty string, which
is not an error.

The near-miss: the evaluation judge must never fall back, because the fallback
is the model under evaluation, so a rate-limited judge would silently turn
cross-model judging into self-grading. That is a single-element `order`, and
nothing enforced it.

Both are behaviours of this module and neither had a test. The clients are
replaced with fakes here, so nothing needs a key or a network.
"""
from __future__ import annotations

import importlib.util
import logging
import pathlib
import types

import pytest

# Loaded by path rather than `from llm import client`, so another test module
# swapping something in sys.modules cannot hand this one a different file.
_spec = importlib.util.spec_from_file_location(
    "_llm_client_under_test",
    pathlib.Path(__file__).resolve().parent.parent / "llm" / "client.py",
)
assert _spec and _spec.loader
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


# --------------------------------------------------------------- fake clients

class _Choice:
    def __init__(self, content): self.message = types.SimpleNamespace(content=content)


class FakeGroq:
    """Returns `text`, or raises `error` on every call. Counts its calls."""

    def __init__(self, text: str = "groq says so", error: Exception | None = None):
        self.text, self.error, self.calls = text, error, 0
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls += 1
                outer.last_kwargs = kw
                if outer.error:
                    raise outer.error
                return types.SimpleNamespace(choices=[_Choice(outer.text)])

        self.chat = types.SimpleNamespace(completions=_Completions())


class FakeGemini:
    def __init__(self, text: str = "gemini says so", error: Exception | None = None):
        self.text, self.error, self.calls = text, error, 0
        outer = self

        class _Models:
            def generate_content(self, **kw):
                outer.calls += 1
                outer.last_kwargs = kw
                if outer.error:
                    raise outer.error
                return types.SimpleNamespace(text=outer.text)

        self.models = _Models()


@pytest.fixture()
def providers(monkeypatch):
    """Both clients replaced, and the retry sleep removed.

    The half-second sleep between providers is deliberate in production and
    pure cost in a test; leaving it in would make the failure paths the slowest
    tests in the suite and quietly discourage adding more of them.
    """
    groq, gemini = FakeGroq(), FakeGemini()
    monkeypatch.setattr(client, "_groq_client", lambda: groq)
    monkeypatch.setattr(client, "_gemini_client", lambda: gemini)
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)
    return groq, gemini


# ------------------------------------------------------------------- complete

def test_primary_answers_and_the_fallback_is_never_called(providers):
    groq, gemini = providers
    text, used = client.complete(MSGS)
    assert (text, used) == ("groq says so", "groq")
    assert (groq.calls, gemini.calls) == (1, 0)


def test_falls_back_when_the_primary_raises(providers):
    groq, gemini = providers
    groq.error = RuntimeError("404 model_not_found")
    text, used = client.complete(MSGS)
    assert (text, used) == ("gemini says so", "gemini")
    assert gemini.calls == 1


def test_a_fallback_is_logged_loudly_enough_to_notice(providers, caplog):
    # The incident: a retired model meant every request fell through to the
    # backup and nothing visible changed. A fallback that leaves no trace is
    # indistinguishable from a system that is working.
    groq, _ = providers
    groq.error = RuntimeError("404 model_not_found")
    with caplog.at_level(logging.WARNING):
        client.complete(MSGS)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("fallback" in w for w in warnings)
    assert any("groq" in w for w in warnings)


def test_all_providers_failing_raises_with_the_last_error(providers):
    groq, gemini = providers
    groq.error = RuntimeError("groq down")
    gemini.error = RuntimeError("gemini down")
    with pytest.raises(client.LLMError, match="gemini down"):
        client.complete(MSGS)


def test_prefer_gemini_reverses_the_order(providers):
    groq, gemini = providers
    text, used = client.complete(MSGS, prefer="gemini")
    assert used == "gemini"
    assert groq.calls == 0


# ---------------------------------------------------- the judge must not fall back

def test_a_single_element_order_never_reaches_the_other_provider(providers):
    """The property the evaluation judge depends on.

    The judge runs on Gemini so it does not grade the model that produced the
    answer. If a rate-limited judge fell back to Groq, the run would still
    produce scores and nothing in the output would say they were self-graded.
    A missing score is recoverable; a plausible wrong one is not.
    """
    groq, gemini = providers
    gemini.error = RuntimeError("429 RESOURCE_EXHAUSTED")
    with pytest.raises(client.LLMError):
        client.complete(MSGS, order=["gemini"])
    assert groq.calls == 0, "fell back to the model under evaluation"


def test_an_explicit_order_overrides_prefer(providers):
    groq, gemini = providers
    client.complete(MSGS, prefer="groq", order=["gemini"])
    assert (groq.calls, gemini.calls) == (0, 1)


# -------------------------------------------------------------- empty responses

def test_an_empty_response_is_returned_rather_than_raised(providers):
    # It is not an error, and treating it as one would hide that the request
    # succeeded. What it must not be is silent.
    groq, gemini = providers
    groq.text = ""
    text, used = client.complete(MSGS)
    assert (text, used) == ("", "groq")
    assert gemini.calls == 0


def test_an_empty_response_is_logged_with_the_budget_that_produced_it(providers, caplog):
    # A reasoning model spends part of max_tokens thinking before it emits any
    # text, so too small a budget returns an empty string with no error at all.
    # The budget is the diagnosis, so it belongs in the message.
    groq, _ = providers
    groq.text = "   "
    with caplog.at_level(logging.WARNING):
        client.complete(MSGS, max_tokens=10)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("empty" in w for w in warnings)
    assert any("10" in w for w in warnings)


def test_whitespace_counts_as_empty_but_is_returned_verbatim(providers):
    # The emptiness check strips before testing, so spaces trigger the warning.
    # The caller still gets exactly what the provider sent: trimming here would
    # erase the difference between "no answer" and "an answer of spaces".
    groq, _ = providers
    groq.text = "\n\n  "
    text, _used = client.complete(MSGS)
    assert text == "\n\n  "


def _role(content):
    """The Gemini SDK returns typed objects; the stub above returns dicts.

    Both shapes have to work, because CI has the real SDK installed and a
    machine without it does not, and a test that only passes on one of them is
    a test that will fail somewhere nobody is looking.
    """
    return getattr(content, "role", None) or content["role"]


# --------------------------------------------------------------- what is sent

def test_the_token_budget_and_temperature_reach_the_provider(providers):
    groq, _ = providers
    client.complete(MSGS, temperature=0.0, max_tokens=600)
    assert groq.last_kwargs["temperature"] == 0.0
    assert groq.last_kwargs["max_tokens"] == 600


def test_gemini_receives_the_system_prompt_separately_from_the_turns():
    # Gemini takes the system instruction as its own argument and only
    # user/model turns as contents, so a client that passed the system message
    # through as a turn would change what the model was told.
    system, contents = client._msgs_to_gemini(MSGS)
    assert system == "be brief"
    assert len(contents) == 1
    assert _role(contents[0]) == "user"


def test_an_assistant_turn_becomes_a_model_turn_for_gemini():
    _system, contents = client._msgs_to_gemini(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    )
    assert [_role(c) for c in contents] == ["user", "model"]
