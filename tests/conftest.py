"""Provider SDK stubs, for machines that do not have them installed.

The unit suite is meant to run with no API keys, no network and no model
download. Several modules import a provider SDK at module level purely to build
a client they never reach in these tests, so the import is the only obstacle.

Two rules, both learned the hard way:

**Only stub what is missing.** CI installs the real packages, and a stub left
in `sys.modules` is what the next test module imports. An earlier version of
this stubbed unconditionally, which worked on a laptop and would have handed
`tests/test_selector.py` a fake `SentenceTransformer` in the one environment
where the real one exists.

**A stub is not a fixture.** Nothing here replaces behaviour under test. Tests
that need a provider to do something substitute their own fake at the call
site; these only make `import` succeed.
"""
from __future__ import annotations

import importlib.util
import sys
import types


def _missing(name: str) -> bool:
    if name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if _missing("groq"):
    _stub("groq", Groq=object)

if _missing("tavily"):
    _stub("tavily", TavilyClient=object)

if _missing("trafilatura"):
    _stub("trafilatura")

if _missing("tiktoken"):
    _stub("tiktoken")

if _missing("sentence_transformers"):
    _stub("sentence_transformers", SentenceTransformer=object)

if _missing("google.genai"):
    class _Part:
        @staticmethod
        def from_text(text: str):
            return {"text": text}

    gtypes = _stub(
        "google.genai.types",
        Content=lambda role, parts: {"role": role, "parts": parts},
        Part=_Part,
        GenerateContentConfig=lambda **kw: kw,
    )
    genai = _stub("google.genai", Client=object, types=gtypes)
    google = sys.modules.get("google") or _stub("google")
    google.genai = genai                                   # type: ignore[attr-defined]
