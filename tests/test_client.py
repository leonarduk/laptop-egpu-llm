"""`_request` is the single place transport and parsing failures become
`OllamaUnavailable`, which the CLI maps to exit code 2 -- the documented
contract. The other tests use a fake client that raises that exception
directly, so they bypass the translation entirely; these drive it.
"""

import io
import json
import urllib.error

import pytest

from ollama_tools.client import (
    LoadedModel,
    ModelInfo,
    OllamaClient,
    OllamaUnavailable,
    normalise_model_name,
)


class FakeResponse(io.BytesIO):
    """Enough of an http.client.HTTPResponse for urlopen's context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def stub_urlopen(monkeypatch, result):
    """Replace the transport. `result` is either an exception to raise or
    the body bytes to return."""
    calls = []

    def fake(request, timeout=None):
        calls.append(request)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)

    monkeypatch.setattr("ollama_tools.client.urllib.request.urlopen", fake)
    return calls


def test_http_error_becomes_ollama_unavailable(monkeypatch):
    error = urllib.error.HTTPError(
        url="http://localhost:11434/api/tags",
        code=500,
        msg="Server Error",
        hdrs=None,
        fp=io.BytesIO(b"upstream exploded"),
    )
    stub_urlopen(monkeypatch, error)
    with pytest.raises(OllamaUnavailable) as caught:
        OllamaClient().list_models()
    # The status and the server's own words both survive: "500" alone does
    # not tell you which of several things went wrong.
    assert "500" in str(caught.value)
    assert "upstream exploded" in str(caught.value)


def test_url_error_becomes_ollama_unavailable(monkeypatch):
    """The ordinary case: the server is not running at all."""
    stub_urlopen(monkeypatch, urllib.error.URLError("connection refused"))
    with pytest.raises(OllamaUnavailable) as caught:
        OllamaClient().list_models()
    assert "unreachable" in str(caught.value)
    assert "connection refused" in str(caught.value)


def test_socket_error_becomes_ollama_unavailable(monkeypatch):
    """A timeout surfaces as OSError rather than URLError, and must not
    escape as a bare socket error."""
    stub_urlopen(monkeypatch, TimeoutError("timed out"))
    with pytest.raises(OllamaUnavailable):
        OllamaClient().list_models()


def test_unparsable_body_becomes_ollama_unavailable(monkeypatch):
    stub_urlopen(monkeypatch, b"not json at all")
    with pytest.raises(OllamaUnavailable) as caught:
        OllamaClient().list_models()
    assert "unparsable JSON" in str(caught.value)


def test_json_that_is_not_an_object_is_rejected(monkeypatch):
    """Valid JSON, wrong shape. Letting a list through would fail later at
    .get() with a much less obvious message."""
    stub_urlopen(monkeypatch, b'["a", "list"]')
    with pytest.raises(OllamaUnavailable) as caught:
        OllamaClient().list_models()
    assert "not an object" in str(caught.value)


def test_list_models_sorts_largest_first(monkeypatch):
    stub_urlopen(
        monkeypatch,
        json.dumps(
            {
                "models": [
                    {"name": "small", "size": 100},
                    {"name": "big", "size": 900},
                    {"name": "", "size": 500},
                ]
            }
        ).encode(),
    )
    models = OllamaClient().list_models()
    # The nameless entry is dropped: it cannot be matched against a request.
    assert [m.name for m in models] == ["big", "small"]
    assert isinstance(models[0], ModelInfo)


def test_loaded_models_carries_offload(monkeypatch):
    stub_urlopen(
        monkeypatch,
        json.dumps(
            {"models": [{"name": "m", "size": 1000, "size_vram": 800}]}
        ).encode(),
    )
    loaded = OllamaClient().loaded_models()
    assert isinstance(loaded[0], LoadedModel)
    assert loaded[0].offload_fraction == pytest.approx(0.8)


def test_offload_of_a_zero_sized_model_does_not_divide_by_zero():
    assert LoadedModel("m", 0, 0).offload_fraction == 0.0


def test_capabilities_missing_from_response_is_empty_not_an_error(monkeypatch):
    """An older Ollama may not report capabilities. Empty means "not known
    to be embedding-only", which keeps the generate path -- the safe
    default, since that is what every completion model uses."""
    stub_urlopen(monkeypatch, b"{}")
    assert OllamaClient().capabilities("m") == []


def test_endpoint_trailing_slash_does_not_double_up(monkeypatch):
    calls = stub_urlopen(monkeypatch, b'{"models": []}')
    OllamaClient("http://localhost:11434/").list_models()
    assert calls[0].full_url == "http://localhost:11434/api/tags"


@pytest.mark.parametrize(
    "given, expected",
    [
        ("qwen3.8-216k", "qwen3.8-216k:latest"),
        ("qwen2.5-coder:7b", "qwen2.5-coder:7b"),
        ("qwen3.8-216k:latest", "qwen3.8-216k:latest"),
        ("hf.co/unsloth/Qwen3-GGUF", "hf.co/unsloth/Qwen3-GGUF:latest"),
        # The colon here is a port, not a tag.
        ("registry.local:5000/team/model", "registry.local:5000/team/model:latest"),
        ("registry.local:5000/team/model:q4", "registry.local:5000/team/model:q4"),
    ],
)
def test_normalise_model_name_adds_latest_only_when_untagged(given, expected):
    assert normalise_model_name(given) == expected


def test_show_posts_the_model_name(monkeypatch):
    calls = stub_urlopen(monkeypatch, b'{"model_info": {"general.architecture": "qwen35"}}')
    payload = OllamaClient().show("qwen3.8-100k:latest")
    assert payload["model_info"]["general.architecture"] == "qwen35"
    assert calls[0].full_url == "http://localhost:11434/api/show"
    assert json.loads(calls[0].data) == {"model": "qwen3.8-100k:latest"}
