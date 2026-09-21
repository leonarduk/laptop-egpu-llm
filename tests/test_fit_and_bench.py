import pytest

from ollama_tools.bench import is_embedding_model, run_embedding, run_generation
from ollama_tools.fit import judge, required_bytes
from ollama_tools.gpu import GIB

SEVEN_B = int(4.36 * GIB)
TWENTY_SEVEN_B = int(11.29 * GIB)
INTERNAL_ONLY = int(7.55 * GIB)


def test_headroom_is_applied():
    assert required_bytes(1000, 20) == 1200
    assert required_bytes(1000, 0) == 1000


def test_negative_headroom_is_rejected():
    with pytest.raises(ValueError):
        required_bytes(1000, -10)


def test_seven_b_fits_the_internal_card():
    verdict = judge("qwen2.5-coder:7b", SEVEN_B, INTERNAL_ONLY, 20)
    assert verdict.fits
    assert verdict.short_bytes == 0


def test_the_27b_does_not_and_reports_the_shortfall():
    verdict = judge("qwen3.8-64k:latest", TWENTY_SEVEN_B, INTERNAL_ONLY, 20)
    assert not verdict.fits
    assert verdict.short_gib == pytest.approx(11.29 * 1.2 - 7.55, abs=0.05)


def test_wider_headroom_can_flip_a_marginal_model():
    """The 27B against the even-split ceiling with both cards attached.

    The ceiling is built from *free* memory, not nominal totals: 2 x 7.55
    GB free, not 2 x 7.93 GB installed. At ordinary context (20%) the 27B
    fits; at long context (40%, where the KV cache is real) it does not.
    That gap is the whole reason headroom is a parameter, and why the 27B
    at 64k wants a proportional split rather than an even one.
    """
    even_split_ceiling = int(2 * 7.55 * GIB)
    assert judge("m", TWENTY_SEVEN_B, even_split_ceiling, 20).fits
    assert not judge("m", TWENTY_SEVEN_B, even_split_ceiling, 40).fits


def test_exactly_equal_fits():
    assert judge("m", 1000, 1200, 20).fits


class FakeClient:
    """Records the call and returns a canned Ollama response."""

    def __init__(self, payload=None):
        self.payload = payload or {}
        self.calls = []

    def generate(self, model, prompt, num_predict, timeout=600):
        self.calls.append(("generate", model))
        return self.payload

    def embed(self, model, text, timeout=600):
        self.calls.append(("embed", model))
        return {"embeddings": [[0.1] * 768]}


def test_generation_rates_come_from_ollamas_own_timings():
    """eval_duration is nanoseconds: 200 tokens in 4s is 50 tok/s."""
    client = FakeClient(
        {
            "eval_count": 200,
            "eval_duration": 4_000_000_000,
            "prompt_eval_count": 30,
            "prompt_eval_duration": 100_000_000,
            "total_duration": 4_500_000_000,
        }
    )
    run = run_generation(client, "m", "p", 200)
    assert run.generation_tokens_per_second == pytest.approx(50.0)
    assert run.prompt_tokens_per_second == pytest.approx(300.0)
    assert run.total_seconds == pytest.approx(4.5)


def test_zero_duration_does_not_divide_by_zero():
    client = FakeClient({"eval_count": 0, "eval_duration": 0})
    assert run_generation(client, "m", "p", 1).generation_tokens_per_second == 0.0


def test_embedding_run_reports_dimensions():
    client = FakeClient()
    run = run_embedding(client, "nomic-embed-text", "text")
    assert run.dimensions == 768
    assert client.calls == [("embed", "nomic-embed-text")]


def test_embedding_models_are_detected_by_capability():
    """Asked, not inferred from the name: an embedding model has no
    generate endpoint at all, and name-matching is wrong both ways."""
    assert is_embedding_model(["embedding"])
    assert not is_embedding_model(["completion", "tools", "insert"])
    # A model that can do both is not embedding-only; generate still works.
    assert not is_embedding_model(["completion", "embedding"])
    assert not is_embedding_model([])
