"""Timing a model, whichever kind it is.

An embedding model has no ``generate`` endpoint -- Ollama answers
``"nomic-embed-text" does not support generate`` -- so the capability is
asked for first and the right endpoint used. Reporting one "speed" number
would hide which of two different things is the bottleneck, so generation
and prompt evaluation are kept apart: generation is memory-bandwidth-bound
and sequential, prompt evaluation is compute-bound and parallel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .client import OllamaClient

DEFAULT_PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted "
    "list, with a docstring."
)
DEFAULT_EMBED_TEXT = (
    "def merge(left, right):\n"
    "    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n"
    "    return sorted(left + right)\n"
)
NANOSECONDS = 1e9


@dataclass(frozen=True)
class GenerationRun:
    generated_tokens: int
    generation_tokens_per_second: float
    prompt_tokens_per_second: float
    total_seconds: float

    kind = "generate"


@dataclass(frozen=True)
class EmbedRun:
    dimensions: int
    total_seconds: float

    kind = "embed"


def _rate(count: int, duration_ns: int) -> float:
    if not duration_ns:
        return 0.0
    return count / (duration_ns / NANOSECONDS)


def run_generation(client: OllamaClient, model: str, prompt: str, num_predict: int) -> GenerationRun:
    """One timed completion, using Ollama's own timings rather than a
    stopwatch -- they separate prompt evaluation from generation, which a
    wall clock cannot."""
    response = client.generate(model, prompt, num_predict)
    return GenerationRun(
        generated_tokens=int(response.get("eval_count", 0)),
        generation_tokens_per_second=_rate(
            int(response.get("eval_count", 0)), int(response.get("eval_duration", 0))
        ),
        prompt_tokens_per_second=_rate(
            int(response.get("prompt_eval_count", 0)),
            int(response.get("prompt_eval_duration", 0)),
        ),
        total_seconds=int(response.get("total_duration", 0)) / NANOSECONDS,
    )


def run_embedding(client: OllamaClient, model: str, text: str) -> EmbedRun:
    """One timed embedding call.

    Timed with a wall clock because /api/embed reports no durations of its
    own. Fine for the purpose: an embedding call is a single forward pass,
    with no prompt-eval-versus-generation split to tease apart.
    """
    started = time.perf_counter()
    response = client.embed(model, text)
    elapsed = time.perf_counter() - started
    vectors = response.get("embeddings") or []
    dimensions = len(vectors[0]) if vectors and isinstance(vectors[0], list) else 0
    return EmbedRun(dimensions=dimensions, total_seconds=elapsed)


def is_embedding_model(capabilities: list[str]) -> bool:
    return "embedding" in capabilities and "completion" not in capabilities
