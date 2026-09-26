"""What the model pickers need to know about Ollama itself, not just the GPU.

``nvidia-smi``'s free figure alone undercounts the budget whenever Ollama is
already holding a model: that model's VRAM shows as used, yet Ollama evicts
it to load the next one. Picking against free VRAM alone therefore downgrades
the pick every time the previous stage's model is still resident -- the very
next call after a qwen3.8 load sees ~6 GiB and asks for a 1.5b that may not
even be pulled.

Everything here degrades to "no extra information" rather than raising: an
unreachable server means no VRAM is added back and no tier is skipped, which
is exactly how the pickers behaved before this module existed.
"""

from __future__ import annotations

import os
from dataclasses import replace

from .client import DEFAULT_ENDPOINT, OllamaClient, OllamaUnavailable, normalise_model_name
from .gpu import Gpu


def _endpoint(endpoint: str | None) -> str:
    return endpoint or os.environ.get("OLLAMA_ENDPOINT", "").strip() or DEFAULT_ENDPOINT


def resident_vram_bytes(endpoint: str | None = None) -> int:
    """VRAM held by models Ollama has resident right now, or 0 if unknown."""
    try:
        loaded = OllamaClient(_endpoint(endpoint)).loaded_models()
    except OllamaUnavailable:
        return 0
    return sum(max(m.size_vram_bytes, 0) for m in loaded)


def installed_models(endpoint: str | None = None) -> frozenset[str] | None:
    """Normalised names of every pulled model, or ``None`` if unknown.

    ``None`` (not an empty set) when the server can't be asked, so a caller
    can tell "nothing is pulled" from "could not find out".
    """
    try:
        models = OllamaClient(_endpoint(endpoint)).list_models()
    except OllamaUnavailable:
        return None
    return frozenset(normalise_model_name(m.name) for m in models)


def reclaim_resident(gpus: list[Gpu], held_bytes: int) -> list[Gpu]:
    """``gpus`` with ``held_bytes`` of resident-model VRAM counted as free.

    ``/api/ps`` reports one ``size_vram`` per model, not its per-card split,
    so the held bytes are attributed to each card in proportion to what that
    card has in use, never more than it has in use (the rest of "used" is the
    desktop, other processes -- not Ollama's to give back). That keeps the
    per-card figures the ``even``/``conservative`` strategies depend on
    honest rather than piling everything onto one card.
    """
    if held_bytes <= 0 or not gpus:
        return list(gpus)
    used = [max(g.total_bytes - g.free_bytes, 0) for g in gpus]
    total_used = sum(used)
    if total_used <= 0:
        return list(gpus)
    held_bytes = min(held_bytes, total_used)
    return [
        replace(g, free_bytes=g.free_bytes + min(u, held_bytes * u // total_used))
        for g, u in zip(gpus, used)
    ]


def pick_tier(
    tiers: tuple[tuple[int, str], ...],
    fallback: str,
    budget: int,
    installed: frozenset[str] | None = None,
) -> str:
    """First tier the budget clears, skipping any model that isn't pulled.

    ``installed`` of ``None`` means "unknown": no tier is skipped. A tier
    the budget clears but that isn't pulled would only 404 at request time,
    so the next one down that *is* pulled is the better answer.
    """
    for minimum, model in tiers:
        if budget < minimum:
            continue
        if installed is None or normalise_model_name(model) in installed:
            return model
    return fallback
