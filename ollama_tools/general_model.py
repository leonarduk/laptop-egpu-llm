"""Which general-purpose (chat/reasoning) model to reach for, given the
VRAM attached right now.

``GENERAL_TIERS`` is a fixed table, not a live lookup: the boundaries were
derived once from models actually measured on this machine (see
docs/model-picker.md), not from advertised parameter counts, and are in
GiB. ``qwen3.8-216k`` always reserves its full 216k KV cache and measured
19.29 GB total at 100% GPU, so it only belongs at the top tier -- a single
eGPU is not enough to promote it. ``qwen3.8-100k`` is the same weights with
a 100k cache, ~15 GB total, which the conservative budget of an 8 + 16 GB
pair does reach. Re-measuring a model (a new build, a different quant) means
updating this table by hand; nothing here reads bench results at runtime.
"""

from __future__ import annotations

from .gpu import CONSERVATIVE, GIB, Gpu, GpuUnavailable, budget_bytes, query_gpus
from .live import installed_models, pick_tier, reclaim_resident, resident_vram_bytes

# (minimum budget in bytes, model), highest tier first. The first one the
# budget clears wins.
GENERAL_TIERS: tuple[tuple[int, str], ...] = (
    (18 * GIB, "qwen3.8-216k"),
    (14 * GIB, "qwen3.8-100k"),
    (7 * GIB, "qwen3.5:9b"),
)
GENERAL_FALLBACK = "gemma3:4b"


def general_model_for_budget(budget: int) -> str:
    """Pick a tier by VRAM budget alone, with no GPU query involved."""
    for minimum, model in GENERAL_TIERS:
        if budget >= minimum:
            return model
    return GENERAL_FALLBACK


def get_general_model(
    strategy: str = CONSERVATIVE,
    gpus: list[Gpu] | None = None,
    *,
    resident_bytes: int = 0,
    installed: frozenset[str] | None = None,
) -> str:
    """Best general-purpose model for the VRAM attached right now.

    Unlike ``fit.judge``, this never refuses: no GPU detected (no
    ``nvidia-smi``, or it reports nothing) falls back to the smallest
    model rather than raising. ``GpuUnavailable`` there means "do not load
    anything sized against an unknown budget"; picking a name worth trying
    is a lower-stakes question that still has a sane answer.

    VRAM held by models Ollama already has resident counts as available
    (``resident_bytes``): Ollama evicts them to load the pick, so treating
    that memory as spoken for downgraded every call made while the previous
    stage's model was still loaded. Tiers not in ``installed`` are skipped
    (``None`` = unknown, skip nothing). With ``gpus`` omitted all three are
    read live -- nvidia-smi, then Ollama's /api/ps and /api/tags.
    """
    if gpus is None:
        try:
            gpus = query_gpus()
        except GpuUnavailable:
            return GENERAL_FALLBACK
        resident_bytes = resident_vram_bytes()
        installed = installed_models()
    if not gpus:
        return GENERAL_FALLBACK
    budget = budget_bytes(reclaim_resident(gpus, resident_bytes), strategy)
    return pick_tier(GENERAL_TIERS, GENERAL_FALLBACK, budget, installed)
