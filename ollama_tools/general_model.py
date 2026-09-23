"""Which general-purpose (chat/reasoning) model to reach for, given the
VRAM attached right now.

``GENERAL_TIERS`` is a fixed table, not a live lookup: the boundaries were
derived once from models actually measured on this machine (see
docs/model-picker.md), not from advertised parameter counts.
``qwen3.8-216k`` always reserves its full 216k KV cache and needs ~18-19 GB
total, so it only belongs at the top tier -- a single eGPU is not enough to
promote it. Re-measuring a model (a new build, a different quant) means
updating this table by hand; nothing here reads bench results at runtime.
"""

from __future__ import annotations

from .gpu import CONSERVATIVE, GIB, Gpu, GpuUnavailable, budget_bytes, query_gpus

# (minimum budget in bytes, model), highest tier first. The first one the
# budget clears wins.
GENERAL_TIERS: tuple[tuple[int, str], ...] = (
    (18 * GIB, "qwen3.8-216k"),
    (7 * GIB, "qwen3.5:9b"),
)
GENERAL_FALLBACK = "gemma3:4b"


def general_model_for_budget(budget: int) -> str:
    """Pick a tier by VRAM budget alone, with no GPU query involved."""
    for minimum, model in GENERAL_TIERS:
        if budget >= minimum:
            return model
    return GENERAL_FALLBACK


def get_general_model(strategy: str = CONSERVATIVE, gpus: list[Gpu] | None = None) -> str:
    """Best general-purpose model for the VRAM attached right now.

    Unlike ``fit.judge``, this never refuses: no GPU detected (no
    ``nvidia-smi``, or it reports nothing) falls back to the smallest
    model rather than raising. ``GpuUnavailable`` there means "do not load
    anything sized against an unknown budget"; picking a name worth trying
    is a lower-stakes question that still has a sane answer.
    """
    if gpus is None:
        try:
            gpus = query_gpus()
        except GpuUnavailable:
            return GENERAL_FALLBACK
    if not gpus:
        return GENERAL_FALLBACK
    return general_model_for_budget(budget_bytes(gpus, strategy))
