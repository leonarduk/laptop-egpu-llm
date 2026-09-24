"""Which coder model to reach for, given the VRAM attached right now.

``CODER_TIERS`` is a fixed table, not a live lookup: the boundaries were
derived once from models actually measured on this machine (see
docs/model-picker.md), not from advertised parameter counts, and are in
GiB. The top two tiers are the qwen3.8 builds, which reserve their whole KV
cache up front: ``qwen3.8-216k`` measured 19.29 GB total at 100% GPU and
``qwen3.8-100k`` ~15 GB, so the conservative budget of an asymmetric
8 + 16 GB pair (2 x the smaller card) reaches the 100k build but not the
216k one. ``qwen2.5-coder:32b`` is no longer on the machine and is no
longer in the table. Re-measuring a model (a new build, a different quant)
means updating this table by hand; nothing here reads bench results at
runtime.
"""

from __future__ import annotations

from .gpu import CONSERVATIVE, GIB, Gpu, GpuUnavailable, budget_bytes, query_gpus

# (minimum budget in bytes, model), highest tier first. The first one the
# budget clears wins.
CODER_TIERS: tuple[tuple[int, str], ...] = (
    (18 * GIB, "qwen3.8-216k"),
    (14 * GIB, "qwen3.8-100k"),
    (10 * GIB, "qwen2.5-coder:14b"),
    (7 * GIB, "qwen2.5-coder:7b"),
    (3 * GIB, "qwen2.5-coder:1.5b"),
)
CODER_FALLBACK = "qwen2.5-coder:0.5b"


def coder_model_for_budget(budget: int) -> str:
    """Pick a tier by VRAM budget alone, with no GPU query involved."""
    for minimum, model in CODER_TIERS:
        if budget >= minimum:
            return model
    return CODER_FALLBACK


def get_coder_model(strategy: str = CONSERVATIVE, gpus: list[Gpu] | None = None) -> str:
    """Best coder model for the VRAM attached right now.

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
            return CODER_FALLBACK
    if not gpus:
        return CODER_FALLBACK
    return coder_model_for_budget(budget_bytes(gpus, strategy))
