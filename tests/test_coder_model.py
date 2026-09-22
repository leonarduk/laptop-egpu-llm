import pytest

from ollama_tools.coder_model import (
    CODER_FALLBACK,
    coder_model_for_budget,
    get_coder_model,
)
from ollama_tools.gpu import CONSERVATIVE, GIB, Gpu, GpuUnavailable

MIB = 1024 * 1024

# Same machine shapes as test_gpu.py: 8 GB internal, 16 GB eGPU.
INTERNAL_8GB = Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB)
EGPU_16GB = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 16000 * MIB)
THREE_GB = Gpu(0, "some 3 GB card", 3 * 1024 * MIB, 3 * 1024 * MIB)
ONE_GB = Gpu(0, "tiny card", 1024 * MIB, 1024 * MIB)


def test_both_egpus_pick_qwen3():
    budget = 18 * GIB
    assert coder_model_for_budget(budget) == "qwen3.8-216k"


def test_just_below_qwen3_tier_falls_to_7b():
    """18 GiB matches the docstring's "~18-19 GB total" for qwen3.8-216k;
    one byte under that must not promote it."""
    assert coder_model_for_budget(18 * GIB - 1) == "qwen2.5-coder:7b"


def test_single_8gb_card_picks_7b():
    assert coder_model_for_budget(int(7.7 * GIB)) == "qwen2.5-coder:7b"


def test_just_below_7b_tier_falls_to_1_5b():
    assert coder_model_for_budget(7 * GIB - 1) == "qwen2.5-coder:1.5b"


def test_3gb_picks_1_5b():
    assert coder_model_for_budget(3 * GIB) == "qwen2.5-coder:1.5b"


def test_just_below_1_5b_tier_falls_back():
    assert coder_model_for_budget(3 * GIB - 1) == CODER_FALLBACK


def test_below_3gb_falls_back_to_0_5b():
    assert coder_model_for_budget(2 * GIB) == CODER_FALLBACK


def test_get_coder_model_uses_conservative_budget_of_both_cards():
    model = get_coder_model(CONSERVATIVE, [INTERNAL_8GB, EGPU_16GB])
    # conservative caps an asymmetric pair at 2x the smaller card, so this
    # stays below the 18 GB qwen3.8-216k tier even though free VRAM sums to
    # ~23 GB.
    assert model == "qwen2.5-coder:7b"


def test_get_coder_model_single_8gb_card():
    assert get_coder_model(CONSERVATIVE, [INTERNAL_8GB]) == "qwen2.5-coder:7b"


def test_get_coder_model_3gb_card():
    assert get_coder_model(CONSERVATIVE, [THREE_GB]) == "qwen2.5-coder:1.5b"


def test_get_coder_model_tiny_card_falls_back():
    assert get_coder_model(CONSERVATIVE, [ONE_GB]) == CODER_FALLBACK


def test_get_coder_model_no_gpus_falls_back():
    assert get_coder_model(CONSERVATIVE, []) == CODER_FALLBACK


def test_get_coder_model_queries_when_no_gpus_given(monkeypatch):
    import ollama_tools.coder_model as coder_model_mod

    monkeypatch.setattr(coder_model_mod, "query_gpus", lambda: [INTERNAL_8GB])
    assert get_coder_model() == "qwen2.5-coder:7b"


def test_get_coder_model_falls_back_when_gpu_unavailable(monkeypatch):
    import ollama_tools.coder_model as coder_model_mod

    def raise_unavailable():
        raise GpuUnavailable("no nvidia-smi")

    monkeypatch.setattr(coder_model_mod, "query_gpus", raise_unavailable)
    assert get_coder_model() == CODER_FALLBACK
