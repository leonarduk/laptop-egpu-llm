import pytest

from ollama_tools.general_model import (
    GENERAL_FALLBACK,
    GENERAL_TIERS,
    general_model_for_budget,
    get_general_model,
)
from ollama_tools.gpu import CONSERVATIVE, GIB, PROPORTIONAL, Gpu, GpuUnavailable

MIB = 1024 * 1024

# Same machine shapes as test_gpu.py: 8 GB internal, 16 GB eGPU. Free
# figures are what nvidia-smi reports with nothing loaded.
INTERNAL_8GB = Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 7700 * MIB)
EGPU_16GB = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 15600 * MIB)
THREE_GB = Gpu(0, "some 3 GB card", 3 * 1024 * MIB, 3 * 1024 * MIB)
ONE_GB = Gpu(0, "tiny card", 1024 * MIB, 1024 * MIB)


def test_tiers_are_ordered_highest_first():
    """First match wins, so an out-of-order row would shadow the ones
    below it."""
    minimums = [minimum for minimum, _ in GENERAL_TIERS]
    assert minimums == sorted(minimums, reverse=True)


@pytest.mark.parametrize(
    "budget, model",
    [
        (18 * GIB, "qwen3.8-216k"),
        (18 * GIB - 1, "qwen3.8-100k"),
        (14 * GIB, "qwen3.8-100k"),
        (14 * GIB - 1, "qwen3.5:9b"),
        (7 * GIB, "qwen3.5:9b"),
        (7 * GIB - 1, GENERAL_FALLBACK),
    ],
)
def test_tier_boundaries(budget, model):
    """Each boundary exactly, and one byte under it."""
    assert general_model_for_budget(budget) == model


def test_single_8gb_card_picks_9b():
    assert general_model_for_budget(int(7.7 * GIB)) == "qwen3.5:9b"


def test_below_3gb_falls_back():
    assert general_model_for_budget(2 * GIB) == GENERAL_FALLBACK


def test_real_pair_conservative_picks_100k():
    """Conservative caps the asymmetric pair at 2 x 7700 MiB (~15.0 GiB):
    the 100k build (~15 GB measured) but not the 216k one (19.29 GB)."""
    assert get_general_model(CONSERVATIVE, [INTERNAL_8GB, EGPU_16GB]) == "qwen3.8-100k"


def test_real_pair_proportional_picks_216k():
    """7700 + 15600 MiB (~22.8 GiB) clears the 18 GiB top tier."""
    assert get_general_model(PROPORTIONAL, [INTERNAL_8GB, EGPU_16GB]) == "qwen3.8-216k"


def test_default_strategy_is_still_conservative():
    """A safety choice, not an oversight: the default must not assume the
    runtime splits layers proportionally."""
    assert get_general_model(gpus=[INTERNAL_8GB, EGPU_16GB]) == "qwen3.8-100k"


def test_get_general_model_single_8gb_card():
    assert get_general_model(CONSERVATIVE, [INTERNAL_8GB]) == "qwen3.5:9b"


def test_get_general_model_3gb_card_falls_back():
    assert get_general_model(CONSERVATIVE, [THREE_GB]) == GENERAL_FALLBACK


def test_get_general_model_tiny_card_falls_back():
    assert get_general_model(CONSERVATIVE, [ONE_GB]) == GENERAL_FALLBACK


def test_get_general_model_no_gpus_falls_back():
    assert get_general_model(CONSERVATIVE, []) == GENERAL_FALLBACK


def test_get_general_model_queries_when_no_gpus_given(monkeypatch):
    import ollama_tools.general_model as general_model_mod

    monkeypatch.setattr(general_model_mod, "query_gpus", lambda: [INTERNAL_8GB])
    monkeypatch.setattr(general_model_mod, "resident_vram_bytes", lambda: 0)
    monkeypatch.setattr(general_model_mod, "installed_models", lambda: None)
    assert get_general_model() == "qwen3.5:9b"


def test_get_general_model_falls_back_when_gpu_unavailable(monkeypatch):
    import ollama_tools.general_model as general_model_mod

    def raise_unavailable():
        raise GpuUnavailable("no nvidia-smi")

    monkeypatch.setattr(general_model_mod, "query_gpus", raise_unavailable)
    assert get_general_model() == GENERAL_FALLBACK
