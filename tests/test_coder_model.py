import pytest

from ollama_tools.coder_model import (
    CODER_FALLBACK,
    CODER_TIERS,
    coder_model_for_budget,
    get_coder_model,
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
    minimums = [minimum for minimum, _ in CODER_TIERS]
    assert minimums == sorted(minimums, reverse=True)


def test_the_deleted_32b_is_never_picked():
    assert all("32b" not in model for _, model in CODER_TIERS)


@pytest.mark.parametrize(
    "budget, model",
    [
        (18 * GIB, "qwen3.8-216k"),
        (18 * GIB - 1, "qwen3.8-100k"),
        (14 * GIB, "qwen3.8-100k"),
        (14 * GIB - 1, "qwen2.5-coder:14b"),
        (10 * GIB, "qwen2.5-coder:14b"),
        (10 * GIB - 1, "qwen2.5-coder:7b"),
        (7 * GIB, "qwen2.5-coder:7b"),
        (7 * GIB - 1, "qwen2.5-coder:1.5b"),
        (3 * GIB, "qwen2.5-coder:1.5b"),
        (3 * GIB - 1, CODER_FALLBACK),
    ],
)
def test_tier_boundaries(budget, model):
    """Each boundary exactly, and one byte under it."""
    assert coder_model_for_budget(budget) == model


def test_single_8gb_card_picks_7b():
    assert coder_model_for_budget(int(7.7 * GIB)) == "qwen2.5-coder:7b"


def test_below_3gb_falls_back_to_0_5b():
    assert coder_model_for_budget(2 * GIB) == CODER_FALLBACK


def test_real_pair_conservative_picks_100k():
    """Conservative caps the asymmetric pair at 2 x 7700 MiB (~15.0 GiB):
    the 100k build (~15 GB measured) but not the 216k one (19.29 GB)."""
    assert get_coder_model(CONSERVATIVE, [INTERNAL_8GB, EGPU_16GB]) == "qwen3.8-100k"


def test_real_pair_proportional_picks_216k():
    """7700 + 15600 MiB (~22.8 GiB) clears the 18 GiB top tier."""
    assert get_coder_model(PROPORTIONAL, [INTERNAL_8GB, EGPU_16GB]) == "qwen3.8-216k"


def test_default_strategy_is_still_conservative():
    """A safety choice, not an oversight: the default must not assume the
    runtime splits layers proportionally."""
    assert get_coder_model(gpus=[INTERNAL_8GB, EGPU_16GB]) == "qwen3.8-100k"


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
    monkeypatch.setattr(coder_model_mod, "resident_vram_bytes", lambda: 0)
    monkeypatch.setattr(coder_model_mod, "installed_models", lambda: None)
    assert get_coder_model() == "qwen2.5-coder:7b"


def test_get_coder_model_falls_back_when_gpu_unavailable(monkeypatch):
    import ollama_tools.coder_model as coder_model_mod

    def raise_unavailable():
        raise GpuUnavailable("no nvidia-smi")

    monkeypatch.setattr(coder_model_mod, "query_gpus", raise_unavailable)
    assert get_coder_model() == CODER_FALLBACK


# The machine as it actually was when a pipeline run failed: qwen3.8-100k
# resident (15249189105 bytes of VRAM per /api/ps), leaving 3144 + 5598 MiB
# free. Free VRAM alone budgets 2 x 3144 MiB (~6.1 GiB) -> 1.5b.
INTERNAL_BUSY = Gpu(0, "RTX 5070 Laptop", 8151 * MIB, 3144 * MIB)
EGPU_BUSY = Gpu(1, "RTX 5060 Ti", 16311 * MIB, 5598 * MIB)
QWEN_100K_RESIDENT = 15249189105


def test_resident_ollama_model_counts_as_reclaimable():
    """Ollama evicts a resident model to load the next one, so its VRAM is
    part of the budget -- otherwise every call made while the previous
    stage's model is loaded gets downgraded to a tiny coder."""
    assert get_coder_model(CONSERVATIVE, [INTERNAL_BUSY, EGPU_BUSY]) == "qwen2.5-coder:1.5b"
    assert (
        get_coder_model(
            CONSERVATIVE, [INTERNAL_BUSY, EGPU_BUSY], resident_bytes=QWEN_100K_RESIDENT
        )
        == "qwen3.8-100k"
    )


def test_uninstalled_tier_is_skipped():
    installed = frozenset({"qwen2.5-coder:7b", "qwen3.8-216k:latest"})
    # Budget clears 100k, which isn't pulled; 14b isn't either; 7b is.
    assert (
        get_coder_model(CONSERVATIVE, [INTERNAL_8GB, EGPU_16GB], installed=installed)
        == "qwen2.5-coder:7b"
    )


def test_unknown_installed_set_skips_nothing():
    assert (
        get_coder_model(CONSERVATIVE, [INTERNAL_8GB, EGPU_16GB], installed=None)
        == "qwen3.8-100k"
    )


def test_live_query_reads_resident_and_installed(monkeypatch):
    import ollama_tools.coder_model as coder_model_mod

    monkeypatch.setattr(coder_model_mod, "query_gpus", lambda: [INTERNAL_BUSY, EGPU_BUSY])
    monkeypatch.setattr(coder_model_mod, "resident_vram_bytes", lambda: QWEN_100K_RESIDENT)
    monkeypatch.setattr(
        coder_model_mod, "installed_models", lambda: frozenset({"qwen3.8-100k:latest"})
    )
    assert get_coder_model() == "qwen3.8-100k"
