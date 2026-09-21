import pytest

from ollama_tools.gpu import (
    CONSERVATIVE,
    EVEN,
    GIB,
    Gpu,
    GpuUnavailable,
    PROPORTIONAL,
    budget_bytes,
    parse_nvidia_smi,
)

MIB = 1024 * 1024

# The real shape of this machine: 8 GB internal + 16 GB eGPU.
INTERNAL = Gpu(0, "NVIDIA GeForce RTX 5070 Laptop GPU", 8151 * MIB, 7700 * MIB)
EGPU = Gpu(1, "NVIDIA GeForce RTX 5060 Ti", 16311 * MIB, 16000 * MIB)


def test_parses_csv_rows_in_mib():
    gpus = parse_nvidia_smi("0, NVIDIA GeForce RTX 5070 Laptop GPU, 8151, 7700\n")
    assert len(gpus) == 1
    assert gpus[0].index == 0
    assert gpus[0].total_bytes == 8151 * MIB
    assert gpus[0].free_bytes == 7700 * MIB


def test_skips_malformed_rows_without_losing_good_ones():
    """One bad line must not hide the cards that did report."""
    gpus = parse_nvidia_smi("0, GPU A, 8151, 7700\ngarbage\n1, GPU B, notanumber, 5\n")
    assert [g.name for g in gpus] == ["GPU A"]


def test_blank_output_is_no_gpus():
    assert parse_nvidia_smi("\n  \n") == []


def test_single_gpu_budget_is_its_free_memory():
    for strategy in (CONSERVATIVE, PROPORTIONAL, EVEN):
        assert budget_bytes([INTERNAL], strategy) == 7700 * MIB


def test_even_split_caps_asymmetric_pair_at_twice_the_smaller():
    """The trap from docs/lmstudio-multi-gpu.md: 7.7 + 15.6 is not 23.3."""
    assert budget_bytes([INTERNAL, EGPU], EVEN) == 2 * 7700 * MIB
    assert budget_bytes([INTERNAL, EGPU], PROPORTIONAL) == (7700 + 16000) * MIB


def test_conservative_picks_the_lower_ceiling():
    conservative = budget_bytes([INTERNAL, EGPU], CONSERVATIVE)
    assert conservative == budget_bytes([INTERNAL, EGPU], EVEN)
    assert conservative < budget_bytes([INTERNAL, EGPU], PROPORTIONAL)


def test_conservative_equals_sum_when_cards_are_identical():
    """With symmetric cards an even split wastes nothing, so the two
    ceilings coincide and conservative costs you nothing."""
    twin = Gpu(1, "same", 8151 * MIB, 7700 * MIB)
    assert budget_bytes([INTERNAL, twin], CONSERVATIVE) == budget_bytes(
        [INTERNAL, twin], PROPORTIONAL
    )


def test_the_27b_needs_the_egpu():
    """The case that started this: 11.3 GB against one 8 GB card."""
    model_bytes = int(11.29 * GIB)
    assert model_bytes > budget_bytes([INTERNAL], CONSERVATIVE)
    assert model_bytes < budget_bytes([INTERNAL, EGPU], CONSERVATIVE)


def test_no_gpus_raises_rather_than_returning_zero_budget():
    """'Could not ask' must not read as a 0 GB budget - the caller maps the
    two to different exit codes."""
    with pytest.raises(GpuUnavailable):
        budget_bytes([], CONSERVATIVE)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        budget_bytes([INTERNAL], "wishful")
