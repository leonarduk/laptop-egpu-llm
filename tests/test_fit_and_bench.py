import pytest

from ollama_tools.bench import is_embedding_model, run_embedding, run_generation
from ollama_tools.fit import (
    DEFAULT_NUM_CTX,
    KvUnknown,
    estimate_kv_cache,
    judge,
    kv_cache_type,
    num_parallel,
    required_bytes,
)
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


# --- KV cache estimate ------------------------------------------------------

MIB = 1024 * 1024
# The server this machine runs, as the CLI would read it from the shell.
MACHINE_ENV = {
    "OLLAMA_KV_CACHE_TYPE": "q4_0",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_SCHED_SPREAD": "1",
}


def test_kv_estimate_matches_the_measured_qwen3_8_100k_load(qwen35_show):
    """The number this estimate is pinned to. Loading qwen3.8-100k on this
    machine logged:

        llama_kv_cache: size = 1759.50 MiB (100096 cells, 16 layers, 1/1 seqs),
        K (q4_0): 879.75 MiB, V (q4_0): 879.75 MiB

    100000 rounds up to 100096 cells; only 16 of the 64 blocks cache (one
    in four is full attention); each caches 4 heads x (256 K + 256 V) =
    2048 elements per token, which at q4_0's 18 bytes per 32 is 1152 bytes.
    """
    kv = estimate_kv_cache(qwen35_show, MACHINE_ENV)
    assert kv.cells == 100096
    assert kv.kv_layers == 16
    assert kv.block_count == 64
    assert kv.cache_type == "q4_0"
    assert kv.num_ctx == 100000 and kv.ctx_source == "Modelfile"
    assert kv.bytes == 100096 * 16 * 1152
    assert kv.bytes / MIB == pytest.approx(1759.50)


@pytest.mark.parametrize(
    "cache_type, expected_mib",
    [("q4_0", 1759.50), ("q8_0", 1759.50 * 34 / 18), ("f16", 1759.50 * 64 / 18)],
)
def test_kv_cache_type_sets_bytes_per_element(qwen35_show, cache_type, expected_mib):
    env = dict(MACHINE_ENV, OLLAMA_KV_CACHE_TYPE=cache_type)
    assert estimate_kv_cache(qwen35_show, env).bytes / MIB == pytest.approx(expected_mib)


def test_unset_or_unknown_cache_type_is_f16(qwen35_show):
    """Ollama's own default, and what it does with a value it does not know."""
    assert kv_cache_type({}) == "f16"
    assert kv_cache_type({"OLLAMA_KV_CACHE_TYPE": "q3_k"}) == "f16"


def test_quantised_cache_without_flash_attention_is_f16():
    """Ollama cannot quantise the cache with flash attention switched off,
    and silently uses f16 -- four times the q4_0 figure."""
    assert kv_cache_type({"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "0"}) == "f16"


def test_parallel_slots_multiply_the_cache(qwen35_show):
    one = estimate_kv_cache(qwen35_show, MACHINE_ENV)
    two = estimate_kv_cache(qwen35_show, dict(MACHINE_ENV, OLLAMA_NUM_PARALLEL="2"))
    assert two.parallel == 2
    assert two.cells == 200192  # 2 x 100000, rounded up to 256
    assert two.bytes == 200192 * 16 * 1152
    assert num_parallel({"OLLAMA_NUM_PARALLEL": "nonsense"}) == 1


def test_context_without_a_modelfile_num_ctx(qwen35_show):
    """No num_ctx in the Modelfile: OLLAMA_CONTEXT_LENGTH if set, else
    Ollama's default, capped at the trained context either way."""
    qwen35_show["parameters"] = 'stop "<|im_end|>"'
    assert estimate_kv_cache(qwen35_show, MACHINE_ENV).num_ctx == DEFAULT_NUM_CTX

    env = dict(MACHINE_ENV, OLLAMA_CONTEXT_LENGTH="32768")
    kv = estimate_kv_cache(qwen35_show, env)
    assert (kv.num_ctx, kv.ctx_source) == (32768, "OLLAMA_CONTEXT_LENGTH")

    env = dict(MACHINE_ENV, OLLAMA_CONTEXT_LENGTH="1000000")
    kv = estimate_kv_cache(qwen35_show, env)
    assert kv.num_ctx == 262144 and "capped" in kv.ctx_source


def test_plain_transformer_caches_every_layer_and_derives_head_dim():
    """No full_attention_interval: every block caches. No key_length:
    embedding_length / head_count, as llama.cpp derives it."""
    show = {
        "parameters": "num_ctx 8192",
        "model_info": {
            "general.architecture": "qwen2",
            "qwen2.block_count": 28,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
            "qwen2.embedding_length": 3584,
            "qwen2.context_length": 32768,
        },
    }
    kv = estimate_kv_cache(show, {})
    assert kv.kv_layers == 28
    # 8192 cells x 28 layers x 4 heads x (128 + 128) x 2 bytes = 448 MiB.
    assert kv.bytes == 8192 * 28 * 4 * 256 * 2
    assert kv.bytes / MIB == pytest.approx(448)


def test_per_layer_kv_head_counts_skip_layers_without_attention():
    show = {
        "parameters": "num_ctx 256",
        "model_info": {
            "general.architecture": "hybrid",
            "hybrid.block_count": 4,
            "hybrid.attention.head_count_kv": [0, 2, 0, 2],
            "hybrid.attention.key_length": 64,
        },
    }
    kv = estimate_kv_cache(show, {})
    assert kv.kv_layers == 2
    assert kv.bytes == 256 * 2 * 2 * (64 + 64) * 2


def _hybrid(kv_heads, **extra):
    info = {
        "general.architecture": "hybrid",
        "hybrid.block_count": 4,
        "hybrid.attention.head_count_kv": kv_heads,
        "hybrid.attention.key_length": 64,
    }
    info.update(extra)
    return {"parameters": "num_ctx 256", "model_info": info}


@pytest.mark.parametrize("bad", ["2", None, 2.5, True, {"n": 2}])
def test_a_bad_per_layer_entry_is_refused_not_dropped(bad):
    """Dropping it would shift every later layer down one index, and the
    full_attention_interval test would then pick the wrong layers."""
    show = _hybrid([bad, 2, 0, 2], **{"hybrid.full_attention_interval": 2})
    with pytest.raises(KvUnknown) as caught:
        estimate_kv_cache(show, {})
    assert "head_count_kv[0]" in str(caught.value)


def test_a_per_layer_list_shorter_than_block_count_is_refused():
    with pytest.raises(KvUnknown) as caught:
        estimate_kv_cache(_hybrid([0, 2, 0]), {})
    assert "block_count is 4" in str(caught.value)


def test_integral_floats_are_accepted_per_layer_and_scalar():
    """JSON can carry 2 as 2.0; that is still a whole number."""
    assert estimate_kv_cache(_hybrid([0, 2.0, 0, 2]), {}).kv_layers == 2
    show = _hybrid(4.0, **{"hybrid.block_count": 4.0})
    assert estimate_kv_cache(show, {}).kv_layers == 4


@pytest.mark.parametrize(
    "key", ["hybrid.block_count", "hybrid.attention.key_length", "hybrid.full_attention_interval"]
)
def test_a_non_integral_scalar_is_refused_not_truncated(key):
    """4.5 is a malformed payload; truncating it to 4 would produce a
    confident wrong number."""
    show = _hybrid(2, **{key: 4.5})
    with pytest.raises(KvUnknown) as caught:
        estimate_kv_cache(show, {})
    assert "4.5" in str(caught.value)


def test_num_ctx_override_beats_the_modelfile_and_is_capped(qwen35_show):
    """fit --num-ctx: sizing a context before a tag is rebuilt with it."""
    kv = estimate_kv_cache(qwen35_show, MACHINE_ENV, num_ctx=50000)
    assert (kv.num_ctx, kv.ctx_source) == (50000, "--num-ctx")
    kv = estimate_kv_cache(qwen35_show, MACHINE_ENV, num_ctx=1_000_000)
    assert kv.num_ctx == 262144 and kv.ctx_source.startswith("--num-ctx, capped")


def test_modelfile_num_ctx_beats_ollama_context_length(qwen35_show):
    """OLLAMA_CONTEXT_LENGTH is only the server's default for models that
    do not set num_ctx; a Modelfile PARAMETER num_ctx does set it."""
    env = dict(MACHINE_ENV, OLLAMA_CONTEXT_LENGTH="8192")
    kv = estimate_kv_cache(qwen35_show, env)
    assert (kv.num_ctx, kv.ctx_source) == (100000, "Modelfile")


@pytest.mark.parametrize(
    "show",
    [
        {},
        {"model_info": {}},
        {"model_info": {"general.architecture": "llama"}},
        {"model_info": {"general.architecture": "llama", "llama.block_count": 32}},
    ],
)
def test_missing_architecture_keys_raise_kv_unknown(show):
    """Not a guess: the fit check falls back to file size plus headroom."""
    with pytest.raises(KvUnknown):
        estimate_kv_cache(show, MACHINE_ENV)


def test_kv_is_added_on_top_of_weights_and_headroom():
    verdict = judge("m", 1000, 10_000, 20, kv_bytes=500)
    assert verdict.needed_bytes == 1200 + 500
    assert verdict.kv_bytes == 500


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
