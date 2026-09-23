import pytest

from ollama_tools.fit import judge
from ollama_tools.gpu import GIB
from ollama_tools.kvcache import KvCacheUnknown, kv_shape, manifest_num_ctx

# qwen2.5-coder:14b as /api/show reports it: 48 layers, GQA with 8 KV heads
# of 128 dims, no explicit key/value length.
QWEN25_14B = {
    "model_info": {
        "general.architecture": "qwen2",
        "qwen2.block_count": 48,
        "qwen2.attention.head_count": 40,
        "qwen2.attention.head_count_kv": 8,
        "qwen2.embedding_length": 5120,
        "qwen2.context_length": 32768,
    },
    "parameters": 'stop                           "<|im_end|>"',
}


def test_f16_slot_matches_the_measured_14b_load():
    """48 x 8 x (128 + 128) x 32768 x 2 bytes is exactly 6 GiB. Measured on
    this machine at f16: 15.05 GB total for 8.37 GB of weights, so the
    formula lands within the compute buffers of reality."""
    shape = kv_shape(QWEN25_14B)
    assert shape.context == 32768
    assert shape.slot_bytes("f16") == 6 * GIB


def test_quantised_kv_types_shrink_the_slot():
    shape = kv_shape(QWEN25_14B)
    assert shape.slot_bytes("q8_0") == pytest.approx(6 * GIB * 34 / 64, rel=1e-9)
    assert shape.slot_bytes("q4_0") == pytest.approx(6 * GIB * 18 / 64, rel=1e-9)


def test_unknown_kv_type_is_rejected():
    with pytest.raises(ValueError):
        kv_shape(QWEN25_14B).slot_bytes("q2_k")


def test_manifest_num_ctx_beats_the_architecture_max():
    show = dict(QWEN25_14B, parameters='num_ctx                        8192\nstop "x"')
    assert kv_shape(show).context == 8192


def test_explicit_num_ctx_beats_the_manifest():
    show = dict(QWEN25_14B, parameters="num_ctx 8192")
    assert kv_shape(show, num_ctx=54000).context == 54000


def test_manifest_num_ctx_parsing():
    assert manifest_num_ctx("num_ctx                        216000") == 216000
    assert manifest_num_ctx("num_ctx_extra 5") is None
    assert manifest_num_ctx("") is None


def test_explicit_key_and_value_lengths_and_per_layer_heads():
    show = {
        "model_info": {
            "general.architecture": "x",
            "x.block_count": 2,
            "x.attention.head_count": 8,
            "x.attention.head_count_kv": [4, 0],
            "x.attention.key_length": 256,
            "x.attention.value_length": 128,
            "x.context_length": 10,
        }
    }
    assert kv_shape(show).elements_per_token == 4 * (256 + 128)


@pytest.mark.parametrize(
    "key",
    ["qwen35.full_attention_interval", "qwen35.ssm.state_size", "qwen35.attention.kv_lora_rank"],
)
def test_hybrid_and_compressed_kv_architectures_are_refused(key):
    """The layer-shape formula is wrong for these, possibly by 2x in the
    direction that hangs the machine, so no figure is better than that one."""
    info = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 64,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.context_length": 262144,
        key: 4,
    }
    with pytest.raises(KvCacheUnknown, match="measure-context-ceiling"):
        kv_shape({"model_info": info})


def test_missing_architecture_is_unknown():
    with pytest.raises(KvCacheUnknown):
        kv_shape({"model_info": {}})


def test_judge_without_parallel_is_unchanged():
    assert judge("m", 1000, 1200, 20).needed_bytes == 1200


def test_judge_adds_every_slot():
    verdict = judge("m", 1000, 10_000, 20, parallel=4, kv_slot_bytes=500)
    assert verdict.needed_bytes == 1200 + 4 * 500
    assert verdict.kv_total_bytes == 2000


def test_judge_rejects_zero_slots():
    with pytest.raises(ValueError):
        judge("m", 1000, 1200, 20, parallel=0)
