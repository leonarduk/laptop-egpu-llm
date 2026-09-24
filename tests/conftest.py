"""Shared fixtures.

``qwen35_show`` is an ``/api/show`` payload shaped like the one Ollama
returns for ``qwen3.8-100k`` on this machine: a qwen35 hybrid build with 64
blocks, a full-attention layer every 4th block, 4 KV heads of 256, and
``num_ctx 100000`` set in its Modelfile. Tokenizer arrays are left out, as
Ollama itself leaves them out unless ``verbose`` is asked for.
"""

import copy

import pytest

QWEN35_SHOW = {
    "license": "Apache License 2.0",
    "modelfile": "FROM qwen3.8:27b\nPARAMETER num_ctx 100000\n",
    "parameters": (
        'num_ctx                        100000\n'
        'presence_penalty               1.5\n'
        'stop                           "<|im_start|>"\n'
        'stop                           "<|im_end|>"\n'
        'temperature                    1\n'
        'top_k                          20\n'
        'top_p                          0.95'
    ),
    "template": "{{ .Prompt }}",
    "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen35",
        "families": ["qwen35"],
        "parameter_size": "27.8B",
        "quantization_level": "Q4_K_M",
    },
    "model_info": {
        "general.architecture": "qwen35",
        "general.file_type": 15,
        "general.parameter_count": 27_781_427_952,
        "general.quantization_version": 2,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-06,
        "qwen35.attention.value_length": 256,
        "qwen35.block_count": 64,
        "qwen35.context_length": 262144,
        "qwen35.embedding_length": 5120,
        "qwen35.feed_forward_length": 17408,
        "qwen35.full_attention_interval": 4,
        "qwen35.rope.dimension_count": 64,
        "qwen35.rope.freq_base": 10_000_000,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.inner_size": 6144,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.time_step_rank": 48,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": None,
    },
    "capabilities": ["completion", "tools", "thinking"],
    "modified_at": "2026-09-20T10:12:44.1234567+01:00",
}


@pytest.fixture
def qwen35_show():
    """A fresh copy each time, so a test that edits it cannot leak."""
    return copy.deepcopy(QWEN35_SHOW)
