"""How much VRAM one parallel slot's KV cache costs, from the model's own shape.

Ollama gives every parallel slot (``OLLAMA_NUM_PARALLEL``) its own full
``num_ctx`` of KV cache, reserved at load time whether or not a request
ever gets that long. So N slots cost N times the KV cache, which a
percentage headroom on the weights cannot stand in for: at 216k context
one slot's KV cache is about the size of the weights themselves.

The per-token cost is plain architecture arithmetic:

    bytes/token = sum over layers (kv_heads x (key_dim + value_dim)) x bytes/element

It is only computed for plain attention models. Hybrid architectures
(recurrent/SSM layers mixed with attention, e.g. qwen35 / qwen3next) and
compressed-KV attention (MLA, e.g. deepseek2) do not follow that formula.
For those, :class:`KvCacheUnknown` is raised and the CLI exits 2, instead
of returning a figure that could be off by a factor of two in the
direction that hangs the machine. Measure those with
diagnostics/measure-context-ceiling.sh.

Sliding-window layers (e.g. gemma3) are counted as full attention. That
over-counts, which is the safe direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Bytes per element as llama.cpp stores them: q8_0 is 32 int8 + one f16
# scale per block of 32; q4_0 is 32 nibbles + one f16 scale.
KV_CACHE_TYPES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
DEFAULT_KV_CACHE_TYPE = "f16"

# Keys (after the "<arch>." prefix) whose presence means the attention-only
# formula does not apply. Prefixes, not substrings, so an unrelated key that
# merely contains "ssm." somewhere does not trip it.
_UNSUPPORTED_PREFIXES = ("full_attention_interval", "ssm.", "attention.kv_lora_rank")

_NUM_CTX = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


class KvCacheUnknown(ValueError):
    """The KV cache cost cannot be computed for this model.

    A "could not answer", like OllamaUnavailable -- not a "does not fit".
    """


@dataclass(frozen=True)
class KvShape:
    architecture: str
    elements_per_token: int
    context: int

    def slot_bytes(self, kv_cache_type: str = DEFAULT_KV_CACHE_TYPE) -> int:
        try:
            per_element = KV_CACHE_TYPES[kv_cache_type]
        except KeyError:
            raise ValueError(f"unknown KV cache type {kv_cache_type!r}") from None
        return int(round(self.elements_per_token * self.context * per_element))


def _per_layer(value, layers: int, name: str) -> list[int]:
    if isinstance(value, list):
        if len(value) != layers:
            raise KvCacheUnknown(f"{name} lists {len(value)} layers, block_count says {layers}")
        return [int(v) for v in value]
    return [int(value)] * layers


def manifest_num_ctx(parameters: str) -> int | None:
    """``PARAMETER num_ctx`` from a Modelfile, as /api/show reports it."""
    match = _NUM_CTX.search(parameters or "")
    return int(match.group(1)) if match else None


def kv_shape(show: dict, num_ctx: int | None = None) -> KvShape:
    """Read the KV cache shape from an /api/show response.

    Context, in order: an explicit ``num_ctx``, the manifest's
    ``PARAMETER num_ctx``, then the architecture's maximum. The last is an
    upper bound -- a server with ``OLLAMA_CONTEXT_LENGTH`` set may load
    with less -- so it errs towards refusing.
    """
    try:
        return _kv_shape(show, num_ctx)
    except KvCacheUnknown:
        raise
    except (TypeError, ValueError) as exc:
        # A model_info value of an unexpected type: say so and exit 2, not
        # a traceback.
        raise KvCacheUnknown(f"unexpected /api/show model_info: {exc}") from exc


def _kv_shape(show: dict, num_ctx: int | None) -> KvShape:
    info = show.get("model_info") or {}
    arch = info.get("general.architecture")
    if not arch:
        raise KvCacheUnknown("/api/show reported no general.architecture")

    prefix = f"{arch}."
    for key in info:
        if key.startswith(prefix) and key[len(prefix):].startswith(_UNSUPPORTED_PREFIXES):
            raise KvCacheUnknown(
                f"{arch} is not a plain attention architecture ({key}); its KV cache "
                "cannot be computed from layer shape. Measure it with "
                "diagnostics/measure-context-ceiling.sh"
            )

    def get(name: str):
        return info.get(f"{arch}.{name}")

    layers = get("block_count")
    heads = get("attention.head_count")
    if not layers:
        raise KvCacheUnknown(f"{arch}.block_count missing from /api/show")
    layers = int(layers)

    kv_heads = get("attention.head_count_kv")
    if kv_heads is None:
        kv_heads = heads  # no GQA: every head has its own K and V
    if kv_heads is None:
        raise KvCacheUnknown(f"{arch}.attention.head_count_kv missing from /api/show")
    kv_heads = _per_layer(kv_heads, layers, "head_count_kv")

    key_dim = get("attention.key_length")
    value_dim = get("attention.value_length")
    if key_dim is None or value_dim is None:
        embedding = get("embedding_length")
        if not embedding or not heads or isinstance(heads, list):
            raise KvCacheUnknown(f"{arch} head dimensions missing from /api/show")
        derived = int(embedding) // int(heads)
        key_dim = derived if key_dim is None else key_dim
        value_dim = derived if value_dim is None else value_dim

    key_dims = _per_layer(key_dim, layers, "key_length")
    value_dims = _per_layer(value_dim, layers, "value_length")
    elements = sum(h * (k + v) for h, k, v in zip(kv_heads, key_dims, value_dims))

    context = num_ctx or manifest_num_ctx(show.get("parameters", "")) or get("context_length")
    if not context:
        raise KvCacheUnknown(f"no num_ctx in the manifest and no {arch}.context_length")

    return KvShape(architecture=arch, elements_per_token=elements, context=int(context))
