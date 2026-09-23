"""Does a model fit in the VRAM that is actually attached?

The file size is not the whole footprint: the KV cache lives in VRAM beside
the weights, grows with context length, and is multiplied by the number of
parallel slots. When Ollama's ``/api/show`` reports the architecture, that
cache is estimated here from the same numbers llama.cpp sizes it with (see
:func:`estimate_kv_cache`) and added on top. ``headroom_percent`` stays as
an explicit margin on the weights for what is still not computed -- the
compute graph, CUDA context, recurrent state on hybrid models -- and is the
only margin left when the architecture cannot be read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .gpu import GIB

DEFAULT_HEADROOM_PERCENT = 20

# Bytes per cached element, as (numerator, denominator) so the arithmetic
# stays exact. The quantised types are ggml blocks of 32 elements: q8_0 is
# 32 int8 plus one f16 scale (34 bytes), q4_0 is 32 nibbles plus one f16
# scale (18 bytes). Anything else Ollama does not recognise it treats as f16.
KV_CACHE_TYPES: dict[str, tuple[int, int]] = {
    "f16": (2, 1),
    "q8_0": (34, 32),
    "q4_0": (18, 32),
}
DEFAULT_KV_CACHE_TYPE = "f16"

# What Ollama uses when neither the Modelfile's num_ctx nor
# OLLAMA_CONTEXT_LENGTH says otherwise: 4096 on current builds with under
# 24 GiB of VRAM (older builds used 2048; newer ones raise the default on
# bigger cards). Pin num_ctx in the Modelfile if the difference matters --
# the estimate then reads it instead of assuming.
DEFAULT_NUM_CTX = 4096

# llama.cpp rounds the context up to a multiple of 256 cells, which is why
# num_ctx 100000 logs as n_ctx 100096.
KV_CELL_PADDING = 256


class KvUnknown(ValueError):
    """``/api/show`` did not carry enough to size the KV cache.

    Distinct from a model that is too big: the fit check then falls back to
    file size plus headroom and says so, rather than inventing a number.
    """


@dataclass(frozen=True)
class KvEstimate:
    bytes: int
    cache_type: str
    num_ctx: int
    ctx_source: str
    cells: int
    kv_layers: int
    block_count: int
    parallel: int

    @property
    def gib(self) -> float:
        return self.bytes / GIB


def _truthy_off(value: str | None) -> bool:
    return (value or "").strip().lower() in ("0", "false", "no", "off")


def kv_cache_type(env: Mapping[str, str]) -> str:
    """The cache type Ollama would use, read from ``env``.

    This is the *client's* environment standing in for the server's -- the
    server does not report its setting -- so it is only right if both were
    started with the same variables. Quantised caches need flash attention;
    with it switched off explicitly Ollama falls back to f16.
    """
    requested = (env.get("OLLAMA_KV_CACHE_TYPE") or "").strip().lower()
    if requested not in KV_CACHE_TYPES:
        return DEFAULT_KV_CACHE_TYPE
    if requested != "f16" and _truthy_off(env.get("OLLAMA_FLASH_ATTENTION")):
        return DEFAULT_KV_CACHE_TYPE
    return requested


def num_parallel(env: Mapping[str, str]) -> int:
    """OLLAMA_NUM_PARALLEL, or 1. Ollama allocates num_ctx per slot, so the
    cache scales linearly with it."""
    try:
        value = int((env.get("OLLAMA_NUM_PARALLEL") or "").strip())
    except ValueError:
        return 1
    return value if value > 0 else 1


def _modelfile_num_ctx(parameters: object) -> int | None:
    """num_ctx from /api/show's ``parameters``, a Modelfile-style string of
    ``name value`` lines."""
    if not isinstance(parameters, str):
        return None
    for line in parameters.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "num_ctx":
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def _int(info: Mapping[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _context(
    show: Mapping[str, object], info: Mapping[str, object], arch: str, env: Mapping[str, str]
) -> tuple[int, str]:
    """(num_ctx, where it came from), in Ollama's own order of precedence:
    the Modelfile, then OLLAMA_CONTEXT_LENGTH, then the built-in default --
    each capped at the context the model was trained for, as Ollama does."""
    num_ctx = _modelfile_num_ctx(show.get("parameters"))
    source = "Modelfile"
    if num_ctx is None:
        try:
            num_ctx = int((env.get("OLLAMA_CONTEXT_LENGTH") or "").strip())
            source = "OLLAMA_CONTEXT_LENGTH"
        except ValueError:
            num_ctx = None
    if num_ctx is None or num_ctx <= 0:
        num_ctx, source = DEFAULT_NUM_CTX, "Ollama default"
    trained = _int(info, f"{arch}.context_length")
    if trained and num_ctx > trained:
        num_ctx, source = trained, f"{source}, capped at the trained length"
    return num_ctx, source


def _kv_elements_per_token(info: Mapping[str, object], arch: str) -> tuple[int, int, int]:
    """(K+V elements per token summed over the layers that cache, number of
    such layers, block_count)."""
    blocks = _int(info, f"{arch}.block_count")
    if not blocks:
        raise KvUnknown(f"no {arch}.block_count")

    heads = _int(info, f"{arch}.attention.head_count")
    key_length = _int(info, f"{arch}.attention.key_length")
    if key_length is None:
        embedding = _int(info, f"{arch}.embedding_length")
        if not embedding or not heads:
            raise KvUnknown(f"no {arch}.attention.key_length, and no embedding_length/head_count to derive it")
        key_length = embedding // heads
    value_length = _int(info, f"{arch}.attention.value_length") or key_length

    # Hybrid architectures (qwen35, qwen3next) interleave recurrent layers
    # with full attention; only every Nth layer -- the ones where
    # (layer + 1) % N == 0 -- keeps a KV cache.
    interval = _int(info, f"{arch}.full_attention_interval") or 1

    kv_heads = info.get(f"{arch}.attention.head_count_kv")
    if isinstance(kv_heads, list):
        # Per-layer counts; a zero marks a layer without attention.
        per_layer = [int(h) for h in kv_heads[:blocks] if isinstance(h, (int, float))]
    else:
        count = _int(info, f"{arch}.attention.head_count_kv") or heads
        if not count:
            raise KvUnknown(f"no {arch}.attention.head_count_kv or head_count")
        per_layer = [count] * blocks
    cached = [h for i, h in enumerate(per_layer) if h > 0 and (i + 1) % interval == 0]
    if not cached:
        raise KvUnknown("no layer reports a KV head count")
    elements = sum(h * (key_length + value_length) for h in cached)
    return elements, len(cached), blocks


def estimate_kv_cache(show: Mapping[str, object], env: Mapping[str, str]) -> KvEstimate:
    """KV cache bytes llama.cpp will reserve for this model, from an
    ``/api/show`` payload and the (client-side) Ollama environment.

    cells x (K+V elements per token over the caching layers) x bytes per
    element, where cells is num_ctx x OLLAMA_NUM_PARALLEL rounded up to 256.
    Checked against a real load: qwen35 with 16 of 64 layers caching, 4 KV
    heads of 256, num_ctx 100000 at q4_0 logs ``size = 1759.50 MiB`` and so
    does this. Sliding-window layers (gemma3) are sized as full attention,
    which overstates their cache -- the safe direction.

    Raises:
        KvUnknown: the architecture keys needed are missing.
    """
    info = show.get("model_info")
    if not isinstance(info, Mapping):
        raise KvUnknown("/api/show returned no model_info")
    arch = info.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise KvUnknown("/api/show returned no general.architecture")

    elements, kv_layers, blocks = _kv_elements_per_token(info, arch)
    num_ctx, source = _context(show, info, arch, env)
    parallel = num_parallel(env)
    requested = num_ctx * parallel
    cells = -(-requested // KV_CELL_PADDING) * KV_CELL_PADDING
    cache_type = kv_cache_type(env)
    numerator, denominator = KV_CACHE_TYPES[cache_type]
    return KvEstimate(
        bytes=cells * elements * numerator // denominator,
        cache_type=cache_type,
        num_ctx=num_ctx,
        ctx_source=source,
        cells=cells,
        kv_layers=kv_layers,
        block_count=blocks,
        parallel=parallel,
    )


@dataclass(frozen=True)
class Verdict:
    model: str
    size_bytes: int
    needed_bytes: int
    budget_bytes: int
    kv_bytes: int = 0

    @property
    def fits(self) -> bool:
        return self.needed_bytes <= self.budget_bytes

    @property
    def short_bytes(self) -> int:
        return max(0, self.needed_bytes - self.budget_bytes)

    @property
    def size_gib(self) -> float:
        return self.size_bytes / GIB

    @property
    def kv_gib(self) -> float:
        return self.kv_bytes / GIB

    @property
    def needed_gib(self) -> float:
        return self.needed_bytes / GIB

    @property
    def short_gib(self) -> float:
        return self.short_bytes / GIB


def required_bytes(size_bytes: int, headroom_percent: int = DEFAULT_HEADROOM_PERCENT) -> int:
    if headroom_percent < 0:
        raise ValueError("headroom_percent cannot be negative")
    return int(round(size_bytes * (1 + headroom_percent / 100)))


def judge(
    model: str,
    size_bytes: int,
    budget: int,
    headroom_percent: int = DEFAULT_HEADROOM_PERCENT,
    kv_bytes: int = 0,
) -> Verdict:
    """Weights plus headroom, plus the KV cache when it could be estimated
    (``kv_bytes`` 0 means it could not, and headroom is the only margin)."""
    return Verdict(
        model=model,
        size_bytes=size_bytes,
        needed_bytes=required_bytes(size_bytes, headroom_percent) + kv_bytes,
        budget_bytes=budget,
        kv_bytes=kv_bytes,
    )
