"""Does a model fit in the VRAM that is actually attached?

The file size is not the whole footprint: the KV cache lives in VRAM beside
the weights, grows with context length, and is multiplied by the number of
parallel slots. By default ``headroom_percent`` is an explicit margin for
that rather than a computed figure, because a confident wrong number would
be worse than an honest one -- see docs/model-picker.md.

Asking about parallel slots is the exception. N slots cost N full KV
caches, which no percentage of the weights can stand in for, so the KV
cache is then computed from the model's shape (see kvcache.py) and added
on top: ``weights x (1 + headroom) + slots x kv_cache``. The headroom stays,
now covering compute buffers and runtime overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gpu import GIB

DEFAULT_HEADROOM_PERCENT = 20


@dataclass(frozen=True)
class Verdict:
    model: str
    size_bytes: int
    needed_bytes: int
    budget_bytes: int
    parallel: int | None = None
    kv_slot_bytes: int = 0

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
    def needed_gib(self) -> float:
        return self.needed_bytes / GIB

    @property
    def short_gib(self) -> float:
        return self.short_bytes / GIB

    @property
    def kv_total_bytes(self) -> int:
        return (self.parallel or 0) * self.kv_slot_bytes


def required_bytes(size_bytes: int, headroom_percent: int = DEFAULT_HEADROOM_PERCENT) -> int:
    if headroom_percent < 0:
        raise ValueError("headroom_percent cannot be negative")
    return int(round(size_bytes * (1 + headroom_percent / 100)))


def judge(
    model: str,
    size_bytes: int,
    budget: int,
    headroom_percent: int = DEFAULT_HEADROOM_PERCENT,
    parallel: int | None = None,
    kv_slot_bytes: int = 0,
) -> Verdict:
    """``parallel=None`` is the headroom-only check. Any slot count,
    including 1, switches to adding ``parallel x kv_slot_bytes``."""
    if parallel is not None and parallel < 1:
        raise ValueError("parallel must be at least 1")
    if kv_slot_bytes < 0:
        raise ValueError("kv_slot_bytes cannot be negative")
    needed = required_bytes(size_bytes, headroom_percent) + (parallel or 0) * kv_slot_bytes
    return Verdict(
        model=model,
        size_bytes=size_bytes,
        needed_bytes=needed,
        budget_bytes=budget,
        parallel=parallel,
        kv_slot_bytes=kv_slot_bytes,
    )
