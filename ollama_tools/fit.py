"""Does a model fit in the VRAM that is actually attached?

The file size is not the whole footprint: the KV cache lives in VRAM beside
the weights, grows with context length, and is multiplied by the number of
parallel slots. ``headroom_percent`` is an explicit margin for that rather
than a computed figure, because a confident wrong number would be worse
than an honest one -- see docs/model-picker.md.
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


def required_bytes(size_bytes: int, headroom_percent: int = DEFAULT_HEADROOM_PERCENT) -> int:
    if headroom_percent < 0:
        raise ValueError("headroom_percent cannot be negative")
    return int(round(size_bytes * (1 + headroom_percent / 100)))


def judge(
    model: str,
    size_bytes: int,
    budget: int,
    headroom_percent: int = DEFAULT_HEADROOM_PERCENT,
) -> Verdict:
    return Verdict(
        model=model,
        size_bytes=size_bytes,
        needed_bytes=required_bytes(size_bytes, headroom_percent),
        budget_bytes=budget,
    )
