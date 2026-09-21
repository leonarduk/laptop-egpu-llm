"""What GPUs are attached, and how much of them a model may actually use.

The budget is the interesting part. Free VRAM summed across cards is only
reachable if the runtime places layers in proportion to free memory. LM
Studio's default is an even split, which caps you at the number of cards
times the *smallest* card, however big the others are -- 7.93 + 15.90 GB
gives ~15.9 GB, not 23.8 GB (see docs/lmstudio-multi-gpu.md).

Guessing high is what hangs the machine, so the default strategy takes
whichever of those two ceilings is lower.

**NVIDIA only.** Detection is ``nvidia-smi``, so Apple Silicon, AMD and
Intel GPUs raise :class:`GpuUnavailable` and every command exits 2. That
is a refusal, not support: without a VRAM figure there is no way to tell a
model that fits from one that hangs the machine, and guessing is the
failure this package exists to prevent. Cross-platform here means Windows
and Linux with an NVIDIA card, not every GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

GIB = 1024**3

CONSERVATIVE = "conservative"
PROPORTIONAL = "proportional"
EVEN = "even"
STRATEGIES = (CONSERVATIVE, PROPORTIONAL, EVEN)


class GpuUnavailable(RuntimeError):
    """nvidia-smi is missing, broken, or reports no GPU.

    Raised rather than returning an empty list: "no GPUs" and "could not
    ask" both mean the fit cannot be established, and a caller that
    silently treated either as a 0 GB budget would refuse everything for
    the wrong reason.
    """


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    total_bytes: int
    free_bytes: int

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB


def parse_nvidia_smi(output: str) -> list[Gpu]:
    """Parse ``--query-gpu=index,name,memory.total,memory.free`` CSV rows.

    Values are MiB, as ``--format=csv,noheader,nounits`` emits them. Rows
    that do not parse are skipped rather than raising: one malformed line
    should not hide the cards that did report.
    """
    gpus: list[Gpu] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            gpus.append(
                Gpu(
                    index=int(fields[0]),
                    name=fields[1],
                    total_bytes=int(float(fields[2])) * 1024 * 1024,
                    free_bytes=int(float(fields[3])) * 1024 * 1024,
                )
            )
        except ValueError:
            continue
    return gpus


def query_gpus() -> list[Gpu]:
    """The GPUs nvidia-smi reports right now.

    NVIDIA only -- there is no AMD, Intel or Apple Silicon path, and none
    is faked. See this module's docstring for why an unsupported GPU is a
    refusal rather than a gap.

    Raises:
        GpuUnavailable: nvidia-smi missing, failing, or reporting nothing.
    """
    if shutil.which("nvidia-smi") is None:
        raise GpuUnavailable(
            "nvidia-smi not found, so free VRAM cannot be established"
        )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuUnavailable(f"could not run nvidia-smi: {exc}") from exc

    if completed.returncode != 0:
        raise GpuUnavailable(
            f"nvidia-smi exited {completed.returncode}: {completed.stderr.strip()}"
        )

    gpus = parse_nvidia_smi(completed.stdout)
    if not gpus:
        raise GpuUnavailable("nvidia-smi reported no GPUs")
    return gpus


def budget_bytes(gpus: list[Gpu], strategy: str = CONSERVATIVE) -> int:
    """Free VRAM a model may actually reach, under ``strategy``.

    ``proportional`` sums; ``even`` is len(gpus) * smallest free, the
    ceiling an even split imposes; ``conservative`` takes the lower of the
    two, which equals ``even`` whenever the cards are asymmetric.
    """
    if not gpus:
        raise GpuUnavailable("no GPUs to budget")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")

    total = sum(g.free_bytes for g in gpus)
    even = min(g.free_bytes for g in gpus) * len(gpus)
    if strategy == PROPORTIONAL:
        return total
    if strategy == EVEN:
        return even
    return min(total, even)
