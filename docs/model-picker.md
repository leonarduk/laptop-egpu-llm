# Picking a model for the VRAM you actually have

Three numbers matter on this build, and which one applies depends on whether
the eGPU is attached and how the runtime splits across cards.

| Tier | VRAM budget | When |
|---|---|---|
| Internal only | **7.93 GB** | eGPU detached, or on Code 31 (see [`device-error-codes.md`](device-error-codes.md)) |
| Both cards, even split | **~15.9 GB** | Both attached, runtime splitting evenly — `2 x 7.93` |
| Both cards, proportional | **~23.8 GB** | Both attached, runtime placing in proportion to free memory |

The middle row is the trap. An even split caps you at twice the *smaller*
card however big the other one is, so the 16 GB card sits half empty — see
[`lmstudio-multi-gpu.md`](lmstudio-multi-gpu.md) for changing Strategy away
from "Split evenly". `Test-ModelFits.ps1` assumes the middle row by default,
because guessing high is what hangs the machine.

## The arithmetic, which does not go stale

```
weights ≈ parameters × bits-per-weight ÷ 8
```

| Quant | Effective bits | 7B | 14B | 27B | 32B |
|---|---|---|---|---|---|
| `IQ3_S` | ~3.5 | 3.1 GB | 6.1 GB | **11.3 GB** | 14.0 GB |
| `Q4_K_M` | ~4.8 | **4.4 GB** | **8.4 GB** | 16.2 GB | 19.2 GB |
| `Q5_K_M` | ~5.7 | 5.0 GB | 10.0 GB | 19.2 GB | 22.8 GB |
| `Q8_0` | ~8.5 | 7.4 GB | 14.9 GB | 28.7 GB | 34.0 GB |

Bold entries are measured on this machine via `ollama list`; the rest are
the formula. Checked against four models actually pulled here, the formula
runs within about 6% of the real file size — which is the point: you can
size a model you have not pulled yet, and 6% is well inside the headroom you
should be leaving anyway.

**Then add the KV cache**, which is not in those numbers and is what turns a
"fits on paper" model into a failed load:

- It scales linearly with context length. A model happy at 8k may not load at 64k.
- It is multiplied by `n_slots`. Four parallel slots quadruple it; set slots to 1 unless you are genuinely serving concurrent requests.
- `Q8_0` KV quantisation roughly halves it for negligible quality cost; `Q4_0` quarters it.

Rule of thumb: budget 15–20% on top at ordinary context, 35–50% at long
context. That is exactly what `-HeadroomPercent` is for.

## What to run per tier

Sizes are what fits. Which model is *best* at a size changes every few
months, so treat the names as "worth trying, then benchmark", not a
ranking — this repo has no benchmarks of its own yet.

### Internal only — 7.93 GB

One model at a time, and that is the real constraint: two 6 GB models cannot
both stay resident, so a pipeline that alternates between them will evict and
reload at every step. Prefer **one model doing several jobs** here.

| Job | Size to aim for | On this machine |
|---|---|---|
| Coder | 7B at Q4 | `qwen2.5-coder:7b` (4.36 GB) |
| General | 8–9B at Q4 | `qwen3.5:9b` (6.14 GB), `deepseek-r1:8b` (4.87 GB) |
| Triage / classify | 3–4B at Q4 | `gemma3:4b` (3.11 GB) |
| Embeddings | any | ~0.3–1 GB — fits *alongside* a 7B, unlike everything else here |

A 14B at Q4 (8.4 GB) does **not** fit this tier. That is not a close call
once headroom is counted.

### Both cards, even split — ~15.9 GB

| Job | Size to aim for | On this machine |
|---|---|---|
| Coder | 14B at Q4 | `qwen2.5-coder:14b` (8.37 GB) |
| General | 27B at IQ3, *short context only* | `qwen3.8-64k` at a `num_ctx` under ~30k, ordinary context (11.29 + 20% = 13.5 GB) |

A model's KV cache is reserved for its manifest's `num_ctx` on every load,
whether or not a given prompt is anywhere near that long — see
[`ollama-multi-gpu.md`](ollama-multi-gpu.md) for the measured bytes-per-token
cost of raising it. `qwen3.8-216k` (this repo's actual pulled tag, rebuilt with
`num_ctx 216000`) always reserves the full 216k KV cache and needs ~18-19 GB
total, so it belongs in the proportional tier below, not here — it no longer
fits an even split regardless of how short the prompt actually is. Only a
build of this model with a short `num_ctx` (roughly 30k or less) fits this
tier; measure with `ollama-tools fit`, do not assume from the model's
architecture-max size.

### Both cards, proportional — ~23.8 GB

| Job | Size to aim for | On this machine |
|---|---|---|
| Coder | 32B at Q4 (~19.2 GB), or 14B at Q8 for higher fidelity at the same size | `qwen2.5-coder:32b` (18.49 GB) |
| General | 27B at Q4 (~16.2 GB) instead of IQ3 — same model, better quantisation | `qwen3.8-216k` (11.29 GB weights + ~8 GB KV cache at 216k context, ~18-19 GB total) |

Going from IQ3_S to Q4_K_M on a model you already run is often a better use
of new VRAM than a bigger model at a worse quant.

## Model families worth trying

Current as of September 2026 and **not benchmarked here** — verify before
trusting, and prefer whatever your own `Measure-ModelSpeed.ps1` numbers say.

- **Coding**: Qwen2.5-Coder (1.5B/7B/14B/32B) is the reliable spread across every tier here. DeepSeek-Coder-V2-Lite and Codestral are the usual alternatives to compare against.
- **General / reasoning**: the Qwen3 line, Llama 3.x, Gemma 3, Mistral Small. Reasoning models (`deepseek-r1` and kin) spend many more tokens per answer, so judge them on time-to-answer, not tokens/sec.
- **Embeddings**: `nomic-embed-text`, `mxbai-embed-large`, `bge-m3`. Small enough to ignore in VRAM planning, and a different job from chat — do not use a chat model for retrieval.
- **Reranking**: `bge-reranker` family, once retrieval exists and precision matters more than recall.
- **Vision**: Qwen2.5-VL, Llama 3.2 Vision, if anything needs to read screenshots or diagrams.

## Checking before you load

```powershell
.\diagnostics\Test-ModelFits.ps1 -Model qwen3.8-216k:latest
.\ollama\Start-Model.ps1 -Model qwen3.8-216k:latest     # fit-checked launch
```
