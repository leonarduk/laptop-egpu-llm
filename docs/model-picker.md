# Picking a model for the VRAM you actually have

Three numbers matter on this build, and which one applies depends on whether
the eGPU is attached and how the runtime splits across cards.

| Tier | VRAM budget | When |
|---|---|---|
| Internal only | **7.93 GB** | eGPU detached, or on Code 31 (see [`device-error-codes.md`](device-error-codes.md)) |
| Both cards, even split | **~15.9 GB** | Both attached, runtime splitting evenly — `2 x 7.93` |
| Both cards, proportional | **~23.8 GB** | Both attached, runtime placing in proportion to free memory |

Units: `ollama` reports decimal GB, `nvidia-smi` reports MiB (binary), and the
`ollama-tools` CLI and its tier tables use GiB (1 GiB = 1024 MiB = 1.074 GB). The GB
figures in this doc are decimal: the even-split ~15.9 GB is ~14.8 GiB, and
`qwen3.8-216k` at 19.29 GB is 17.97 GiB.

The middle row is the trap. An even split caps you at twice the *smaller*
card however big the other one is, so the 16 GB card sits half empty — see
[`lmstudio-multi-gpu.md`](lmstudio-multi-gpu.md) for changing Strategy away
from "Split evenly". `ollama-tools fit` assumes the middle row by default (`--strategy conservative`),
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
- It is multiplied by `n_slots`. Four parallel slots quadruple it; set slots to 1 unless you are genuinely serving concurrent requests. `ollama-tools fit <model> --parallel N` sizes it for N slots.
- `Q8_0` KV quantisation roughly halves it for negligible quality cost; `Q4_0` quarters it.

Rule of thumb: budget 15–20% on top at ordinary context, 35–50% at long
context. That is exactly what `ollama-tools fit --headroom` is for.

## What to run per tier

Sizes are what fits. Which model is *best* at a size changes every few
months, so treat the names as "worth trying, then benchmark", not a
ranking. What was actually measured here is in
[Models tried on this machine](#models-tried-on-this-machine).

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
| General | 27B at IQ3, *reduced context* | `qwen3.8-100k` (~15 GB total with a `q4_0` KV cache, measured 100% GPU under Ollama). **Tight** against this tier's ~15.9 GB even-split budget: well under the 15-20% headroom rule, so only if nothing else is using the cards |

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
| Coder + general | 27B with long context | `qwen3.8-216k` (11.29 GB weights + ~8 GB KV cache at 216k context, ~18-19 GB total), or `qwen3.8-100k` (15 GB total) to leave room for a second model |
| Quick jobs | 7B or 14B coder | `qwen2.5-coder:7b`, `qwen2.5-coder:14b` |

A 32B coder at Q4 was tried here and dropped; see [Models tried on this machine](#models-tried-on-this-machine).

Going from IQ3_S to Q4_K_M on a model you already run is often a better use
of new VRAM than a bigger model at a worse quant.

## Models tried on this machine

Measured on the 8 GB + 16 GB pair with `OLLAMA_SCHED_SPREAD=1`, `OLLAMA_FLASH_ATTENTION=1`,
`OLLAMA_KV_CACHE_TYPE=q4_0` and `OLLAMA_NUM_PARALLEL=1` unless stated. Numbers come from
[`ollama-multi-gpu.md`](ollama-multi-gpu.md); "not measured" means installed but never benchmarked.

| Model | Context | GPU offload | Generation | Verdict |
|---|---|---|---|---|
| `qwen3.8-216k` (27B, IQ3_S, `qwen35` arch) | 216,000 | 100% | 25.6 tok/s | **Kept.** Main model for long-context work. |
| `qwen3.8-100k` (same weights, `num_ctx 100000`) | 100,000 | 100% (15 GB, KV 1.76 GB) | not measured | **Kept.** Frees ~4 GB for a second model. The `qwen35` arch ignores `OLLAMA_NUM_PARALLEL`, so it is not a way to get two chats on one model. |
| `qwen2.5-coder:14b` | 32,768 | 100% | 37-38 tok/s | **Kept.** Fast coder for short tasks. |
| `qwen2.5-coder:7b` | 32,768 | 100% | 65.9 tok/s | **Kept.** Fastest coder; also fits the internal card alone. |
| `qwen3.5:9b` | 262,144 | 100% | 53.4 tok/s | **Kept.** Long context on a small budget. |
| `nomic-embed-text` | n/a | 100% | 768 dims in ~2.0 s | **Kept.** Embeddings. |
| `qwen2.5-coder:32b` (Q4) | 32,768 | 92% | 13.2 tok/s | **Removed 2026-09-23.** See below. |
| `MHKetbi/Qwen2.5-Coder-32B-Instruct:q4_K_S` | 32,768 (manifest asks for 131,072) | 85% with `OLLAMA_NUM_PARALLEL=2` | not measured | **Removed 2026-09-23.** Duplicate of the above. |
| `deepseek-r1:8b`, `DeepSeek-Qwen3-8B`, `gemma3:4b`, `logicbeat/qwen3.8-27B_GSQ_RCO` | | | not measured | Installed, not evaluated. |

**Why the 32B coder went.** Qwen2.5-Coder-32B was trained for 32,768 tokens; Ollama loads
it at 32,768 whatever the manifest's `num_ctx` says, and offers no way to switch on YaRN
scaling for a GGUF that does not have it built in. Even at 32k with a `q4_0` KV cache it
only reaches 92% offload, at half the speed of the 27B. At 128k the KV cache alone would be
~9 GB on top of ~18 GB of weights, well over 24 GB. Coding tools fill context fast (system
prompt, files read, conversation), so a 32k ceiling hurts more than any code-quality edge
the specialist model might have. That quality edge was never measured here.

## Model families worth trying

Current as of September 2026 and **not benchmarked here** — verify before
trusting, and prefer whatever your own `ollama-tools bench` numbers say.

- **Coding**: Qwen2.5-Coder (1.5B/7B/14B/32B) is the reliable spread across every tier here. DeepSeek-Coder-V2-Lite and Codestral are the usual alternatives to compare against.
- **General / reasoning**: the Qwen3 line, Llama 3.x, Gemma 3, Mistral Small. Reasoning models (`deepseek-r1` and kin) spend many more tokens per answer, so judge them on time-to-answer, not tokens/sec.
- **Embeddings**: `nomic-embed-text`, `mxbai-embed-large`, `bge-m3`. Small enough to ignore in VRAM planning, and a different job from chat — do not use a chat model for retrieval.
- **Reranking**: `bge-reranker` family, once retrieval exists and precision matters more than recall.
- **Vision**: Qwen2.5-VL, Llama 3.2 Vision, if anything needs to read screenshots or diagrams.

## Checking before you load

```powershell
ollama-tools fit qwen3.8-216k --strategy proportional     # check only, loads nothing
ollama-tools start qwen3.8-216k --strategy proportional   # fit-checked launch
```

`qwen3.8-216k` is a locally rebuilt tag (`ollama create` with `num_ctx 216000`
`FROM logicbeat/qwen3.8-27B_GSQ_RCO`, the base pull), not something `ollama pull` will fetch by
that name. If it's missing on a machine, that means the tag was never built
there, not that the model failed to fit — see
[`ollama-multi-gpu.md`](ollama-multi-gpu.md) for how the context ceiling was
measured and how to rebuild it (and re-bisect for a different VRAM budget
before assuming 216,000 carries over).
