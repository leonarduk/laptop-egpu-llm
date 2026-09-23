# Making Ollama actually use both GPUs

`ollama-tools fit` says a model fits by checking its size (weights plus headroom plus
an estimate of the KV cache for its `num_ctx`) against a VRAM budget: by default the
even-split ceiling, or the sum of free VRAM across both cards with
`--strategy proportional`. That is a budget check, not a guarantee Ollama's own scheduler will spread the model across
both GPUs to reach it. By default it does not, and the failure mode is quiet: the
model loads, `ollama ps` reports it resident, and it is simply slow, with the
shortfall silently pushed onto CPU.

> **Status: confirmed by benchmark on this machine.** Numbers below are from
> `ollama-tools bench`, same model, same prompt, before and after.

## 1. `OLLAMA_SCHED_SPREAD` — without it, the eGPU is not used

Ollama's default scheduler tries to fit a model onto as few GPUs as possible —
in practice, whichever one it picks first. If that GPU cannot hold the whole
model, the remainder spills to **CPU**, not to the second GPU, even when the
second GPU has 16 GB sitting idle.

```
generation 14.2 tok/s   GPU offload: 62% (5.78 GB of 9.27 GB)   -- default
generation 37.9 tok/s   GPU offload: 100% (15.05 GB of 15.05 GB) -- OLLAMA_SCHED_SPREAD=1
```
(`qwen2.5-coder:14b`, identical prompt, identical `--repeat 3`.)

The `--strategy proportional` flag on `ollama-tools fit`/`bench` does **not** change
this — it only changes the tool's own budget arithmetic for the fit check. Real
placement is controlled by the Ollama server process's own environment at the
time it started.

## 2. KV cache quantisation buys back the rest

For a model that still does not fit even split across both GPUs (a 32B at Q4, for
example), the KV cache is usually the difference, not the weights. `OLLAMA_FLASH_ATTENTION`
must be on for KV cache quantisation to apply at all.

```
qwen2.5-coder:32b, 32768 ctx, both GPUs:

default KV cache              GPU offload: 72% (19.34 / 27.04 GB)   generation  7.7-7.9 tok/s
OLLAMA_KV_CACHE_TYPE=q8_0      GPU offload: 83% (19.33 / 23.29 GB)   generation 10.2-10.4 tok/s
OLLAMA_KV_CACHE_TYPE=q4_0      GPU offload: 92% (19.52 / 21.28 GB)   generation 13.2 tok/s
```

`q4_0` is more lossy than `q8_0`; on this machine it was still the difference between
"mostly on CPU" and "almost entirely on GPU", so it is the default here. Drop to
`q8_0` if output quality noticeably suffers and 83% offload is an acceptable trade.

## 3. Where these variables actually have to be set

Ollama runs as a plain per-login background process here, not a Windows service:

- Launched at login by a shortcut in `shell:startup`
  (`...\Start Menu\Programs\Startup\Ollama.lnk`) pointing at
  `...\Programs\Ollama\ollama app.exe`.
- That tray app spawns the real server as a child process: `ollama.exe serve`.
- The server reads its environment **once, at that startup**. A `$env:` set in a
  shell afterwards, or even a registry write via
  `[Environment]::SetEnvironmentVariable(..., "User")`, does **nothing** to an
  already-running server — only new processes launched after the registry write
  pick it up.

Set these as persistent **User** environment variables, then fully quit and
relaunch Ollama (or log out/in) for them to take effect:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_SCHED_SPREAD", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q4_0", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\OllamaModels", "User")
```

A launcher app spawned from an interactive shell (e.g. `Start-Process` from a
terminal) does not reliably pick up a User-scope registry change either, unless
that shell's own process was started after the registry write. The safest way to
verify what a running server actually has is to check its own startup log line
(`level=INFO source=routes.go msg="server config" env=...`), not to assume from
`[Environment]::GetEnvironmentVariable`.

## 4. Killing the wrong process leaves VRAM held by an orphan

`ollama app.exe` / `ollama.exe serve` spawn a further child, `llama-server.exe`,
which is the actual process holding the CUDA allocation for a loaded model. Run
`ollama stop <model>` **before** killing or restarting the server. Killing the
parent (`Stop-Process -Name ollama`) without unloading first leaves
`llama-server.exe` running, orphaned, still holding VRAM — and it does not show up
in `ollama ps` any more, because that command asks the (now-different) server
process, not the orphan.

Check for it directly if `nvidia-smi` shows VRAM in use that nothing accounts for:

```powershell
Get-CimInstance Win32_Process | Where-Object Name -match 'llama-server' |
    Select-Object ProcessId, Name
Stop-Process -Id <id> -Force
```

## What this changes on the tiers in `model-picker.md`

The tiers in [`model-picker.md`](model-picker.md) are budget ceilings — what the
VRAM math allows. Whether a given model actually reaches that ceiling on this
runtime depends on the three settings above. Measured with all three set:

| Model | Context (actual) | GPU offload | Generation |
|---|---|---|---|
| `qwen2.5-coder:7b` | 32,768 (arch default, no `num_ctx` override) | 100% | 65.9 tok/s |
| `qwen2.5-coder:14b` | 32,768 (arch default) | 100% | 37-38 tok/s |
| `qwen2.5-coder:32b` | 32,768 (arch default) | 92% | 13.2 tok/s |
| `qwen3.5:9b` | 262,144 (arch default, no `num_ctx` override) | 100% | 53.4 tok/s |
| `qwen3.8-216k` | 216,000 (`PARAMETER num_ctx`, raised from the original 64,000) | 100% | 25.6 tok/s |

### How far the `qwen3.8-64k` tag (27B, IQ3_S) actually stretches

The original 64,000 `num_ctx` in this tag's Modelfile was a conservative guess, not a
measured ceiling. Bisecting `num_ctx` by hand — measured under
`OLLAMA_SCHED_SPREAD=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q4_0`
(the settings this doc sets as persistent User env vars above; the ceiling below is
only valid for that combination, and **must be re-measured** if any of the three
changes) using the script at
[`diagnostics/measure-context-ceiling.sh`](../diagnostics/measure-context-ceiling.sh),
unloading between each `num_ctx` with `ollama stop` to avoid a stale allocation
skewing the next result, and confirmed with `nvidia-smi` that free VRAM was within
~20 MiB across every run below (7325-7347 MiB free on GPU0 in every case, so this
is not an artifact of some other app's GPU usage shifting between runs):

| `num_ctx` | GPU offload |
|---|---|
| 64,000 (original) | 100% (13.03 / 13.03 GB) |
| 96,000 | 100% (15.11 / 15.11 GB) |
| 128,000 | 100% (16.22 / 16.22 GB) |
| 160,000 | 100% (17.33 / 17.33 GB) |
| 192,000 | 100% (18.45 / 18.45 GB) |
| 200,000 | 100% (18.73 / 18.73 GB) |
| **216,000** | **100% (19.29 / 19.29 GB) — highest confirmed fully-resident value** |
| 224,000 | 95.7% (17.98 / 18.79 GB) |
| 240,000 | 95.7% (18.44 / 19.27 GB) |
| 262,144 (arch max) | 95.7% (19.08 / 19.93 GB) |

**The 224,000 row's *total* (18.79 GB) is smaller than 216,000's total (19.29 GB),
despite the larger context — read that as a warning, not a typo.** Ollama's
`/api/ps` "size" is not `weights + linear(num_ctx)`; it is however much VRAM the
runtime's own fit-to-available-memory logic actually committed to for the
allocation plan it landed on. Below 216k it can commit to a plan that puts
everything on GPU. Above that, it falls back to a *different, smaller* plan that
accepts partial CPU placement — not a bigger version of the same GPU-resident
plan. That is why total footprint can drop while offload also drops: it is
answering "what did the runtime actually build", not "how much would this
context truly cost if fully resident". Take the offload percentage as the signal,
not the size column, and do not assume either one grows monotonically with
`num_ctx` past the ceiling.

The tag's manifest was rebuilt with `num_ctx 216000` (`ollama create qwen3.8-64k:latest -f
Modelfile`), which reclaims the difference for free: same 100% offload and near-identical
speed as the 64k version, 3.4x the usable context. The tag was then renamed to match —
`ollama cp qwen3.8-64k:latest qwen3.8-216k:latest && ollama rm qwen3.8-64k:latest` — since a
name that no longer matches its actual context is worse than no name at all.

**The old `qwen3.8-64k:latest` tag no longer exists on this machine.** Anyone
following an older copy of this repo's README or `model-picker.md` that still
names it will get "model not found" from Ollama, not a fit check — pull the base
model again and rebuild the tag with a `num_ctx` you've bisected for your own
hardware, rather than assuming 216,000 carries over (it was measured on this
specific 8 GB + 16 GB pair; a different VRAM budget needs its own bisection).

Re-run this bisection after any `OLLAMA_KV_CACHE_TYPE` change — it directly changes
the KV cache's bytes-per-token, so the 100%-offload ceiling moves with it. It also
directly changes **every model's** VRAM footprint, not just this one: a model
reserves KV cache for its manifest's `num_ctx` on every load regardless of the
actual prompt length, so a large `num_ctx` is a standing cost, not a peak one.

Two different things determine "context" here, and they are easy to conflate:
`ollama show <model>` reports the **architecture's** maximum supported context
(e.g. `qwen35.context_length: 262144`), but a model's own Modelfile can override
the actually-used context with `PARAMETER num_ctx`. Check `ollama show <model>
--parameters`, not just `--modelfile`'s base architecture line, to know what a
given tag actually runs at. `OLLAMA_CONTEXT_LENGTH=0` (the default, unset) means
"use whichever of those two applies" — it does not mean "no limit".

## `OLLAMA_NUM_PARALLEL`: each parallel slot is a full KV cache

Measured 2026-09-23 with `OLLAMA_NUM_PARALLEL=2` plus the three settings above. Check the
server log's `n_slots` line to see what a load actually got; the env var is a request, not a
guarantee.

| Model | Result |
|---|---|
| `qwen3.8-100k` (27B, `qwen35` arch, `num_ctx 100000`) | Ollama logs `model architecture does not currently support parallel requests architecture=qwen35` and loads with `n_slots = 1`. 100% GPU, 15 GB total, KV cache 1.76 GB. A second request queues. |
| `MHKetbi/Qwen2.5-Coder-32B-Instruct:q4_K_S` (`qwen2` arch) | `n_slots = 2`, 32,768 context per slot, KV cache 4.6 GB (2/2 seqs). 24 GB total, **15%/85% CPU/GPU**. |

Two consequences:

- For the hybrid `qwen35` models, parallel chats are not available at all, so shrinking
  `num_ctx` to "make room for two chats" only frees VRAM for a second model.
- For models that do support it, `OLLAMA_NUM_PARALLEL=2` doubles the KV cache and can push
  a model that fitted off the GPU. Leave it at 1 unless you genuinely serve concurrent requests.

That 32B tag's manifest sets `num_ctx 131072`, but its architecture's trained context is
32,768 and Ollama loaded it at 32,768 per slot.

**If `OLLAMA_MODELS` seems ignored:** the Windows tray app (`ollama app.exe`) starts its own
server using its own model-location setting (`C:\Users\<you>\.ollama\models` by default),
not the `OLLAMA_MODELS` user variable. Start `ollama serve` from a shell that has the
variables set, or change the location in the app's settings.
