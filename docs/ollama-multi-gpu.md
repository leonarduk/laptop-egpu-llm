# Making Ollama actually use both GPUs

`ollama-tools fit` says a model fits by summing free VRAM across both cards. That is
a budget check, not a guarantee Ollama's own scheduler will spread the model across
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
measured ceiling. Bisecting `num_ctx` by hand (`curl .../api/generate` with an explicit
`num_ctx` override, checking `/api/ps` for `size_vram == size` between each run, unloading
with `ollama stop` between tests to avoid stale allocations skewing the result):

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
| 262,144 (arch max) | 95.7% (19.08 / 19.93 GB) |

The tag's manifest was rebuilt with `num_ctx 216000` (`ollama create qwen3.8-64k:latest -f
Modelfile`), which reclaims the difference for free: same 100% offload and near-identical
speed as the 64k version, 3.4x the usable context. The tag was then renamed to match —
`ollama cp qwen3.8-64k:latest qwen3.8-216k:latest && ollama rm qwen3.8-64k:latest` — since a
name that no longer matches its actual context is worse than no name at all. Re-run this
bisection after any `OLLAMA_KV_CACHE_TYPE` change — it directly changes the KV cache's
bytes-per-token, so the 100%-offload ceiling moves with it. It also directly changes
**every model's** VRAM footprint, not just this one: a model reserves KV cache for its
manifest's `num_ctx` on every load regardless of the actual prompt length, so a large
`num_ctx` is a standing cost, not a peak one.

Two different things determine "context" here, and they are easy to conflate:
`ollama show <model>` reports the **architecture's** maximum supported context
(e.g. `qwen35.context_length: 262144`), but a model's own Modelfile can override
the actually-used context with `PARAMETER num_ctx`. Check `ollama show <model>
--parameters`, not just `--modelfile`'s base architecture line, to know what a
given tag actually runs at. `OLLAMA_CONTEXT_LENGTH=0` (the default, unset) means
"use whichever of those two applies" — it does not mean "no limit".
