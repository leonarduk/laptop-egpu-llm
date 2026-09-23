# laptop-egpu-llm

Running local LLMs on a laptop with an external GPU, using both the internal and external GPU at once for ~24 GB of combined VRAM.

This repo holds the diagnostic scripts and reference notes from getting that working on Windows 11. The narrative write-up lives on Medium (link TBC).

## The build

| Part | Detail |
|---|---|
| Laptop | Lenovo Legion 5 15IAX10, Intel Core Ultra 9 275HX, 31.4 GB RAM |
| Internal GPU | NVIDIA RTX 5070 Laptop GPU, 8 GB (7.93 GB usable) |
| eGPU | NVIDIA RTX 5060 Ti 16 GB (15.90 GB usable) |
| Enclosure | Razer Core X V2, USB4 |
| OS | Windows 11 Pro, BitLocker on C: |
| Runtimes | LM Studio, Ollama (llama.cpp backend) |

**Combined VRAM: 23.83 GB.**

## The three things that will cost you an evening

### 1. Two NVIDIA driver packages means only one GPU works

If you have a laptop NVIDIA GPU and add a desktop NVIDIA GPU, Windows Update will install a *second* NVIDIA driver package for the new card, at a different version. Both packages install the same kernel-mode driver, `nvlddmkm.sys`, and Windows loads exactly one instance of it. Whichever card binds first wins; the other fails with **Code 31**.

Check for it:

```powershell
.\diagnostics\Test-DriverConflict.ps1
```

Fix: install one driver version covering both device IDs, using [`Install-NvidiaDriver.ps1`](diagnostics/Install-NvidiaDriver.ps1).

### 2. Hot-plugging an eGPU gives you Code 12

**Code 12** ("cannot find enough free resources") is PCIe address-space exhaustion. Firmware allocates its MMIO window at POST, and a hot-plugged 16 GB card has nowhere to go.

Fix: cold boot with the enclosure attached. On Windows 11 that means **Restart**, not Shut down. With Fast Startup enabled, "Shut down" is a hybrid hibernate that restores a saved kernel session; only Restart forces a full POST.

### 3. LM Studio's default split wastes your bigger card

With GPUs of different sizes, an **even split caps usable VRAM at twice the smaller card**. An 8 GB + 16 GB pair gives you ~15.9 GB, not 23.8 GB. See [`docs/lmstudio-multi-gpu.md`](docs/lmstudio-multi-gpu.md).

## Quick start

Driver and enclosure diagnosis is Windows-only — `Get-PnpDevice`, `pnputil` and Device Manager error codes have no Ubuntu equivalent, so those stay PowerShell:

```powershell
# Full diagnostic dump - run this first
.\diagnostics\Get-GpuState.ps1

# Just check for the driver version collision
.\diagnostics\Test-DriverConflict.ps1
```

Everything about Ollama and VRAM is Python, and runs the same on Windows and Ubuntu.

**Requires an NVIDIA GPU with `nvidia-smi` on PATH.** There is no AMD, Intel or Apple Silicon path: without a VRAM figure there is no way to tell a model that fits from one that hangs the machine, so unsupported hardware exits 2 — a refusal, by design, rather than a guess.

```bash
pip install -e .

ollama-tools fit                              # survey: what can I run right now?
ollama-tools fit qwen3.8-216k:latest          # exit 1 if it will not fit
ollama-tools fit --env-file ../issue-worm/issue-worm-pro/.env   # what will this *run* load?
```

`qwen3.8-216k` is a locally rebuilt tag, not one `ollama pull` fetches by that
name — see [`docs/ollama-multi-gpu.md`](docs/ollama-multi-gpu.md) for how it
was built and measured.

## Contents

- [`ollama_tools/`](ollama_tools) - cross-platform fit checks and Ollama wrappers (Python, stdlib only)
- [`diagnostics/`](diagnostics) - Windows-only PowerShell for inspecting and fixing GPU/driver state
- [`docs/model-picker.md`](docs/model-picker.md) - what to run at 7.9 / 15.9 / 23.8 GB, and the sizing arithmetic behind it
- [`docs/device-error-codes.md`](docs/device-error-codes.md) - what Device Manager codes actually mean here
- [`docs/driver-fix-walkthrough.md`](docs/driver-fix-walkthrough.md) - the full evidence trail: driver versions, installer logs, event log, `pnputil` output
- [`docs/lmstudio-multi-gpu.md`](docs/lmstudio-multi-gpu.md) - making LM Studio use asymmetric GPUs properly
- [`docs/ollama-multi-gpu.md`](docs/ollama-multi-gpu.md) - making Ollama use asymmetric GPUs properly (`OLLAMA_SCHED_SPREAD`, KV cache quantisation, where the env vars actually have to be set)
- [`docs/bitlocker-notes.md`](docs/bitlocker-notes.md) - which steps risk a recovery-key prompt
- [`logs/`](logs) - real failure output, for comparison against your own

## Running a model safely

With the eGPU detached, a 27B model asks for ~11.3 GB against the internal card's 7.93 GB, spills into system memory and hangs the machine hard enough to need a reboot. `ollama run` will let you do that. These will not:

```bash
ollama-tools start qwen2.5-coder:7b      # loads only if it fits
ollama-tools ps                          # what is holding VRAM, and how much reached the GPU
ollama-tools stop --all                  # free it without stopping the server
ollama-tools bench qwen2.5-coder:7b --repeat 3
ollama-tools coder-model                 # which coder model fits the VRAM attached right now
ollama-tools general-model               # which general-purpose model fits the VRAM attached right now
```

`coder-model` picks from measured results, not advertised size: `qwen2.5-coder:32b`
for both cards, `qwen2.5-coder:7b` for the internal 8 GB card alone,
`qwen2.5-coder:1.5b` at 3 GB, `qwen2.5-coder:0.5b` otherwise (including no
GPU detected at all). See [`ollama_tools/coder_model.py`](ollama_tools/coder_model.py)
for the tier boundaries, and use `ollama_tools.coder_model.get_coder_model()`
directly if another project wants this decision without shelling out.

`general-model` is the same idea for chat/reasoning work: `qwen3.8-216k` for
both cards, `qwen3.5:9b` for the internal 8 GB card alone, `gemma3:4b`
otherwise (including no GPU detected at all). See
[`ollama_tools/general_model.py`](ollama_tools/general_model.py) for the tier
boundaries, and use `ollama_tools.general_model.get_general_model()` directly
if another project wants this decision without shelling out.

Unlike every other subcommand, `coder-model` and `general-model` never refuse
and always exit **0** — they always have a fallback answer, down to "no GPU
at all", so the exit-code contract below does not apply to them.

Exit codes are the contract for every other subcommand, so they can gate a script: **0** fine · **1** does not fit, nothing loaded · **2** the question could not be answered (no `nvidia-smi`, server down, model not pulled), also nothing loaded.

There is deliberately no `--force`. If you believe a model fits because the runtime places layers proportionally, `--strategy proportional` says so in terms the check can act on. Otherwise the default assumes the even-split ceiling from point 3 above, so it will not claim 23.8 GB when your runtime can only reach 15.9 GB.

First numbers from `ollama-tools bench`, **internal card only, eGPU detached**:

| Model | Generation | Prompt eval | GPU offload |
|---|---|---|---|
| `qwen2.5-coder:7b` | 52-56 tok/s | 322 / 2570 tok/s (cold / warm) | 100% |
| `nomic-embed-text` (embedding) | 768 dims in ~2.0s | n/a | 100% |

Embedding models have no `generate` endpoint at all — Ollama answers *"does not support generate"* — so the capability is read from `/api/show` and `/api/embed` used instead, rather than failing confusingly.

Generation is memory-bandwidth-bound, prompt eval is compute-bound, and offload is the figure that explains a disappointing rate - anything under 100% means layers are running on CPU.

## Was it worth it?

Benchmarks pending. Honest answer so far: this is a hobbyist project. The hardware works, but getting there took considerably longer than the shopping did.

## Licence

MIT
