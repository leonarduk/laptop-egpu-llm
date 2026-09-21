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

```powershell
# Full diagnostic dump - run this first
.\diagnostics\Get-GpuState.ps1

# Just check for the driver version collision
.\diagnostics\Test-DriverConflict.ps1

# Before loading anything: does it fit in the VRAM actually attached?
.\diagnostics\Test-ModelFits.ps1                             # survey everything pulled
.\diagnostics\Test-ModelFits.ps1 -Model qwen3.8-64k:latest   # exit 1 if it will not fit
```

With the eGPU detached, a 27B model asks for ~11.3 GB against the internal card's 7.93 GB, spills into system memory and hangs the machine hard enough to need a reboot. `Test-ModelFits.ps1` answers that beforehand and exits non-zero rather than letting anything allocate. It defaults to the conservative even-split ceiling from point 3 above, so it will not tell you 23.8 GB is available when your runtime can only reach 15.9 GB.

## Contents

- [`diagnostics/`](diagnostics) - PowerShell scripts for inspecting and fixing GPU state
- [`ollama/`](ollama) - fit-checked wrappers around Ollama, so a model that will not fit never gets loaded
- [`docs/model-picker.md`](docs/model-picker.md) - what to run at 7.9 / 15.9 / 23.8 GB, and the sizing arithmetic behind it
- [`docs/device-error-codes.md`](docs/device-error-codes.md) - what Device Manager codes actually mean here
- [`docs/lmstudio-multi-gpu.md`](docs/lmstudio-multi-gpu.md) - making LM Studio use asymmetric GPUs properly
- [`docs/bitlocker-notes.md`](docs/bitlocker-notes.md) - which steps risk a recovery-key prompt
- [`logs/`](logs) - real failure output, for comparison against your own

## Running a model safely

`ollama run` will accept a model far larger than your GPUs can hold. These wrap it so that cannot happen:

```powershell
.\ollama\Start-Model.ps1 -Model qwen2.5-coder:7b     # loads only if it fits
.\ollama\Get-LoadedModels.ps1                        # what is holding VRAM right now
.\ollama\Stop-Model.ps1 -All                         # free it without stopping the server
.\ollama\Measure-ModelSpeed.ps1 -Model qwen2.5-coder:7b -Repeat 3
```

First numbers from `Measure-ModelSpeed.ps1`, **internal card only, eGPU detached**:

| Model | Generation | Prompt eval | GPU offload |
|---|---|---|---|
| `qwen2.5-coder:7b` | 52-56 tok/s | 322 / 2570 tok/s (cold / warm) | 100% |

Generation is memory-bandwidth-bound, prompt eval is compute-bound, and offload is the figure that explains a disappointing rate - anything under 100% means layers are running on CPU.

## Was it worth it?

Benchmarks pending. Honest answer so far: this is a hobbyist project. The hardware works, but getting there took considerably longer than the shopping did.

## Licence

MIT
