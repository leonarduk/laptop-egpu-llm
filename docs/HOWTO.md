# How to build it: a laptop plus an eGPU for local LLMs

The step-by-step guide to the build in the [Medium article](https://medium.com/@steveleonard11/how-i-tripled-the-graphics-memory-on-my-laptop-to-get-a-local-llm-with-a-200k-context-window-498c60591631) (source: [`article/medium-draft.md`](../article/medium-draft.md)): a
laptop's own NVIDIA GPU and a desktop NVIDIA card in a USB4 enclosure, both working at once
under Windows 11, running Ollama models across both.

It is written for the machine it was built on (below), but every step says what to check
so you can adapt it. For the evidence behind each step, follow the links to the reference
docs.

| Part | This build |
|---|---|
| Laptop | Lenovo Legion 5 15IAX10, RTX 5070 Laptop GPU 8 GB, 31.4 GB RAM, Windows 11 Pro |
| Enclosure | Razer Core X V2, USB4 |
| eGPU | NVIDIA RTX 5060 Ti 16 GB |
| Result | 23.8 GB usable VRAM across both cards |

**Is this for you?** It is a hobbyist project. Expect driver trouble. If you are buying
hardware from scratch purely for local LLMs, a machine with unified memory may be simpler.

---

## 1. Check your laptop can do it

- **A USB4 or Thunderbolt 4/5 port with PCIe tunnelling.** Not every USB-C port has it.
  Check the laptop's spec sheet for USB4 / Thunderbolt on the specific port.
- **An NVIDIA GPU in the laptop**, if you want to use both cards together. Two NVIDIA cards
  share one driver, which is what this guide sorts out. Mixing vendors is a different
  problem and is not covered here.
- **Administrator rights**, and your **BitLocker recovery key** if the system drive is
  encrypted. Nothing below should trigger a recovery prompt, but changing BIOS settings can.
  See [`bitlocker-notes.md`](bitlocker-notes.md).

## 2. Parts

- **Enclosure** with enough power for your card (the Razer Core X V2 powers a 5060 Ti with
  room to spare).
- **Desktop NVIDIA card.** VRAM is what matters for LLMs; 16 GB is the sweet spot for price.
- **A proper USB4 / Thunderbolt cable** (40 Gbps rated). A USB-C cable that carries only
  power and USB 2.0 looks identical and leaves the enclosure completely invisible to Windows.
  Use the one that came with the enclosure.

## 3. Prepare Windows before plugging anything in

**Record the starting state.** Reinstalling drivers destroys the evidence you need if
something goes wrong:

```powershell
.\diagnostics\Get-GpuState.ps1 -OutFile before.txt
```

**Stop Windows Update installing GPU drivers.** Otherwise, the first time it sees the
desktop card it installs a *desktop* driver at a different version from your laptop's
driver, and only one card will work (step 6 explains why).

```powershell
.\diagnostics\Disable-WindowsUpdateDrivers.ps1            # show the current state
.\diagnostics\Disable-WindowsUpdateDrivers.ps1 -Apply     # elevated: stop driver updates
```

It sets the Group Policy *Do not include drivers with Windows Updates*
(`ExcludeWUDriversInQualityUpdate` = 1, which also works on Home editions) and the device
installation setting to never fetch drivers from Windows Update. `-Undo` reverts both.
On this machine, Windows Update installed the mismatched desktop driver 16 minutes after
the eGPU was first detected.

## 4. Connect the hardware

1. Fit the card in the enclosure, connect the enclosure's power, switch it on.
2. Connect the enclosure to the laptop's USB4 port with the proper cable.
3. **Restart** the laptop with the enclosure attached. Use **Restart, not Shut down**: with
   Fast Startup on, Shut down is a hibernate, and only Restart gives the firmware a fresh
   look at the new card.

**Always boot with the enclosure attached.** Plugging it in after boot gives the card
**Code 12** ("cannot find enough free resources"): a 16 GB card needs a large block of
address space, and the firmware has already handed it out. See
[`device-error-codes.md`](device-error-codes.md).

**Check the enclosure is on the bus:**

```powershell
Get-PnpDevice | Where-Object { $_.FriendlyName -match "USB4|Thunderbolt|Razer" } |
  Select-Object FriendlyName, Status
```

If the enclosure's router shows `Unknown` and nothing appears in the event log when you plug
in, Windows cannot see the enclosure at all: check its power, then the cable, then the port.

## 5. Check both cards and their driver versions

```powershell
.\diagnostics\Test-DriverConflict.ps1
```

| Exit code | Meaning | Next |
|---|---|---|
| 0 | Both NVIDIA cards on the same driver version | Skip to step 7 |
| 1 | Two different versions: the conflict | Step 6 |
| 2 | Fewer than two NVIDIA cards found | Back to step 4 |

`Get-GpuState.ps1` shows the detail: each card's raw error code and bound driver, with the
NVIDIA version decoded (Windows shows `32.0.16.1692`; NVIDIA calls it 616.92).

## 6. Put both cards on one driver version

**Why:** both NVIDIA driver packages install the same kernel driver, `nvlddmkm.sys`, and
Windows loads only one copy. Whichever card starts first loads its version; the other card
fails with **Code 31**, whose text ends "…the configuration parameters to the driver are
incorrect." The failure moves between cards depending on which boots first.

**Do not use NVIDIA's installer with "clean install" while a card is running.** It removes
the working driver first, then cannot replace the locked driver file, leaving the laptop
card with no driver. With the eGPU unplugged, the desktop installer refuses to run at all.
The full log trail is in [`driver-fix-walkthrough.md`](driver-fix-walkthrough.md).

Install with `pnputil` instead. **The short way:** `Update-NvidiaDriver.ps1` does steps
1–5 for you. It downloads the version pinned in
[`diagnostics/nvidia-driver.json`](../diagnostics/nvidia-driver.json), checks NVIDIA's
signature, picks the INF for each of your GPUs (including an unplugged eGPU) and installs
them:

```powershell
.\diagnostics\Update-NvidiaDriver.ps1 -DownloadOnly   # any time; the cards can be running
.\diagnostics\Update-NvidiaDriver.ps1 -Plan           # which INF for which GPU
.\diagnostics\Update-NvidiaDriver.ps1                 # elevated, cards released (step 4)
```

It keeps the installer in `%LOCALAPPDATA%\nvidia-driver\<version>`, so the exact version
stays on the machine. It lists leftover NVIDIA packages at the end but does not remove
them; do that as in step 7. The long way, by hand:

1. **Download one current driver version.** The desktop package (e.g.
   `616.92-desktop-win10-win11-64bit-international-dch-whql.exe`) also contains the OEM
   notebook INFs.
2. **Extract it** with Windows' built-in `tar` (the installer is a 7-Zip archive):

   ```powershell
   mkdir C:\temp\extract; cd C:\temp\extract
   tar.exe -xf "616.92-desktop-win10-win11-64bit-international-dch-whql.exe" "Display.Driver"
   ```

3. **Find the INFs for your two device IDs.** Get the IDs (`DEV_xxxx`) from
   `Get-GpuState.ps1`, then:

   ```powershell
   Get-ChildItem .\Display.Driver -Filter *.inf |
     Select-String -Pattern "DEV_2D18" -SimpleMatch -List | ForEach-Object { $_.Filename }
   ```

   On this build: `nvlti.inf` (Lenovo notebook, `DEV_2D18`) and `nv_dispi.inf` (desktop,
   `DEV_2D04`). Your laptop maker has its own notebook INF (`nvdmi.inf` Dell, `nvhqi.inf` HP
   and so on).
4. **Unload the driver.** Unplug the eGPU, and disable the laptop card in Device Manager if
   it is still running. Confirm:

   ```powershell
   (Get-Service nvlddmkm -ErrorAction SilentlyContinue).Status   # must be Stopped
   ```

5. **Install both INFs**, from an elevated PowerShell:

   ```powershell
   .\diagnostics\Install-NvidiaDriver.ps1 -DisplayDriverPath C:\temp\extract\Display.Driver `
     -Inf nvlti.inf, nv_dispi.inf
   ```

   Each INF can take several minutes; it has not hung. The desktop INF binds to the eGPU
   even while it is unplugged. The script stops if an install fails. Add `-WhatIf` first
   to see what it would do.
6. Re-enable the laptop card, plug the enclosure back in and **Restart**.
7. **Only once both cards work on the new version**, remove the old mismatched package by
   its *published* name (`oemNNN.inf`, from `Get-GpuState.ps1`), elevated:

   ```powershell
   pnputil /enum-drivers | Select-String -Pattern "oem268.inf" -Context 0,4   # check it is the old NVIDIA one
   pnputil /delete-driver oem268.inf /uninstall
   ```

   Doing this as a separate step, after checking both cards, means a bad install can never
   leave a card with no driver. (`Install-NvidiaDriver.ps1 -RemoveStaleInf oem268.inf` does
   the same, only after the installs succeed, and checks the package is an NVIDIA display
   driver and asks for confirmation first.)
8. **Verify:**

   ```powershell
   nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
   .\diagnostics\Test-DriverConflict.ps1    # expect exit code 0
   ```

## 7. Install and configure Ollama

Install Ollama from ollama.com, then set these as **User** environment variables:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_SCHED_SPREAD", "1", "User")      # use both cards
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")   # needed for the next one
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q4_0", "User")  # quarter-size KV cache
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")      # one KV cache per model
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\OllamaModels", "User")  # optional: model store
```

What each one is worth on this machine, with measurements, is in
[`ollama-multi-gpu.md`](ollama-multi-gpu.md). In short: without `OLLAMA_SCHED_SPREAD`,
Ollama puts a model on one card and spills the rest to the CPU (14 → 38 tok/s on a 14B
coder when turned on); a `q4_0` KV cache is a quarter the size of the default.

**Ollama reads these only when its server starts.** Quit Ollama completely (tray icon →
Quit) and start it again, or sign out and back in. Then **check what the running server
actually got**, from the `server config` line in its log:

```powershell
Select-String -Path "$env:LOCALAPPDATA\Ollama\server.log" -Pattern "server config" |
  Select-Object -Last 1
```

Look for `OLLAMA_SCHED_SPREAD:true`, `OLLAMA_KV_CACHE_TYPE:q4_0`, `OLLAMA_NUM_PARALLEL:1`
and your `OLLAMA_MODELS` path. On this machine the tray app's server reported the default
`C:\Users\<you>\.ollama\models` rather than the `OLLAMA_MODELS` value. If yours does too,
set the model location in the app's settings, or quit the tray app and run `ollama serve`
from a shell that has the variables.

## 8. Get a model and size its context

The main model here is a 27B Qwen 3.x build at IQ3_S quantisation (`qwen35` architecture):

```powershell
ollama pull logicbeat/qwen3.8-27B_GSQ_RCO
```

**Build tags with a fixed context.** Ollama reserves the KV cache for the tag's full
`num_ctx` every time it loads the model, so the context is a standing memory cost:

```powershell
"FROM logicbeat/qwen3.8-27B_GSQ_RCO:latest`nPARAMETER num_ctx 216000" | Set-Content Modelfile
ollama create qwen3.8-216k -f Modelfile

"FROM logicbeat/qwen3.8-27B_GSQ_RCO:latest`nPARAMETER num_ctx 100000" | Set-Content Modelfile
ollama create qwen3.8-100k -f Modelfile
```

| Tag | Total on GPU | Leaves free | Use |
|---|---|---|---|
| `qwen3.8-216k` | ~19.3 GB, 100% GPU | ~4.5 GB | Longest context, one model at a time |
| `qwen3.8-100k` | ~15 GB, 100% GPU | ~9 GB | Room for a second, smaller model |

216,000 was the highest context that stayed 100% on GPU **on this 8 + 16 GB pair with the
settings above**. Measure your own ceiling (from Git Bash, with the server running your
final settings) and watch the offload percentage:

```bash
./diagnostics/measure-context-ceiling.sh logicbeat/qwen3.8-27B_GSQ_RCO:latest 64000 128000 192000 216000 224000
```

**Parallel chats:** the `qwen35` architecture does not support parallel requests in Ollama,
so `OLLAMA_NUM_PARALLEL` above 1 does nothing for it; a second chat waits its turn. On
models that do support it, each extra slot is another full KV cache. Keep it at 1. See
[`ollama-multi-gpu.md`](ollama-multi-gpu.md#ollama_num_parallel-each-parallel-slot-is-a-full-kv-cache).

Smaller models worth having alongside, and the ones tried and dropped (including why a 32B
coder lost out to the 27B), are in [`model-picker.md`](model-picker.md).

## 9. Check it fits, then benchmark

Install the helper CLI (Python 3.9+, standard library only):

```powershell
pip install -e .
ollama-tools fit --strategy proportional                 # what fits in the VRAM attached now?
ollama-tools start qwen3.8-100k --strategy proportional --prompt "Say OK."
ollama-tools ps                                          # expect 100% GPU
ollama-tools bench qwen3.8-100k --repeat 3               # tok/s and offload
```

`fit` sizes a model as its weights plus headroom (20% by default) plus an estimate of the KV
cache for the model's `num_ctx`, using the `OLLAMA_KV_CACHE_TYPE` and `OLLAMA_NUM_PARALLEL`
in *your shell*. It assumes the server was started with the same values, so set them in
both. The estimate is still an estimate: `ollama-tools ps` showing 100% GPU is the real test.

Use `--strategy proportional` on two unequal cards: the default `conservative` budget
assumes an even split (twice the smaller card, ~15.9 GB here) and will refuse models that
Ollama actually places fine with `OLLAMA_SCHED_SPREAD=1`. Exit codes: 0 fits, 1 does not
fit, 2 cannot tell.

Expected on this build: `qwen3.8-216k` 100% GPU at ~25.6 tok/s; `qwen2.5-coder:14b` 100% at
~38 tok/s; `qwen2.5-coder:7b` 100% at ~66 tok/s.

## 10. Living with it

- **Boot with the enclosure attached, and Restart rather than Shut down** after any change.
- **Unload models before restarting Ollama** (`ollama stop <model>`). Killing the server
  with a model loaded can leave an orphaned `llama-server.exe` holding VRAM; see
  [`ollama-multi-gpu.md`](ollama-multi-gpu.md).
- **Switching between long chats is slow.** With one KV cache per model, returning to a
  different long conversation means the model re-reads it first. Keep one long working chat
  per model; start fresh chats for side questions.
- **Monitors:** on this build, three external monitors plus the LLM caused display resets
  and flickering; two were turned off while the model runs.
- **Driver updates:** change the version in `diagnostics/nvidia-driver.json` and run
  `Update-NvidiaDriver.ps1` (step 6). Now and then, check that
  `Disable-WindowsUpdateDrivers.ps1` still reports driver updates off: a Windows feature
  update or an organisation policy can reset it.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Only one NVIDIA card works; which one varies by boot | Two driver versions (Code 31 on the loser) | Step 6 |
| eGPU shows Code 12 | Plugged in after boot | Restart with it attached (step 4) |
| Laptop card shows Code 28 after a driver install | NVIDIA "clean install" removed the driver and could not install the new one | Step 6 with `pnputil` |
| Nothing at all happens when the enclosure is plugged in | Power, cable (USB 2.0-only) or port | Step 4 check |
| Model loads but is slow; `ollama ps` shows CPU/GPU split | `OLLAMA_SCHED_SPREAD` not set in the running server, or the context is too large | Step 7 log check; smaller `num_ctx` (step 8) |
| `ollama list` is missing your models after a restart | Server started with a different model store | Step 7 log check (`OLLAMA_MODELS`) |
| VRAM in use but `ollama ps` shows nothing | Orphaned `llama-server.exe` | [`ollama-multi-gpu.md`](ollama-multi-gpu.md) section 4 |
| Display flicker or resets while a model runs | Too many monitors on the same GPUs | Disconnect monitors while running models |
