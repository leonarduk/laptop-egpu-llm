# How I turn a laptop into a desktop in my pursuit of a usable Local LLM

When I bought my laptop, I thought a graphics card with 8 GB was good. Then along came my experiment with local LLMs and realized, no, not quite. 24 GB seems to be the minimum. The problem with laptops is they are hard to upgrade, or so I thought. In reality, all you need is a massive ugly black box and a big enough desk. This article is how I managed to get the external- and onboard- GPUs playing nice and running a decent LLM.

Why bother? I love Claude Code, but it is expensive, and even on Claude Max I was burning through my tokens too quickly. I had been sending my overflow work to DeepSeek, then I came across the article below about running a decent model locally and wanted to try it myself.

My inspiration was this article: [The 16GB threshold](https://augmentedmind.substack.com/p/the-16gb-threshold), where he shows a specific "budget" 16 GB GPU was capable of something decent. I wanted to do one better, and combine my existing 8 GB GPU with the external 16 GB one, to make something half decent. Spoiler, I eventually got it working.

I will say ahead of time, it was indeed hard to upgrade, and in hindsight, this is for hobbyists only.

## The setup

- **Laptop:** Lenovo Legion 5 15IAX10, 31.4 GB RAM, Windows 11 Pro
- **Internal dGPU:** NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB
- **iGPU:** Intel(R) Graphics
- **eGPU:** NVIDIA GeForce RTX 5060 Ti 16 GB, 16311 MiB
- **Enclosure:** Razer Core X V2 over USB4
- **Software:** Ollama 0.34.2, used through Kun Desktop (DeepSeek's app). I started with LM Studio 0.4.21, but it kept failing.

The goal was roughly 24 GB of combined VRAM, so that big (27B to 32B) models could live on the GPUs instead of spilling into system RAM.

The symptom: I could use the internal GPU or the external one, never both. Along the way the NVIDIA driver would break and I would get an error mentioning "wrong parameters."

## What Windows was actually saying

Device Manager's friendly text hides the information you need. Use the `ConfigManagerErrorCode` instead:

```powershell
Get-PnpDevice -Class Display | Where-Object { $_.Present -eq $true } |
  Select-Object FriendlyName, Status, ConfigManagerErrorCode | Format-List
```

With both GPUs attached, hot-plugged:

```text
FriendlyName           : NVIDIA GeForce RTX 5060 Ti
Status                 : Error
ConfigManagerErrorCode : CM_PROB_NORMAL_CONFLICT     <- Code 12

FriendlyName           : Intel(R) Graphics
Status                 : OK
ConfigManagerErrorCode : CM_PROB_NONE

FriendlyName           : NVIDIA GeForce RTX 5070 Laptop GPU
Status                 : Error
ConfigManagerErrorCode : CM_PROB_FAILED_ADD          <- Code 31
```

`nvidia-smi` returned `Failed to initialize NVML: Not Found`. Both NVIDIA cards were dead.

Two different error codes on two GPUs. That distinction is the entire story. (The full list of codes, and what each one means here, is in the repo linked at the end.)

## Code 12: real, but not the disease

This is PCIe memory-mapped I/O exhaustion. A 16 GB card with Resizable BAR enabled requests a 16 GB BAR1 aperture instead of the legacy 256 MB, and when you hot-plug it, the firmware's MMIO window was already allocated at POST. There is nowhere to put it.

The fix is a cold boot with the enclosure attached. Specifically:

> **Use Restart, not Shut down.** With Fast Startup (hiberboot) enabled, "Shut down" is a hybrid hibernate that restores a saved kernel session. Restart forces a genuine full POST. Counterintuitive, but Restart is your cold boot on Windows 11.

After a restart with the enclosure attached, Code 12 disappeared and the 5060 Ti worked. And the internal 5070 became the broken one, now showing Code 31. The failure had swapped seats.

That swap is the diagnostic. A resource-contention problem does not ping-pong between two devices based on boot order. A binding problem does.

## Code 31: the error that swapped seats

> This device is not working properly because Windows cannot load the drivers required for this device. (Code 31) The I/O device is configured incorrectly or the configuration parameters to the driver are incorrect.

That second line was my mysterious "wrong parameters" error. Note that it appeared on the internal GPU. I had spent weeks assuming it was the eGPU failing, and looking in the wrong place.

Now check what is actually installed:

```powershell
Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" |
  Select-Object DeviceName, DriverVersion, DriverDate, InfName | Format-List
```

```text
DeviceName    : NVIDIA GeForce RTX 5060 Ti
DriverVersion : 32.0.15.9186
DriverDate    : 20/01/2026
InfName       : oem268.inf        (original: nv_dispig.inf - desktop)

DeviceName    : NVIDIA GeForce RTX 5070 Laptop GPU
DriverVersion : 32.0.16.1692
DriverDate    : 09/04/2026
InfName       : oem277.inf        (original: nvlti.inf - notebook)
```

Two NVIDIA driver packages. Two different versions.

## One kernel, two drivers

Both packages install the same kernel-mode driver, `nvlddmkm.sys`. Windows loads exactly one instance of it. It cannot be 591.86 and 616.92 at the same time. Whichever GPU binds first loads its version; the second card is handed a driver built for a different version and fails with Code 31.

That is the complete mechanism behind "Windows only lets me use one GPU at a time." It is not a Windows policy, not a MUX switch or Advanced Optimus issue, not eGPU bandwidth, and not a corrupted driver store. Windows runs multi-GPU NVIDIA configurations fine. It just cannot run two versions of one kernel driver.

(Inference, not something I can prove from logs: the 591.86 desktop package almost certainly arrived via Windows Update the first time the 5060 Ti was detected. Windows Update ships older WHQL packages, and 591.86 was eight months older than the notebook driver already installed.)

## One driver for both cards

The requirement: both GPUs on the same driver version. Either one package covering both device IDs, or two packages at identical versions.

Download the current package. Note NVIDIA ships separate desktop and notebook builds. I took the desktop one, 945 MB.

Inspect it before installing. NVIDIA's `.exe` is a 7-Zip self-extracting archive, and Windows' bundled `tar.exe` (bsdtar/libarchive) opens it without installing 7-Zip:

```powershell
cd C:\temp\extract
tar.exe -xf "616.92-desktop.exe" "Display.Driver"
```

That produced 266 files, 2776.9 MB, including `nvlddmkm.sys`. The "desktop" package contains 43 INFs including all the OEM notebook variants: `nvlti.inf` (Lenovo), `nvdmi.inf` (Dell), `nvhqi.inf` (HP), and so on.

One package, both cards, same version. Exactly what is needed.

## The installer that deleted my driver

I tried the obvious thing first, `setup.exe -s -clean -noreboot`, and it failed in both directions.

With the eGPU connected, the installer ran for eight minutes, then exited. Checking `C:\Windows\INF\setupapi.dev.log`:

```text
inf:  Service 'nvlddmkm' still in use by 2 sources.
cpy:  Skipping isolated file 'nvlddmkm.sys'.
idb:  {Unpublish Driver Package: C:\Windows\System32\DriverStore\FileRepository\
      nvlti.inf_amd64_c284234d5322c92e\nvlti.inf}
idb:  Unregistered driver package 'nvlti.inf_amd64_c284234d5322c92e' from 'oem277.inf'.
```

Read that sequence carefully. `-clean` removed the working notebook package first, then hit `nvlddmkm.sys` locked by the live 5060 Ti and skipped the file copy. It deleted a working driver and installed nothing. The 5070 dropped to Code 28 ("The drivers for this device are not installed") and vanished from the Display class into Other devices.

With the eGPU disconnected, the desktop installer refused outright, exit code `-469762016` (`0xE4000020`), because no desktop GeForce was present to match.

So: connected means the driver is locked, disconnected means the installer will not run. That is the trap.

The lesson: never run `-clean` while a GPU is holding `nvlddmkm`. The removal happens before the install, and the install can fail.

## Going around the installer

`pnputil` has no hardware-presence check and no installer state machine. Run it with `nvlddmkm` unloaded: eGPU disconnected and the other card driverless, or the other card disabled in Device Manager. Verify first:

```powershell
(Get-Service nvlddmkm -ErrorAction SilentlyContinue).Status
# Stopped   <- this is the condition you need
```

Then, elevated:

```powershell
$dd = "C:\temp\extract\Display.Driver"
pnputil.exe /add-driver "$dd\nvlti.inf"    /install
pnputil.exe /add-driver "$dd\nv_dispi.inf" /install
pnputil.exe /delete-driver oem268.inf /uninstall
```

Two things worth noting. First, timing: `nvlti.inf` took 5 minutes 34 seconds, `nv_dispi.inf` took 33 seconds. These are 2.8 GB of files going into the driver store, so do not assume it has hung. Second, the `nv_dispi.inf` line bound to `DEV_2D04` even with the enclosure disconnected. `pnputil` pre-binds to the offline device record, so the card gets the right version the instant it appears.

## It was the cable

Connected the enclosure, restarted, got nothing. No GPU, no error code, no device. The event log after a marker timestamp showed only DistributedCOM and HttpService entries, no PnP events whatsoever.

Zero device events on plug-in means Windows is not detecting a physical connection at all, which is different from a link that comes up and fails. The tell is the enclosure's router status:

```text
USB4(TM) Host Router (Microsoft)        OK        <- laptop controller, fine
USB4 Router (2.0), Razer - Core X V2    Unknown   <- enclosure not enumerated
```

Wrong cable. A USB-C cable carrying power and USB 2.0 only is physically identical to a USB4 cable and produces exactly this silence.

One restart later:

```text
index, name, driver_version, memory.total [MiB], pci.bus_id
0, NVIDIA GeForce RTX 5070 Laptop GPU, 616.92,  8151 MiB, 00000000:02:00.0
1, NVIDIA GeForce RTX 5060 Ti,         616.92, 16311 MiB, 00000000:06:00.0
```

24,462 MiB combined. Different INFs, same driver version, one `nvlddmkm`, both bind.

## The diagnostic checklist

If exactly one NVIDIA GPU works at a time, run this before touching anything:

```powershell
Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" |
  Select-Object DeviceName, DriverVersion, InfName | Format-List
```

If you see two different `DriverVersion` values across two NVIDIA cards, that is your bug, and no amount of DDU, BIOS tweaking or cable-swapping will fix it.

## Preventing the recurrence

**Disable driver delivery via Windows Update**, or it will reinstall a mismatched package. Group Policy:

> Computer Configuration → Administrative Templates → Windows Components → Windows Update → Do not include drivers with Windows Update → Enabled

**Always cold boot with the enclosure attached.** Hot-plugging reliably produces Code 12 on a laptop. Architectural, not a bug.

**Update drivers with `nvlddmkm` unloaded.** Disconnect the eGPU or disable the other card in Device Manager first, and prefer `pnputil /add-driver <inf> /install` over NVIDIA's installer.

## Is an eGPU worth it for inference?

llama.cpp-based runtimes (LM Studio, Ollama) split models by layer, so only activations cross the link, not weights. The narrow USB4 connection matters far less than people assume. Fitting the model entirely into VRAM matters enormously. On that trade, an eGPU is a good deal, once Windows will actually let you use both cards.

## What 24 GB actually buys

Getting Windows to see both cards turned out to be only half the battle. Left to itself, Ollama still put a 14B coding model mostly on one card and spilled the rest into system RAM: 62% on GPU, 14 tokens a second. One environment variable, `OLLAMA_SCHED_SPREAD=1`, spread it across both cards: 100% on GPU, 38 tokens a second.

Two more settings, flash attention and a 4-bit KV cache (`OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q4_0`), and I'm running a 27B model with a 216,000-token context entirely on GPU at about 25 tokens a second. On a laptop. Next to a big ugly black box.

I didn't quite hit my original goal. A 32B coding model at a 32k context still only gets 92% onto the GPUs, at 13 tokens a second. Close, but not all the way.

There was one more cost. I had to unplug two of my three external monitors while the LLM runs. With all three connected I got constant display resets and flickering.

The full command-by-command trail, the diagnostic scripts and the benchmark numbers are all in the repo: [laptop-egpu-llm on GitHub](https://github.com/leonarduk/laptop-egpu-llm).

<!-- Image: screenshot showing both GPUs detected (current Medium draft uses an LM Studio screenshot; consider a Kun Desktop one). -->
