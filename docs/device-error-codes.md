# Device Manager error codes, and what they actually mean here

Device Manager's friendly text hides the information you need. Get the raw constant:

```powershell
Get-PnpDevice -Class Display | Where-Object { $_.Present -eq $true } |
  Select-Object FriendlyName, Status, ConfigManagerErrorCode | Format-List
```

## The codes you will hit

| Constant | Code | Meaning | What it usually is here |
|---|---|---|---|
| `CM_PROB_NORMAL_CONFLICT` | 12 | Cannot find enough free resources | PCIe/MMIO address space exhausted. Hot-plug problem. |
| `CM_PROB_REINSTALL` | 18 | Reinstall the drivers for this device | Transient state during a failed install. |
| `CM_PROB_FAILED_INSTALL` | 28 | Drivers are not installed | A `-clean` install removed the driver and did not replace it. |
| `CM_PROB_FAILED_ADD` | 31 | Windows cannot load the required drivers | **Driver version collision.** The real bug. |
| `CM_PROB_PHANTOM` | 45 | Stale record, device not present | Normal when the enclosure is unplugged. Not a failure. |

## Why the distinction matters

**Code 12 and Code 31 are completely different problems.** Collapsing them into "my eGPU
doesn't work" is what sends people down the wrong path.

### Code 12 - a resource problem

> This device cannot find enough free resources that it can use. (Code 12)
> If you want to use this device, you will need to disable one of the other devices on this system.

A 16 GB card with Resizable BAR asks for a large memory aperture. Firmware allocates its
MMIO window at POST, so a card hot-plugged after boot has nowhere to go.

**Fix:** cold boot with the enclosure attached. On Windows 11 use **Restart**, not Shut
down — with Fast Startup enabled, Shut down is a hybrid hibernate that restores a saved
kernel session, and only Restart forces a genuine POST.

If Code 12 survives a proper cold boot, the next lever is firmware: disable Resizable BAR,
or enable Above 4G Decoding. See [bitlocker-notes.md](bitlocker-notes.md) before entering
BIOS setup on an encrypted machine.

### Code 31 - a driver problem

> This device is not working properly because Windows cannot load the drivers required for this device. (Code 31)
> The I/O device is configured incorrectly or the configuration parameters to the driver are incorrect.

That second sentence is what people report as a "wrong parameters" error. It means the
driver could not bind — almost always because another NVIDIA package at a different
version has already loaded `nvlddmkm.sys`.

**Fix:** get both GPUs onto one driver version. See
[`Test-DriverConflict.ps1`](../diagnostics/Test-DriverConflict.ps1).

## The diagnostic that gives it away

If the failure **swaps between cards depending on boot order**, it is a binding problem,
not a resource problem. Resource contention doesn't ping-pong; driver version collisions do.

In my case:

- Hot-plugged: eGPU got Code 12, internal GPU got Code 31
- After a cold boot: eGPU worked, internal GPU got Code 31

The seat-swap was the clue that the resource story was a red herring.

## When nothing appears at all

If plugging the enclosure in produces **no device event whatsoever** — not an error, nothing
in the System log — then Windows isn't detecting a physical connection. Check the enclosure's
USB4 router status:

```powershell
Get-PnpDevice | Where-Object { $_.FriendlyName -match "USB4|Thunderbolt|Razer" } |
  Select-Object FriendlyName, Status
```

`Status: Unknown` on the enclosure's router means it is not on the bus. Suspect, in order:
enclosure power, the cable, then the port. A USB-C cable that carries power and USB 2.0 only
is physically identical to a USB4 one and produces exactly this silence.
