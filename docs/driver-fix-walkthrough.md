# Driver fix walkthrough: the full evidence trail

The step-by-step record of getting both NVIDIA GPUs working at once, with the commands and
the output they produced on this machine. For what each Device Manager code means, see
[`device-error-codes.md`](device-error-codes.md); for the scripts that automate the fix, see
[`Test-DriverConflict.ps1`](../diagnostics/Test-DriverConflict.ps1) and
[`Install-NvidiaDriver.ps1`](../diagnostics/Install-NvidiaDriver.ps1).

## The hardware, by device ID

| Device | Hardware ID |
|---|---|
| RTX 5070 Laptop GPU (8151 MiB) | `PCI\VEN_10DE&DEV_2D18&SUBSYS_3E3817AA&REV_A1`, bus 2 |
| RTX 5060 Ti 16 GB (16311 MiB) | `PCI\VEN_10DE&DEV_2D04&SUBSYS_8A071043&REV_A1` |
| Intel Graphics (iGPU) | `PCI\VEN_8086&DEV_7D67&SUBSYS_3E3817AA&REV_06`, driver 32.0.101.6737 |
| Razer Core X V2 | `USB4\VID_8087&PID_5786` |
| Intel USB4 host router | `PCI\VEN_8086&DEV_7EC2&SUBSYS_3E1717AA&REV_10` |

Software at the time: LM Studio 0.4.21, Ollama 0.34.2.

## 1. The starting state

Both GPUs attached, enclosure hot-plugged:

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

`nvidia-smi` returned `Failed to initialize NVML: Not Found`: neither NVIDIA card was usable.

After a Restart with the enclosure attached, Code 12 cleared and the 5060 Ti worked, but the
5070 stayed on Code 31. The failure had moved from one card to the other.

## 2. Two driver packages, two versions

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

### Decoding NVIDIA's WDDM version strings

Windows shows `32.0.16.1692`; NVIDIA calls it 616.92. Concatenate the last two groups, take
the final five digits, and put a decimal point before the last two:

```text
32.0.16.1692  ->  16 + 1692 = "161692"  ->  "61692"  ->  616.92
32.0.15.9186  ->  15 + 9186 = "159186"  ->  "59186"  ->  591.86
```

### Watching the loaded version follow the winner

Both packages ship `nvlddmkm.sys`, and only one instance loads. `nvidia-smi` reports
whichever version won the bind:

```text
# internal GPU only, eGPU unplugged
NVIDIA-SMI 616.92    KMD Version: 616.92    CUDA UMD Version: 13.4

# eGPU attached and winning the bind
index, name, driver_version, memory.total [MiB], pci.bus_id
0, NVIDIA GeForce RTX 5060 Ti, 591.86, 16311 MiB, 00000000:06:00.0
```

Probable source of 591.86 (inferred, not proven from logs): Windows Update, the first time
the 5060 Ti was detected. It was eight months older than the notebook driver already installed.

## 3. Event log evidence

The `.Message` property comes back empty for `nvlddmkm` events because the provider's message
resource does not resolve. Use `.ToXml()`:

```powershell
$e = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'} -MaxEvents 5
foreach ($x in $e) { $x.ToXml() }
```

```xml
<EventID Qualifiers='49322'>14</EventID>
<Data>\Device\Video8</Data>
<Data>GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)</Data>

<EventID Qualifiers='0'>153</EventID>
<Data>\Device\Video8</Data>
<Data>Error occurred on GPUID: 200</Data>
```

`Microsoft-Windows-Kernel-PnP` Event ID 219 also fired in tight pairs, dozens of times:

```xml
<Data Name='DriverName'>PCI\VEN_8086&DEV_AD03&SUBSYS_3E1717AA&REV_01\3&11583659&1&20</Data>
<Data Name='Status'>3221226341</Data>
<Data Name='FailureName'>\Driver\WUDFRd</Data>
```

`3221226341` is `0xC0000365`, which I believe is `STATUS_FAILED_DRIVER_ENTRY`, against the
Intel USB4 host controller.

## 4. Finding one package that covers both cards

NVIDIA ships separate desktop and notebook builds:

```text
https://us.download.nvidia.com/Windows/616.92/616.92-desktop-win10-win11-64bit-international-dch-whql.exe
https://us.download.nvidia.com/Windows/616.92/616.92-notebook-win10-win11-64bit-international-dch-whql.exe
```

The desktop one is 945 MB. It is a 7-Zip self-extracting archive, and Windows' bundled
`tar.exe` (bsdtar/libarchive) opens it without installing anything:

```powershell
cd C:\temp\extract
tar.exe -xf "616.92-desktop.exe" "Display.Driver"
```

That gives 266 files, 2776.9 MB, including `nvlddmkm.sys`. Despite the name, the desktop
package contains 43 INFs, including the OEM notebook variants: `nvlti.inf` (Lenovo),
`nvdmi.inf` (Dell), `nvhqi.inf` (HP) and others.

Check which INFs cover each device ID:

```powershell
Get-ChildItem -Recurse -Filter *.inf |
  Select-String -Pattern "DEV_2D04" -SimpleMatch -List | ForEach-Object { $_.Filename }
```

```text
DEV_2D04 (desktop RTX 5060 Ti):
  nv_dispi.inf, nvaei.inf, nvaki.inf, nvddi.inf, nvhdci.inf, nvlei.inf, nvmdi.inf

DEV_2D18 (RTX 5070 Laptop GPU):
  nvaci.inf, nvaki.inf, nvami.inf, nvbydi.inf, nvcvi.inf, nvdmi.inf, nvgbi.inf,
  nvhmi.inf, nvhqi.inf, nvlti.inf, nvmii.inf, nvqui.inf, nvsmi.inf, nvtfi.inf
```

Confirm the version inside the INF:

```powershell
Select-String -Path ".\Display.Driver\nvlti.inf" -Pattern "^DriverVer"
# DriverVer = 09/04/2026, 32.0.16.1692
```

## 5. How NVIDIA's installer failed

`setup.exe -s -clean -noreboot` with the eGPU **connected** ran for eight minutes, then exited.
From `C:\Windows\INF\setupapi.dev.log`:

```powershell
Get-Content "C:\Windows\INF\setupapi.dev.log" -Tail 4000 |
  Select-String -Pattern "nvlddmkm|nvlti|nv_dispi|Skipping"
```

```text
inf:  Service 'nvlddmkm' still in use by 2 sources.
cpy:  Skipping isolated file 'nvlddmkm.sys'.
idb:  {Unpublish Driver Package: C:\Windows\System32\DriverStore\FileRepository\
      nvlti.inf_amd64_c284234d5322c92e\nvlti.inf}
idb:  Unregistered driver package 'nvlti.inf_amd64_c284234d5322c92e' from 'oem277.inf'.
```

`-clean` removed the working notebook package first, then found `nvlddmkm.sys` locked and
skipped the copy. The 5070 dropped to Code 28 and moved from Display to Other devices.

The install was not queued for a reboot either. There were no NVIDIA entries pending:

```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager" `
  -Name PendingFileRenameOperations).PendingFileRenameOperations |
  Where-Object { $_ -match "nvlddmkm|nv_dispi|DriverStore" }
```

With the eGPU **disconnected**, the desktop installer refused to run: exit code `-469762016`
(`0xE4000020`), because no desktop GeForce was present.

## 6. Installing with pnputil

First confirm `nvlddmkm` is unloaded (eGPU disconnected and the other card driverless, or
disabled in Device Manager):

```powershell
(Get-Service nvlddmkm -ErrorAction SilentlyContinue).Status
# Stopped
```

Then, elevated:

```powershell
$dd = "C:\temp\extract\Display.Driver"
pnputil.exe /add-driver "$dd\nvlti.inf"    /install
pnputil.exe /add-driver "$dd\nv_dispi.inf" /install
pnputil.exe /delete-driver oem268.inf /uninstall
```

```text
Adding driver package:  nvlti.inf
Driver package added successfully.
Published Name:         oem272.inf
Driver package installed on device: PCI\VEN_10DE&DEV_2D18&SUBSYS_3E3817AA&REV_A1\...

Adding driver package:  nv_dispi.inf
Driver package added successfully.
Published Name:         oem277.inf
Driver package installed on device: PCI\VEN_10DE&DEV_2D04&SUBSYS_8A071043&REV_A1\...
Driver package installed on device: HDAUDIO\FUNC_01&VEN_10DE&DEV_00AD&...

Driver package uninstalled.
Driver package deleted successfully.
```

Notes:

- **`oem277.inf` is a reused name.** Before the fix it was the old notebook package; the
  failed `-clean` install unregistered it, and Windows reused the name for `nv_dispi.inf`.
  Published names are slots, not identities. Always check `Original Name`.
- **Timing:** `nvlti.inf` took 5 min 34 s, `nv_dispi.inf` 33 s. It has not hung.
- **Offline pre-binding:** `nv_dispi.inf` bound to `DEV_2D04` with the enclosure
  disconnected, so the eGPU gets the right version the moment it appears.
- `/force` is ignored with `/uninstall`, so leave it out.

Verify the driver store:

```powershell
pnputil /enum-drivers | Select-String -Pattern "nv_dispi|nvlti|nv_dispig" -Context 0,3
```

```text
Original Name: nvlti.inf       Provider: NVIDIA   Class: Display
Original Name: nv_dispi.inf    Provider: NVIDIA   Class: Display
# nv_dispig.inf (591.86) - gone
```

## 7. The cable

After connecting the enclosure and restarting, there was no device, no error code and no PnP
event in the System log:

```powershell
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'2026-09-19 17:18:51'} |
  Sort-Object TimeCreated | Select-Object TimeCreated, ProviderName, Id
```

Only DistributedCOM and HttpService entries. The enclosure's router showed why:

```text
USB4(TM) Host Router (Microsoft)          OK        <- laptop controller
USB4 Router (2.0), Razer - Core X V2      Unknown   <- enclosure not enumerated
```

It was a USB-C cable that only carried power and USB 2.0. With a real USB4 cable:

```text
USB4 Router (2.0), Razer - Core X V2      OK
Razer Core X V2 - LWI Wizard              OK
NVIDIA GeForce RTX 5060 Ti                Error   CM_PROB_NORMAL_CONFLICT  <- Code 12, expected on hot-plug
```

## 8. Working

One Restart later:

```text
index, name, driver_version, memory.total [MiB], pci.bus_id
0, NVIDIA GeForce RTX 5070 Laptop GPU, 616.92,  8151 MiB, 00000000:02:00.0
1, NVIDIA GeForce RTX 5060 Ti,         616.92, 16311 MiB, 00000000:06:00.0
```

```text
NVIDIA GeForce RTX 5060 Ti          OK   CM_PROB_NONE   32.0.16.1692   oem277.inf
NVIDIA GeForce RTX 5070 Laptop GPU  OK   CM_PROB_NONE   32.0.16.1692   oem272.inf
Intel(R) Graphics                   OK   CM_PROB_NONE   32.0.101.6737  oem123.inf
```

24,462 MiB combined: different INFs, same driver version, one `nvlddmkm`.

## Keeping it working

- **Stop Windows Update delivering drivers**, or it will reinstall a mismatched package.
  Group Policy: *Computer Configuration → Administrative Templates → Windows Components →
  Windows Update → Do not include drivers with Windows Update → Enabled*.
- **Always cold boot (Restart) with the enclosure attached.** Hot-plug gives Code 12.
- **Update drivers with `nvlddmkm` unloaded**, using `pnputil` rather than NVIDIA's installer.
  See [`Install-NvidiaDriver.ps1`](../diagnostics/Install-NvidiaDriver.ps1).
- **BitLocker:** none of the above changes TPM measurements. See
  [`bitlocker-notes.md`](bitlocker-notes.md) before changing anything in BIOS setup.
