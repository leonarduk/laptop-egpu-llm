<#
.SYNOPSIS
    Installs NVIDIA driver INFs directly with pnputil, bypassing NVIDIA's installer.

.DESCRIPTION
    Why not just run NVIDIA's setup.exe? Because with two GPUs it is a catch-22:

      * eGPU CONNECTED   - setup runs, but nvlddmkm.sys is locked by the live GPU.
                           With -clean it removes the existing package FIRST, then
                           fails to copy the new one, leaving you with no driver:
                               inf: Service 'nvlddmkm' still in use by 2 sources.
                               cpy: Skipping isolated file 'nvlddmkm.sys'.
      * eGPU DISCONNECTED - the desktop package refuses to run at all, because no
                           desktop GeForce is present to match (exit 0xE4000020).

    pnputil has no hardware-presence check and no installer state machine. It also
    pre-binds a package to an offline device record, so you can stage the driver for
    a disconnected eGPU and it will bind correctly the moment the card appears.

    PREREQUISITE: nvlddmkm must be unloaded. Disconnect the eGPU, or disable the
    other card in Device Manager, before running this.

.PARAMETER DisplayDriverPath
    Path to the extracted Display.Driver folder. Extract NVIDIA's installer with
    Windows' built-in tar (it is a 7-Zip self-extracting archive):

        tar.exe -xf "616.92-desktop-win10-win11-64bit-international-dch-whql.exe" "Display.Driver"

.PARAMETER Inf
    INF filenames to install. Defaults to the laptop (nvlti.inf) and desktop
    (nv_dispi.inf) variants. Check which INFs cover YOUR device IDs first:

        Get-ChildItem -Recurse -Filter *.inf |
          Select-String "DEV_XXXX" -SimpleMatch -List | ForEach-Object { $_.Filename }

.PARAMETER RemoveStaleInf
    Published name (e.g. oem268.inf) of an old NVIDIA package to delete afterwards.

.PARAMETER Force
    Skip the nvlddmkm-is-running safety check. Not recommended.

.EXAMPLE
    .\Install-NvidiaDriver.ps1 -DisplayDriverPath C:\temp\extract\Display.Driver

.EXAMPLE
    .\Install-NvidiaDriver.ps1 -DisplayDriverPath C:\temp\extract\Display.Driver -RemoveStaleInf oem268.inf

.NOTES
    Must be run elevated. Each package can take several minutes - these are
    gigabytes of files going into the driver store. It has not hung.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DisplayDriverPath,

    [string[]]$Inf = @('nvlti.inf', 'nv_dispi.inf'),

    [string]$RemoveStaleInf,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# --- elevation check -------------------------------------------------------
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'This script must be run as Administrator.'
}

# --- path check ------------------------------------------------------------
if (-not (Test-Path $DisplayDriverPath)) {
    throw "Display.Driver path not found: $DisplayDriverPath"
}

# --- nvlddmkm must be unloaded --------------------------------------------
$svc = Get-Service nvlddmkm -ErrorAction SilentlyContinue
$state = if ($svc) { $svc.Status } else { 'absent' }
Write-Host "nvlddmkm service state: $state"

if ($svc -and $svc.Status -eq 'Running' -and -not $Force) {
    Write-Host ''
    Write-Host 'REFUSING TO CONTINUE.' -ForegroundColor Red
    Write-Host 'nvlddmkm.sys is loaded, so the driver files cannot be replaced and the'
    Write-Host 'install will silently skip them. Do one of the following first:'
    Write-Host '  - disconnect the eGPU enclosure, or'
    Write-Host '  - disable the other NVIDIA card in Device Manager'
    Write-Host 'then re-run. Use -Force to override (you probably should not).'
    exit 1
}

# --- install ---------------------------------------------------------------
foreach ($name in $Inf) {
    $path = Join-Path $DisplayDriverPath $name
    if (-not (Test-Path $path)) {
        Write-Warning "Skipping $name - not found in $DisplayDriverPath"
        continue
    }

    Write-Host ''
    Write-Host "--- Installing $name ---" -ForegroundColor Cyan
    Write-Host '(this can take several minutes - it is copying GBs into the driver store)'
    $start = Get-Date
    & pnputil.exe /add-driver $path /install 2>&1 | Write-Host
    Write-Host ("Elapsed: {0:mm\:ss}" -f ((Get-Date) - $start))
}

# --- remove stale package --------------------------------------------------
if ($RemoveStaleInf) {
    Write-Host ''
    Write-Host "--- Removing stale package $RemoveStaleInf ---" -ForegroundColor Cyan
    # Note: /force is ignored when combined with /uninstall.
    & pnputil.exe /delete-driver $RemoveStaleInf /uninstall 2>&1 | Write-Host
}

# --- verify ----------------------------------------------------------------
Write-Host ''
Write-Host '--- Resulting display drivers ---' -ForegroundColor Cyan
Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" |
    Select-Object DeviceName, DriverVersion, InfName |
    Format-Table -AutoSize

Write-Host ''
Write-Host 'Now reconnect the enclosure and RESTART (not Shut down - Fast Startup'
Write-Host 'makes Shut down a hybrid hibernate, and you need a real POST to allocate'
Write-Host 'PCIe address space for the eGPU).'
