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
    Only attempted if every install above succeeded, and only after
    pnputil /enum-drivers confirms the name is an NVIDIA Display package.
    Asks for confirmation (ConfirmImpact High); -WhatIf shows what it would do,
    -Confirm:$false skips the prompt.

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
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter(Mandatory = $true)]
    [string]$DisplayDriverPath,

    [string[]]$Inf = @('nvlti.inf', 'nv_dispi.inf'),

    [string]$RemoveStaleInf,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# The Display adapters setup class, whatever the local language calls it.
$DisplayClassGuid = '{4d36e968-e325-11ce-bfc1-08002be10318}'

function Get-DriverPackage {
    <#
    .SYNOPSIS
        One package from 'pnputil /enum-drivers', as a hashtable of its
        "Label: value" lines, or $null if the published name is not listed.
    #>
    param([Parameter(Mandatory = $true)][string]$PublishedName)

    $output = & pnputil.exe /enum-drivers
    if ($LASTEXITCODE -ne 0) {
        throw "pnputil /enum-drivers failed with exit code $LASTEXITCODE."
    }

    # Packages are blocks of "Label:  value" lines separated by blank lines.
    $current = @{}
    foreach ($line in @($output) + '') {
        if ($line -match '^\s*$') {
            if ($current['Published Name'] -and $current['Published Name'] -ieq $PublishedName) {
                return $current
            }
            $current = @{}
            continue
        }
        if ($line -match '^\s*([^:]+?)\s*:\s*(.*?)\s*$') {
            $current[$Matches[1]] = $Matches[2]
        }
    }
    return $null
}

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
    Write-Host 'install will silently skip them.'
    Write-Host ''
    Write-Host 'This is the normal state before an update, not a fault: with both cards'
    Write-Host 'working, both hold the driver loaded. To release it, do BOTH of these:'
    Write-Host '  1. Device Manager > Display adapters > the laptop GPU (RTX 5070 Laptop)'
    Write-Host '     > right-click > Disable device'
    Write-Host '  2. unplug the eGPU enclosure (the Thunderbolt/OCuLink cable)'
    Write-Host 'Wait for the display to settle on the integrated GPU, then re-run.'
    Write-Host 'Re-enable the laptop GPU in Device Manager after the restart.'
    Write-Host ''
    Write-Host 'Use -Force to install anyway (you probably should not: the copy will'
    Write-Host 'skip the locked nvlddmkm.sys and leave a half-updated driver).'
    exit 1
}

# --- install ---------------------------------------------------------------
$installed = 0
foreach ($name in $Inf) {
    $path = Join-Path $DisplayDriverPath $name
    if (-not (Test-Path $path)) {
        Write-Warning "Skipping $name - not found in $DisplayDriverPath"
        continue
    }

    Write-Host ''
    Write-Host "--- Installing $name ---" -ForegroundColor Cyan
    if ($WhatIfPreference) {
        # -WhatIf must not install either; counted so the removal step below
        # still reports what it would have deleted.
        Write-Host "What if: pnputil /add-driver $path /install"
        $installed++
        continue
    }
    Write-Host '(this can take several minutes - it is copying GBs into the driver store)'
    $start = Get-Date
    & pnputil.exe /add-driver $path /install 2>&1 | Write-Host
    $code = $LASTEXITCODE
    Write-Host ("Elapsed: {0:mm\:ss}" -f ((Get-Date) - $start))

    # 0 = installed; 3010 = installed, reboot required; 259 = added to the
    # driver store but no present device matched it, which is the expected
    # result when staging for a disconnected eGPU. Anything else is a
    # failure, and must stop the run before the stale package is deleted -
    # otherwise a failed install followed by the delete leaves no driver.
    if ($code -eq 259) {
        Write-Host "$name is staged in the driver store; no present device matched it yet."
    }
    elseif ($code -ne 0 -and $code -ne 3010) {
        throw "pnputil /add-driver $name failed with exit code $code. Stopping before any package is removed."
    }
    $installed++
}

# --- remove stale package --------------------------------------------------
if ($RemoveStaleInf) {
    Write-Host ''
    Write-Host "--- Removing stale package $RemoveStaleInf ---" -ForegroundColor Cyan

    if ($installed -eq 0) {
        throw "No new package was installed, so $RemoveStaleInf is being kept: deleting it would leave no NVIDIA driver."
    }

    # Confirm the published name really is an NVIDIA display package before
    # deleting it. A typo'd oemNN.inf can be any driver on the system -
    # storage, network, chipset - and pnputil will delete it just the same.
    # Matched on English labels; on a localised Windows this finds nothing
    # and refuses, which is the safe failure.
    $package = Get-DriverPackage -PublishedName $RemoveStaleInf
    if (-not $package) {
        throw "$RemoveStaleInf is not in 'pnputil /enum-drivers'. Check the published name (oemNN.inf)."
    }
    $provider = $package['Provider Name']
    $className = $package['Class Name']
    $classGuid = $package['Class GUID']
    if ($provider -notmatch '^NVIDIA') {
        throw "$RemoveStaleInf is provided by '$provider', not NVIDIA. Refusing to delete it."
    }
    if ($className -notmatch 'Display' -and $classGuid -ne $DisplayClassGuid) {
        throw "$RemoveStaleInf is class '$className' $classGuid, not Display. Refusing to delete it."
    }
    Write-Host "$RemoveStaleInf : $provider, $className, $($package['Driver Version'])"

    if ($PSCmdlet.ShouldProcess("$RemoveStaleInf ($provider $($package['Driver Version']))", 'pnputil /delete-driver /uninstall')) {
        # Note: /force is ignored when combined with /uninstall.
        & pnputil.exe /delete-driver $RemoveStaleInf /uninstall 2>&1 | Write-Host
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) {
            Write-Warning "pnputil /delete-driver $RemoveStaleInf exited $LASTEXITCODE; the old package may still be present."
        }
    }
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
