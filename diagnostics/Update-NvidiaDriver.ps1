<#
.SYNOPSIS
    Puts every NVIDIA GPU on one pinned driver version: download, verify, pick the
    right INFs, install with pnputil.

.DESCRIPTION
    The repeatable version of the fix in docs/HOWTO.md step 6. It does not ship the
    driver (it is ~1 GB and NVIDIA's licence does not allow redistributing it);
    instead diagnostics/nvidia-driver.json pins a version, and this script fetches
    exactly that version from NVIDIA.

      1. Downloads <version>-<flavour>-win10-win11-64bit-international-dch-whql.exe
         from us.download.nvidia.com (skipped if already downloaded, or with
         -InstallerPath / -DisplayDriverPath).
      2. Refuses it unless its Authenticode signature is valid and from NVIDIA.
      3. Extracts Display.Driver with Windows' built-in tar.exe.
      4. For every NVIDIA display device Windows knows about - including an eGPU
         that is unplugged right now - finds the INF that covers it: an exact
         DEV_xxxx&SUBSYS_yyyyyyyy match first (the laptop maker's INF), else the
         generic desktop INF for DEV_xxxx.
      5. Checks those INFs carry the version you asked for.
      6. Hands them to Install-NvidiaDriver.ps1, which stops on a failed install.
      7. Lists any other NVIDIA display packages left in the driver store, with the
         command to remove each. It does not remove them itself.

    PREREQUISITE for step 6 (not for -Plan or -DownloadOnly): nvlddmkm must be
    unloaded - disable the laptop GPU in Device Manager and unplug the eGPU.
    Install-NvidiaDriver.ps1 checks this and explains.

.PARAMETER Version
    NVIDIA driver version, e.g. 616.92. Defaults to the one in nvidia-driver.json.

.PARAMETER Flavour
    'desktop' (default) or 'notebook'. The desktop package also contains the OEM
    notebook INFs, so it covers a laptop GPU and a desktop eGPU at once.

.PARAMETER WorkDir
    Where the installer is downloaded and extracted, one folder per version.
    Default: $env:LOCALAPPDATA\nvidia-driver - not TEMP, so Windows' clean-up does
    not delete it and the exact pinned installer stays on the machine for next time.

.PARAMETER InstallerPath
    Use an installer you already downloaded instead of downloading one.

.PARAMETER DisplayDriverPath
    Use an already-extracted Display.Driver folder (skips download and signature check).

.PARAMETER Inf
    Install these INFs instead of the ones the script picks.

.PARAMETER Plan
    Only show which INF would be installed for which GPU, then stop. Needs no
    Administrator rights. Combine with -DisplayDriverPath to skip the download.

.PARAMETER DownloadOnly
    Download, verify and extract, then stop. Do this while both cards are working;
    run again later with the cards released to install.

.EXAMPLE
    .\Update-NvidiaDriver.ps1 -DownloadOnly          # stage the pinned version
    .\Update-NvidiaDriver.ps1 -Plan                  # see what it would install
    .\Update-NvidiaDriver.ps1                        # install (cards released, elevated)

.EXAMPLE
    .\Update-NvidiaDriver.ps1 -Version 620.10 -WhatIf

.NOTES
    Run elevated for the install. Afterwards: restart with the enclosure attached,
    then run Test-DriverConflict.ps1 (expect exit code 0).

    The leftover-package report reads 'pnputil /enum-drivers', whose field names
    are localised; it matches the English ones. On other languages it finds no
    packages and says so, rather than claiming the store is clean.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [string]$Version,

    [ValidateSet('desktop', 'notebook')]
    [string]$Flavour,

    [string]$WorkDir = (Join-Path $env:LOCALAPPDATA 'nvidia-driver'),

    [string]$InstallerPath,

    [string]$DisplayDriverPath,

    [string[]]$Inf,

    [switch]$Plan,

    [switch]$DownloadOnly,

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function ConvertTo-NvidiaMarketingVersion {
    <#
        32.0.16.1692 -> "16" + "1692" = "161692" -> "61692" -> 616.92
        (same rule as Get-GpuState.ps1)
    #>
    param([string]$DriverVersion)
    $parts = $DriverVersion -split '\.'
    if ($parts.Count -lt 4) { return $null }
    $joined = $parts[2] + $parts[3]
    if ($joined.Length -lt 5) { return $null }
    $last5 = $joined.Substring($joined.Length - 5)
    return '{0}.{1}' -f $last5.Substring(0, 3), $last5.Substring(3)
}

function Get-NvidiaDisplayDevice {
    <#
        Every NVIDIA display device Windows has a record of, present or not, as
        objects with Name, Present, Dev (e.g. 2D04) and Subsys (e.g. 8A071043).
        Non-present devices matter: an unplugged eGPU still needs its INF staged.
    #>
    Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -match '^PCI\\VEN_10DE&DEV_([0-9A-F]{4})&SUBSYS_([0-9A-F]{8})' } |
        ForEach-Object {
            $null = $_.InstanceId -match '^PCI\\VEN_10DE&DEV_([0-9A-F]{4})&SUBSYS_([0-9A-F]{8})'
            [pscustomobject]@{
                Name    = $_.FriendlyName
                Present = [bool]$_.Present
                Dev     = $Matches[1].ToUpper()
                Subsys  = $Matches[2].ToUpper()
            }
        } |
        Sort-Object Dev, Subsys -Unique
}

function Find-InfForDevice {
    <#
        The INF that covers one device. Exact DEV&SUBSYS first - that is the
        laptop maker's own INF (nvlti.inf for Lenovo, nvdmi.inf for Dell...).
        Otherwise an INF listing the bare DEV_xxxx with no SUBSYS, which is the
        generic one a retail desktop card binds to (nv_dispi.inf). Returns the
        candidates; the caller decides what to do with none or several.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$DriverPath,
        [Parameter(Mandatory = $true)][string]$Dev,
        [Parameter(Mandatory = $true)][string]$Subsys
    )
    $infs = Get-ChildItem -Path $DriverPath -Filter *.inf -File
    $exact = @($infs | Where-Object {
            Select-String -Path $_.FullName -Pattern "DEV_$Dev&SUBSYS_$Subsys" -SimpleMatch -Quiet
        })
    if ($exact.Count -gt 0) {
        return [pscustomobject]@{ Match = 'exact'; Inf = @($exact | ForEach-Object Name) }
    }
    $generic = @($infs | Where-Object {
            # The id must end here: not followed by another hex digit (DEV_2D04 is
            # not DEV_2D040) and not by &SUBSYS (that is an exact entry for some
            # other maker's board).
            Select-String -Path $_.FullName -Pattern "DEV_$Dev(?![0-9A-Fa-f])(?!&SUBSYS)" -Quiet
        })
    return [pscustomobject]@{ Match = 'generic'; Inf = @($generic | ForEach-Object Name) }
}

function Get-InfDriverVersion {
    param([Parameter(Mandatory = $true)][string]$Path)
    $line = Select-String -Path $Path -Pattern '^\s*DriverVer\s*=' | Select-Object -First 1
    if ($line -and $line.Line -match ',\s*([0-9.]+)') { return $Matches[1] }
    return $null
}

function Get-NvidiaDisplayPackage {
    <# NVIDIA packages in the Display class of the driver store, from pnputil. #>
    $output = & pnputil.exe /enum-drivers /class Display
    if ($LASTEXITCODE -ne 0) {
        throw "pnputil /enum-drivers failed with exit code $LASTEXITCODE."
    }
    $packages = @()
    $current = @{}
    foreach ($line in @($output) + '') {
        if ($line -match '^\s*$') {
            if ($current['Provider Name'] -match '^NVIDIA') { $packages += [pscustomobject]$current }
            $current = @{}
            continue
        }
        if ($line -match '^\s*([^:]+?)\s*:\s*(.*?)\s*$') { $current[$Matches[1]] = $Matches[2] }
    }
    return $packages
}

# --- which version --------------------------------------------------------
$pinFile = Join-Path $PSScriptRoot 'nvidia-driver.json'
if (Test-Path $pinFile) {
    $pin = Get-Content $pinFile -Raw | ConvertFrom-Json
    if (-not $Version) { $Version = $pin.version }
    if (-not $Flavour) { $Flavour = $pin.flavour }
}
if (-not $Flavour) { $Flavour = 'desktop' }
$versionLabel = if ($Version) { $Version } else { 'unpinned' }
if (-not $DisplayDriverPath -and $Version -notmatch '^\d{3}\.\d{2}$') {
    throw "Version '$Version' is not in the form 616.92. Pass -Version or fix $pinFile."
}

$needsAdmin = -not ($Plan -or $DownloadOnly)
if ($needsAdmin) {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Installing needs an elevated PowerShell. -Plan and -DownloadOnly do not.'
    }
}

# --- get Display.Driver ----------------------------------------------------
if (-not $DisplayDriverPath) {
    $fileName = "$Version-$Flavour-win10-win11-64bit-international-dch-whql.exe"
    $versionDir = Join-Path $WorkDir $Version
    New-Item -ItemType Directory -Force -Path $versionDir | Out-Null

    if (-not $InstallerPath) {
        $InstallerPath = Join-Path $versionDir $fileName
        if (Test-Path $InstallerPath) {
            Write-Host "Using already-downloaded $InstallerPath"
        }
        else {
            $url = "https://us.download.nvidia.com/Windows/$Version/$fileName"
            Write-Host "Downloading $url"
            Write-Host '(about 1 GB)'
            $ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest crawl
            try {
                Invoke-WebRequest -Uri $url -OutFile "$InstallerPath.partial" -UseBasicParsing
                Move-Item "$InstallerPath.partial" $InstallerPath
            }
            catch {
                Remove-Item "$InstallerPath.partial" -ErrorAction SilentlyContinue
                throw "Download of $url failed: $($_.Exception.Message)"
            }
        }
    }

    # Only ever install what NVIDIA signed.
    $sig = Get-AuthenticodeSignature -FilePath $InstallerPath
    $signer = if ($sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { '' }
    if ($sig.Status -ne 'Valid' -or $signer -notmatch 'O=NVIDIA Corporation') {
        throw "Refusing $InstallerPath - signature status '$($sig.Status)', signer '$signer'. Delete it and download again."
    }
    Write-Host "Signature OK: $signer"

    $DisplayDriverPath = Join-Path $versionDir 'Display.Driver'
    if (-not (Test-Path $DisplayDriverPath)) {
        Write-Host "Extracting Display.Driver to $versionDir"
        & tar.exe -xf $InstallerPath -C $versionDir 'Display.Driver'
        if ($LASTEXITCODE -ne 0) { throw "tar.exe could not extract Display.Driver (exit $LASTEXITCODE)." }
    }
}
if (-not (Test-Path $DisplayDriverPath)) { throw "Display.Driver not found: $DisplayDriverPath" }

if ($DownloadOnly) {
    Write-Host ''
    Write-Host "Staged: $DisplayDriverPath"
    Write-Host 'Next: .\Update-NvidiaDriver.ps1 -Plan, then release both cards and run it elevated.'
    return
}

# --- pick INFs -------------------------------------------------------------
$devices = @(Get-NvidiaDisplayDevice)
if ($devices.Count -eq 0 -and -not $Inf) {
    throw 'Windows has no record of any NVIDIA display device. Pass -Inf explicitly.'
}

Write-Host ''
Write-Host "--- INF per GPU (driver $versionLabel) ---" -ForegroundColor Cyan
$chosen = @()
if ($Inf) {
    $chosen = $Inf
    Write-Host "Using -Inf as given: $($Inf -join ', ')"
}
else {
    foreach ($d in $devices) {
        $found = Find-InfForDevice -DriverPath $DisplayDriverPath -Dev $d.Dev -Subsys $d.Subsys
        $state = if ($d.Present) { 'present' } else { 'not present' }
        $label = "{0} (DEV_{1} SUBSYS_{2}, {3})" -f $d.Name, $d.Dev, $d.Subsys, $state
        if ($found.Inf.Count -eq 1) {
            Write-Host ("  {0} -> {1} [{2} match]" -f $label, $found.Inf[0], $found.Match)
            $chosen += $found.Inf[0]
        }
        elseif ($found.Inf.Count -eq 0) {
            throw "No INF in $DisplayDriverPath covers $label. Is this the right package ($Flavour)?"
        }
        else {
            throw ("Several INFs cover {0}: {1}. Pick one and pass -Inf." -f $label, ($found.Inf -join ', '))
        }
    }
}
$chosen = @($chosen | Select-Object -Unique)

# --- version check ---------------------------------------------------------
foreach ($name in $chosen) {
    $path = Join-Path $DisplayDriverPath $name
    if (-not (Test-Path $path)) { throw "$name not found in $DisplayDriverPath" }
    $wddm = Get-InfDriverVersion -Path $path
    $marketing = ConvertTo-NvidiaMarketingVersion $wddm
    Write-Host ("  {0}: DriverVer {1} = {2}" -f $name, $wddm, $marketing)
    if ($Version -and $marketing -and $marketing -ne $Version) {
        throw "$name is version $marketing, not $Version. Wrong package?"
    }
}

if ($Plan) {
    Write-Host ''
    Write-Host 'Plan only - nothing installed.'
    return
}

# --- install ---------------------------------------------------------------
$installer = Join-Path $PSScriptRoot 'Install-NvidiaDriver.ps1'
if ($PSCmdlet.ShouldProcess(($chosen -join ', '), "Install NVIDIA driver $versionLabel with pnputil")) {
    & $installer -DisplayDriverPath $DisplayDriverPath -Inf $chosen -Force:$Force -Confirm:$false
}
elseif ($WhatIfPreference) {
    & $installer -DisplayDriverPath $DisplayDriverPath -Inf $chosen -Force:$Force -WhatIf
}
else {
    Write-Host 'Aborted - nothing installed.'
    return
}

# --- what is left in the store --------------------------------------------
try {
    $packages = @(Get-NvidiaDisplayPackage)
}
catch {
    Write-Host "Could not list the driver store: $($_.Exception.Message) Check by hand:" -ForegroundColor Yellow
    Write-Host '  pnputil /enum-drivers /class Display'
    $packages = $null
}
$stale = @()
$unreadable = @()
foreach ($p in $packages) {
    $v = ($p.'Driver Version' -split '\s+')[-1]
    $marketing = if ($v) { ConvertTo-NvidiaMarketingVersion $v } else { $null }
    if (-not $marketing) { $unreadable += $p }
    elseif ($marketing -ne $Version) { $stale += $p }
}
Write-Host ''
if ($null -eq $packages) {
    $packages = @()
}
elseif ($packages.Count -eq 0) {
    Write-Host 'pnputil listed no NVIDIA display packages; its field names are localised, so on non-English Windows check by hand:' -ForegroundColor Yellow
    Write-Host '  pnputil /enum-drivers /class Display'
}
foreach ($p in $unreadable) {
    Write-Host ("Could not read the version of {0} ({1}); check it by hand." -f $p.'Published Name', $p.'Original Name') -ForegroundColor Yellow
}
if ($stale.Count -gt 0) {
    Write-Host 'Other NVIDIA display packages still in the driver store:' -ForegroundColor Yellow
    foreach ($p in $stale) {
        $v = ($p.'Driver Version' -split '\s+')[-1]
        Write-Host ("  {0} ({1}, {2})" -f $p.'Published Name', $p.'Original Name', (ConvertTo-NvidiaMarketingVersion $v))
    }
    Write-Host 'Once both cards work on the new version, remove each with:'
    Write-Host '  pnputil /delete-driver <published name> /uninstall'
    Write-Host 'Windows can re-bind a card to a leftover older package, so do not leave them.'
}
elseif ($packages.Count -gt 0) {
    Write-Host "No other NVIDIA display packages in the driver store."
}
Write-Host ''
Write-Host 'Next: re-enable the laptop GPU, plug the enclosure in, Restart (not Shut down),'
Write-Host 'then run .\diagnostics\Test-DriverConflict.ps1 and expect exit code 0.'
