<#
.SYNOPSIS
    One-command health check for the laptop + eGPU build: is everything up, and if
    not, what to do about it.

.DESCRIPTION
    Read-only. Checks, in the order a fault propagates:

      1. Enclosure link   - is the enclosure's USB4/Thunderbolt router on the bus?
      2. PCIe switch      - did the enclosure's PCIe switch start? Code 10 here means
                            the card behind it is never seen at all.
      3. GPUs             - are the expected NVIDIA GPUs present and working, and if
                            not, which Device Manager code do they show?
      4. Driver versions  - are all NVIDIA GPUs on the same driver version?
      5. nvidia-smi       - does the driver itself see every GPU?
      6. Windows Update   - is driver delivery blocked, so it cannot happen again?
      7. GPU resets       - has nvlddmkm logged "GPU reset required" recently? An
                            early warning, not a fault on its own.

    Each check prints PASS, WARN or FAIL, and every WARN/FAIL prints the fix.
    See docs/HOWTO.md "When the eGPU disappears".

.PARAMETER ExpectedGpus
    How many NVIDIA GPUs should be working. Default 2 (laptop GPU + eGPU).

.PARAMETER ResetWindowHours
    How far back to look for GPU reset events. Default 24.

.OUTPUTS
    Exit code 0 - healthy (warnings allowed)
    Exit code 1 - at least one FAIL

.EXAMPLE
    .\Test-EgpuHealth.ps1

.NOTES
    Needs no Administrator rights and changes nothing.
#>
[CmdletBinding()]
param(
    [int]$ExpectedGpus = 2,
    [int]$ResetWindowHours = 24
)

$script:failed = $false

function Write-Result {
    param(
        [ValidateSet('PASS', 'WARN', 'FAIL')][string]$Level,
        [string]$Check,
        [string]$Detail,
        [string[]]$Fix
    )
    $colour = @{ PASS = 'Green'; WARN = 'Yellow'; FAIL = 'Red' }[$Level]
    Write-Host ("[{0}] {1}: {2}" -f $Level, $Check, $Detail) -ForegroundColor $colour
    foreach ($line in $Fix) { Write-Host "       -> $line" }
    if ($Level -eq 'FAIL') { $script:failed = $true }
}

function ConvertTo-NvidiaMarketingVersion {
    param([string]$DriverVersion)
    $parts = $DriverVersion -split '\.'
    if ($parts.Count -lt 4) { return $DriverVersion }
    $joined = $parts[2] + $parts[3]
    if ($joined.Length -lt 5) { return $DriverVersion }
    $last5 = $joined.Substring($joined.Length - 5)
    return '{0}.{1}' -f $last5.Substring(0, 3), $last5.Substring(3)
}

$powerCycle = @(
    'Power-cycle the enclosure: shut the laptop down, switch the enclosure off at the mains for 30 s,',
    'switch it back on, start the laptop with it attached, then Restart (not Shut down).',
    'A laptop restart alone does not reset the enclosure - it has its own power.'
)

Write-Host ''
Write-Host 'eGPU health check' -ForegroundColor Cyan
Write-Host ''

# --- 1. enclosure link -----------------------------------------------------
# The laptop's own host/root routers are always there; the enclosure adds a
# router of its own (e.g. "USB4 Router (2.0), Razer - Core X V2").
$routers = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -match 'USB4.*Router|Thunderbolt' -and $_.FriendlyName -notmatch 'Host Router|Root Router|Controller' })
$badRouters = @($routers | Where-Object { $_.Status -ne 'OK' })
if ($routers.Count -eq 0) {
    Write-Result FAIL 'Enclosure link' 'no enclosure router on the USB4/Thunderbolt bus' @(
        'Check the enclosure is switched on, then re-seat the cable at both ends or try the other USB4 port.',
        'Use a proper 40 Gbps USB4/Thunderbolt cable: a USB 2.0-only USB-C cable looks identical.')
}
elseif ($badRouters.Count -gt 0) {
    Write-Result FAIL 'Enclosure link' (($badRouters | ForEach-Object { "$($_.FriendlyName) = $($_.Status)" }) -join '; ') @(
        'Re-seat the cable or try the other port. Status Unknown usually means power, cable or port.')
}
else {
    Write-Result PASS 'Enclosure link' (($routers | ForEach-Object FriendlyName) -join '; ')
}

# --- 2. PCIe switch --------------------------------------------------------
$switches = @(Get-PnpDevice -PresentOnly -Class System -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -match 'Upstream Switch Port|Downstream Switch Port' })
$badSwitches = @($switches | Where-Object { $_.ConfigManagerErrorCode -ne 'CM_PROB_NONE' })
if ($badSwitches.Count -gt 0) {
    $detail = ($badSwitches | ForEach-Object { "$($_.FriendlyName) = $($_.ConfigManagerErrorCode)" }) -join '; '
    Write-Result FAIL 'PCIe switch' $detail (@('The enclosure''s PCIe link did not start, so the GPU behind it is invisible.') + $powerCycle +
        @('If that fails: Device Manager > System devices > the switch port > Disable device, then Enable device, then Scan for hardware changes.'))
}
elseif ($switches.Count -eq 0) {
    Write-Result WARN 'PCIe switch' 'no PCIe switch ports found (normal if the enclosure is not attached)'
}
else {
    Write-Result PASS 'PCIe switch' ("{0} switch port(s) OK" -f $switches.Count)
}

# --- 3. GPUs ---------------------------------------------------------------
$nvidia = @(Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | Where-Object { $_.InstanceId -like 'PCI\VEN_10DE*' })
$present = @($nvidia | Where-Object Present)
$phantom = @($nvidia | Where-Object { -not $_.Present })
$working = @($present | Where-Object { $_.ConfigManagerErrorCode -eq 'CM_PROB_NONE' })

foreach ($g in ($present | Where-Object { $_.ConfigManagerErrorCode -ne 'CM_PROB_NONE' })) {
    $code = [string]$g.ConfigManagerErrorCode
    switch ($code) {
        'CM_PROB_NORMAL_CONFLICT' {
            Write-Result FAIL $g.FriendlyName 'Code 12 - not enough free resources (plugged in after boot)' @(
                'Restart with the enclosure attached. Restart, not Shut down: with Fast Startup, Shut down is a hibernate.')
        }
        'CM_PROB_FAILED_POST_START' {
            Write-Result FAIL $g.FriendlyName 'Code 43 - the GPU crashed or was reset and did not recover' $powerCycle
        }
        'CM_PROB_FAILED_ADD' {
            Write-Result FAIL $g.FriendlyName 'Code 31 - driver could not load (usually two NVIDIA driver versions)' @(
                'Run .\diagnostics\Test-DriverConflict.ps1, then .\diagnostics\Update-NvidiaDriver.ps1 (docs/HOWTO.md step 6).')
        }
        'CM_PROB_FAILED_INSTALL' {
            Write-Result FAIL $g.FriendlyName 'Code 28 - no driver installed' @(
                'Install the pinned driver with .\diagnostics\Update-NvidiaDriver.ps1 (docs/HOWTO.md step 6).')
        }
        default {
            Write-Result FAIL $g.FriendlyName $code @('See docs/device-error-codes.md.')
        }
    }
}

if ($working.Count -ge $ExpectedGpus) {
    Write-Result PASS 'GPUs' (($working | ForEach-Object FriendlyName) -join '; ')
}
elseif ($present.Count -lt $ExpectedGpus) {
    $missing = @($phantom | Where-Object { $present.FriendlyName -notcontains $_.FriendlyName } | ForEach-Object FriendlyName | Select-Object -Unique)
    $name = if ($missing) { $missing -join ', ' } else { 'a GPU' }
    Write-Result FAIL 'GPUs' ("{0} of {1} expected NVIDIA GPUs present; missing: {2}" -f $present.Count, $ExpectedGpus, $name) @(
        'If the PCIe switch check above failed, fix that first.',
        'Otherwise: the card is not reaching Windows at all. Power-cycle the enclosure, then check the card is',
        'seated and its power cable is connected.')
}

# --- 4. driver versions ----------------------------------------------------
$bound = @(Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" -ErrorAction SilentlyContinue |
        Where-Object { $_.DeviceName -match 'NVIDIA' })
$versions = @($bound | ForEach-Object { $_.DriverVersion } | Select-Object -Unique)
if ($bound.Count -eq 0) {
    Write-Result WARN 'Driver versions' 'no NVIDIA GPU has a driver bound right now'
}
elseif ($versions.Count -gt 1) {
    $detail = ($bound | ForEach-Object { "{0} = {1}" -f $_.DeviceName, (ConvertTo-NvidiaMarketingVersion $_.DriverVersion) }) -join '; '
    Write-Result FAIL 'Driver versions' "mismatch: $detail" @(
        'Only one version of nvlddmkm.sys can load. Put both cards on one version with',
        '.\diagnostics\Update-NvidiaDriver.ps1 (docs/HOWTO.md step 6).')
}
else {
    Write-Result PASS 'Driver versions' ("all on {0}" -f (ConvertTo-NvidiaMarketingVersion $versions[0]))
}

# --- 5. nvidia-smi ---------------------------------------------------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Write-Result WARN 'nvidia-smi' 'not on PATH; skipped'
}
else {
    $rows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null |
            Where-Object { $_ -and $_ -notmatch 'Unable|Failed|No devices' })
    if ($LASTEXITCODE -ne 0 -or $rows.Count -eq 0) {
        Write-Result FAIL 'nvidia-smi' 'the NVIDIA driver sees no GPUs' @('Fix the checks above first.')
    }
    elseif ($rows.Count -lt $ExpectedGpus) {
        Write-Result FAIL 'nvidia-smi' ("sees {0} of {1}: {2}" -f $rows.Count, $ExpectedGpus, ($rows -join ' | ')) @('Fix the checks above first.')
    }
    else {
        Write-Result PASS 'nvidia-smi' ($rows -join ' | ')
    }
}

# --- 6. Windows Update driver block ----------------------------------------
$policy = (Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate' -Name ExcludeWUDriversInQualityUpdate -ErrorAction SilentlyContinue).ExcludeWUDriversInQualityUpdate
$search = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching' -Name SearchOrderConfig -ErrorAction SilentlyContinue).SearchOrderConfig
if ($policy -eq 1 -and $search -eq 0) {
    Write-Result PASS 'Windows Update' 'driver delivery blocked'
}
else {
    Write-Result WARN 'Windows Update' 'can still install drivers, and may install a mismatched NVIDIA one' @(
        'Run .\diagnostics\Disable-WindowsUpdateDrivers.ps1 -Apply in an elevated PowerShell.')
}

# --- 7. recent GPU resets --------------------------------------------------
# nvlddmkm event 14 carries "GPU recovery action changed ... (GPU Reset Required)";
# 153 is "Error occurred on GPUID". Not a fault by themselves, but a card that
# keeps resetting is the one that ends up on Code 43 or disappears.
$since = (Get-Date).AddHours(-$ResetWindowHours)
$resets = @(Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'nvlddmkm'; StartTime = $since } -ErrorAction SilentlyContinue |
        Where-Object { $_.Id -in 14, 153 })
if ($resets.Count -eq 0) {
    Write-Result PASS 'GPU resets' "none in the last $ResetWindowHours h"
}
else {
    $last = ($resets | Sort-Object TimeCreated -Descending | Select-Object -First 1).TimeCreated
    Write-Result WARN 'GPU resets' ("{0} in the last {1} h, most recent {2:yyyy-MM-dd HH:mm}" -f $resets.Count, $ResetWindowHours, $last) @(
        'Unload models with "ollama stop <model>" before stopping Ollama; never end ollama.exe or llama-server.exe',
        'from Task Manager with a model loaded. Do not plug or unplug the enclosure while a model is loaded.',
        'Fewer monitors on the laptop GPU also helps.')
}

Write-Host ''
if ($script:failed) {
    Write-Host 'Not healthy - work through the FAIL lines from the top: an earlier fault causes the later ones.' -ForegroundColor Red
    exit 1
}
Write-Host 'Healthy.' -ForegroundColor Green
exit 0
