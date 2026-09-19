<#
.SYNOPSIS
    Detects the NVIDIA driver version collision that makes only one GPU work.

.DESCRIPTION
    If two NVIDIA GPUs are bound to driver packages of DIFFERENT versions, only one
    of them can work at a time.

    Both packages install the same kernel-mode driver, nvlddmkm.sys, and Windows
    loads exactly one instance of it. Whichever GPU binds first loads its version;
    the second is handed a driver built for a different version and fails with
    Code 31 (CM_PROB_FAILED_ADD):

        "The I/O device is configured incorrectly or the configuration
         parameters to the driver are incorrect."

    This is not a Windows limitation, a MUX switch issue, or an eGPU bandwidth
    problem. Windows runs multi-GPU NVIDIA setups fine. It just cannot run two
    versions of one kernel driver.

    Typical cause: Windows Update installs a desktop GeForce package when you first
    attach a desktop card, alongside the notebook package you already had.

.OUTPUTS
    Exit code 0 - no conflict
    Exit code 1 - conflict detected
    Exit code 2 - fewer than two NVIDIA GPUs found (nothing to check)

.EXAMPLE
    .\Test-DriverConflict.ps1
#>
[CmdletBinding()]
param()

function ConvertTo-NvidiaMarketingVersion {
    param([string]$DriverVersion)
    $parts = $DriverVersion -split '\.'
    if ($parts.Count -lt 4) { return $null }
    $joined = $parts[2] + $parts[3]
    if ($joined.Length -lt 5) { return $null }
    $last5 = $joined.Substring($joined.Length - 5)
    return $last5.Substring(0, 3) + '.' + $last5.Substring(3)
}

$nvidia = Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" -ErrorAction SilentlyContinue |
    Where-Object { $_.DeviceName -match 'NVIDIA' }

if (-not $nvidia) {
    Write-Host 'No NVIDIA display adapters found.' -ForegroundColor Yellow
    exit 2
}

Write-Host ''
Write-Host 'NVIDIA display adapters:' -ForegroundColor Cyan
foreach ($d in $nvidia) {
    $mk = ConvertTo-NvidiaMarketingVersion $d.DriverVersion
    $label = if ($mk) { "$($d.DriverVersion)  ($mk)" } else { $d.DriverVersion }
    Write-Host ("  {0,-38} {1,-24} {2}" -f $d.DeviceName, $label, $d.InfName)
}
Write-Host ''

if ($nvidia.Count -lt 2) {
    Write-Host 'Only one NVIDIA adapter present - nothing to compare.' -ForegroundColor Yellow
    Write-Host 'If your second GPU is missing entirely, check that the enclosure is'
    Write-Host 'on the bus at all (see Get-GpuState.ps1, USB4 section).'
    exit 2
}

$versions = $nvidia | Select-Object -ExpandProperty DriverVersion -Unique

if ($versions.Count -eq 1) {
    Write-Host 'PASS: all NVIDIA GPUs are on the same driver version.' -ForegroundColor Green
    Write-Host 'If a GPU still is not working, this is not the cause. Check error codes'
    Write-Host 'with Get-GpuState.ps1 - Code 12 is a resource problem, not a driver one.'
    exit 0
}

Write-Host 'FAIL: NVIDIA GPUs are bound to DIFFERENT driver versions.' -ForegroundColor Red
Write-Host ''
Write-Host 'Versions found:' -ForegroundColor Red
foreach ($v in $versions) {
    $mk = ConvertTo-NvidiaMarketingVersion $v
    Write-Host ("  {0}  ({1})" -f $v, $mk)
}
Write-Host ''
Write-Host 'Only one of these GPUs can work at a time. Windows loads a single instance'
Write-Host 'of nvlddmkm.sys, so the card whose package version does not match it fails'
Write-Host 'with Code 31.'
Write-Host ''
Write-Host 'Fix: install one driver version covering both device IDs.'
Write-Host 'See Install-NvidiaDriver.ps1 in this folder.'
exit 1
