<#
.SYNOPSIS
    Stops Windows Update installing device drivers, so it cannot put a second,
    mismatched NVIDIA driver on the eGPU again.

.DESCRIPTION
    This is what broke the build in the first place. setupapi.dev.log on this
    machine shows the eGPU was first detected at 14:59 on 2026-09-17 and bound to
    Microsoft's basic display driver; at 15:15 a "Device Install (Install Windows
    Update driver)" section, run by wuaucltcore.exe, installed nv_dispig.inf
    (591.86) - a different version from the laptop GPU's 616.92, which left one
    card on Code 31. See docs/HOWTO.md step 6.

    Sets two machine-wide values (both documented Windows settings):

      HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate
          ExcludeWUDriversInQualityUpdate = 1
          The Group Policy "Do not include drivers with Windows Update".
          Works on Home editions too, which have no gpedit.msc.

      HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching
          SearchOrderConfig = 0
          Device installation settings: never fetch drivers from Windows Update
          when a new device appears.

    You then install and update GPU drivers yourself, with Update-NvidiaDriver.ps1.
    Windows still delivers security and feature updates; only drivers stop.

    Run with no switches to see the current state and change nothing.

.PARAMETER Apply
    Set both values. Asks for confirmation; -WhatIf shows what would change.

.PARAMETER Undo
    Remove the policy value and set SearchOrderConfig back to 1 (Windows' default),
    letting Windows Update deliver drivers again.

.EXAMPLE
    .\Disable-WindowsUpdateDrivers.ps1            # show the current state
    .\Disable-WindowsUpdateDrivers.ps1 -Apply     # turn driver updates off (elevated)

.NOTES
    Needs an elevated PowerShell for -Apply and -Undo. Takes effect for the next
    Windows Update scan; no restart needed. Organisation-managed PCs may have these
    set centrally, in which case a local change can be overwritten.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High', DefaultParameterSetName = 'Show')]
param(
    [Parameter(ParameterSetName = 'Apply')][switch]$Apply,
    [Parameter(ParameterSetName = 'Undo')][switch]$Undo
)

$ErrorActionPreference = 'Stop'

$policyKey = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate'
$policyName = 'ExcludeWUDriversInQualityUpdate'
$searchKey = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\DriverSearching'
$searchName = 'SearchOrderConfig'

function Get-Value {
    param([string]$Key, [string]$Name)
    $item = Get-ItemProperty -Path $Key -Name $Name -ErrorAction SilentlyContinue
    if ($item) { return $item.$Name }
    return $null
}

function Show-State {
    $policy = Get-Value $policyKey $policyName
    $search = Get-Value $searchKey $searchName
    $policyText = if ($policy -eq 1) { '1 (drivers excluded)' } elseif ($null -eq $policy) { 'not set (drivers included)' } else { "$policy (drivers included)" }
    $searchText = if ($search -eq 0) { '0 (never from Windows Update)' } elseif ($null -eq $search) { 'not set (Windows default)' } else { "$search (Windows Update allowed)" }
    Write-Host "  $policyName : $policyText"
    Write-Host "  $searchName : $searchText"
    return ($policy -eq 1 -and $search -eq 0)
}

Write-Host 'Current state:'
$protected = Show-State

if (-not ($Apply -or $Undo)) {
    Write-Host ''
    if ($protected) { Write-Host 'Windows Update will not install drivers.' -ForegroundColor Green }
    else { Write-Host 'Windows Update can still install drivers. Run with -Apply (elevated) to stop it.' -ForegroundColor Yellow }
    return
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Changing these settings needs an elevated PowerShell.'
}

if ($Apply) {
    if ($PSCmdlet.ShouldProcess('Windows Update', 'Stop installing device drivers')) {
        if (-not (Test-Path $policyKey)) { New-Item -Path $policyKey -Force | Out-Null }
        New-ItemProperty -Path $policyKey -Name $policyName -PropertyType DWord -Value 1 -Force | Out-Null
        if (-not (Test-Path $searchKey)) { New-Item -Path $searchKey -Force | Out-Null }
        New-ItemProperty -Path $searchKey -Name $searchName -PropertyType DWord -Value 0 -Force | Out-Null
    }
}
elseif ($Undo) {
    if ($PSCmdlet.ShouldProcess('Windows Update', 'Allow installing device drivers again')) {
        Remove-ItemProperty -Path $policyKey -Name $policyName -ErrorAction SilentlyContinue
        if (-not (Test-Path $searchKey)) { New-Item -Path $searchKey -Force | Out-Null }
        New-ItemProperty -Path $searchKey -Name $searchName -PropertyType DWord -Value 1 -Force | Out-Null
    }
}

Write-Host ''
Write-Host 'Now:'
$null = Show-State
