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
          The Group Policy "Do not include drivers with Windows Updates".
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
    Put both values back as they were before -Apply (saved by -Apply in
    %ProgramData%\laptop-egpu-llm\wu-driver-settings.json), letting Windows Update
    deliver drivers again. With no saved copy it removes both values, which leaves
    Windows on its own defaults.

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
$backupFile = Join-Path $env:ProgramData 'laptop-egpu-llm\wu-driver-settings.json'

function Set-OrRemove {
    <# Restore one value: $null means it did not exist, so remove it. #>
    param([string]$Key, [string]$Name, $Value)
    if ($null -eq $Value) {
        Remove-ItemProperty -Path $Key -Name $Name -ErrorAction SilentlyContinue
    }
    else {
        if (-not (Test-Path $Key)) { New-Item -Path $Key -Force | Out-Null }
        New-ItemProperty -Path $Key -Name $Name -PropertyType DWord -Value ([int]$Value) -Force | Out-Null
    }
}

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
        # Remember the original values once, so -Undo restores them rather than
        # guessing. A second -Apply must not overwrite the first backup.
        if (-not (Test-Path $backupFile)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $backupFile) | Out-Null
            [pscustomobject]@{
                ExcludeWUDriversInQualityUpdate = Get-Value $policyKey $policyName
                SearchOrderConfig               = Get-Value $searchKey $searchName
                PolicyKeyExisted                = [bool](Test-Path $policyKey)
            } | ConvertTo-Json | Set-Content -Path $backupFile -Encoding UTF8
            Write-Host "Saved the previous values to $backupFile"
        }
        if (-not (Test-Path $policyKey)) { New-Item -Path $policyKey -Force | Out-Null }
        New-ItemProperty -Path $policyKey -Name $policyName -PropertyType DWord -Value 1 -Force | Out-Null
        if (-not (Test-Path $searchKey)) { New-Item -Path $searchKey -Force | Out-Null }
        New-ItemProperty -Path $searchKey -Name $searchName -PropertyType DWord -Value 0 -Force | Out-Null
    }
    else {
        if (-not $WhatIfPreference) { Write-Host 'Aborted - nothing changed.' }
        return
    }
}
elseif ($Undo) {
    if ($PSCmdlet.ShouldProcess('Windows Update', 'Allow installing device drivers again')) {
        if (Test-Path $backupFile) {
            $saved = Get-Content $backupFile -Raw | ConvertFrom-Json
            Set-OrRemove $policyKey $policyName $saved.ExcludeWUDriversInQualityUpdate
            Set-OrRemove $searchKey $searchName $saved.SearchOrderConfig
            # Remove the policy key only if -Apply created it and it is now empty.
            if (-not $saved.PolicyKeyExisted -and (Test-Path $policyKey)) {
                $key = Get-Item $policyKey
                if ($key.ValueCount -eq 0 -and $key.SubKeyCount -eq 0) { Remove-Item $policyKey }
            }
            Remove-Item $backupFile
            Write-Host "Restored the values saved in $backupFile"
        }
        else {
            # Absent is Windows' default for both, so removing them is the undo.
            Write-Warning "No saved values at $backupFile; removing both so Windows uses its defaults."
            Set-OrRemove $policyKey $policyName $null
            Set-OrRemove $searchKey $searchName $null
        }
    }
    else {
        if (-not $WhatIfPreference) { Write-Host 'Aborted - nothing changed.' }
        return
    }
}

Write-Host ''
Write-Host 'Now:'
$null = Show-State
