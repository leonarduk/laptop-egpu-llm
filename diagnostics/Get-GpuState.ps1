<#
.SYNOPSIS
    One-shot diagnostic dump for multi-GPU and eGPU setups on Windows.

.DESCRIPTION
    Collects everything you need to diagnose "only one GPU works" problems:
      - Display devices present, with raw ConfigManagerErrorCode (not the friendly text)
      - Bound driver packages and versions, with NVIDIA marketing version decoded
      - USB4 / Thunderbolt router status (is the enclosure even on the bus?)
      - nvidia-smi output
      - Recent nvlddmkm event log entries

    Run this FIRST, before reinstalling any drivers. Driver reinstallation is the
    folk remedy for every GPU problem and it destroys the evidence.

.EXAMPLE
    .\Get-GpuState.ps1

.EXAMPLE
    .\Get-GpuState.ps1 -OutFile gpu-state.txt
#>
[CmdletBinding()]
param(
    [string]$OutFile
)

function ConvertTo-NvidiaMarketingVersion {
    <#
        Windows shows NVIDIA drivers as e.g. 32.0.16.1692.
        NVIDIA calls that 616.92.

        Rule: concatenate the last two groups, take the final five digits,
        then insert a decimal point before the last two.

            32.0.16.1692 -> "16" + "1692" = "161692" -> "61692" -> 616.92
            32.0.15.9186 -> "15" + "9186" = "159186" -> "59186" -> 591.86
    #>
    param([string]$DriverVersion)

    $parts = $DriverVersion -split '\.'
    if ($parts.Count -lt 4) { return $null }

    $joined = $parts[2] + $parts[3]
    if ($joined.Length -lt 5) { return $null }

    $last5 = $joined.Substring($joined.Length - 5)
    return $last5.Substring(0, 3) + '.' + $last5.Substring(3)
}

function Write-Section {
    param([string]$Title)
    Write-Output ''
    Write-Output ('=' * 70)
    Write-Output "  $Title"
    Write-Output ('=' * 70)
}

$report = & {

    Write-Section 'SYSTEM'
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Output ("Computer     : " + $env:COMPUTERNAME)
    Write-Output ("OS           : " + $os.Caption + " (" + $os.Version + ")")
    Write-Output ("Last boot    : " + $os.LastBootUpTime)
    Write-Output ("Collected    : " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))

    Write-Section 'DISPLAY DEVICES (present only)'
    Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
        Where-Object { $_.Present -eq $true } |
        Select-Object FriendlyName, Status, ConfigManagerErrorCode, InstanceId |
        Format-List

    Write-Section 'PHANTOM / ABSENT DISPLAY RECORDS'
    Write-Output 'CM_PROB_PHANTOM entries are stale records, not live devices.'
    Write-Output 'They are normal if a device is currently unplugged.'
    Write-Output ''
    Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
        Where-Object { $_.Present -ne $true } |
        Select-Object FriendlyName, Status, ConfigManagerErrorCode |
        Format-Table -AutoSize

    Write-Section 'BOUND DISPLAY DRIVERS'
    Write-Output 'If two NVIDIA cards show DIFFERENT DriverVersion values, that is your bug.'
    Write-Output ''
    Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" -ErrorAction SilentlyContinue |
        ForEach-Object {
            [PSCustomObject]@{
                DeviceName    = $_.DeviceName
                DriverVersion = $_.DriverVersion
                Marketing     = ConvertTo-NvidiaMarketingVersion $_.DriverVersion
                DriverDate    = $_.DriverDate
                InfName       = $_.InfName
            }
        } | Format-List

    Write-Section 'NVIDIA DISPLAY PACKAGES IN DRIVER STORE'
    Write-Output 'More than one NVIDIA Display package usually means a version collision.'
    Write-Output ''
    (& pnputil.exe /enum-drivers 2>&1 | Out-String) -split "`r?`n" |
        Select-String -Pattern 'Published Name|Original Name|Class Name|Driver Version' -Context 0, 0 |
        Out-String -Stream |
        Where-Object { $_ -match 'nv_disp|nvlt|nvhda|nvpcf' -or $_ -match 'Published Name' } |
        Select-Object -First 60

    Write-Section 'USB4 / THUNDERBOLT'
    Write-Output 'An enclosure with Status "Unknown" is NOT on the bus.'
    Write-Output 'If nothing appears at all on plug-in, suspect the cable before the driver:'
    Write-Output 'a power-only or USB 2.0 USB-C cable is physically identical to a USB4 one.'
    Write-Output ''
    Get-PnpDevice -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -match 'USB4|Thunderbolt|Razer|Core X' } |
        Select-Object FriendlyName, Status, Class |
        Format-Table -AutoSize

    Write-Section 'NVIDIA-SMI'
    try {
        & nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,pci.bus_id --format=csv 2>&1 | Out-String
    } catch {
        Write-Output "nvidia-smi not available: $($_.Exception.Message)"
    }

    Write-Section 'RECENT nvlddmkm EVENTS'
    Write-Output 'Note: .Message is often empty for these; .ToXml() carries the payload.'
    Write-Output ''
    $events = Get-WinEvent -FilterHashtable @{ LogName = 'System'; ProviderName = 'nvlddmkm' } -MaxEvents 10 -ErrorAction SilentlyContinue
    if ($events) {
        foreach ($e in $events) {
            $xml = [xml]$e.ToXml()
            $data = ($xml.Event.EventData.Data | ForEach-Object { $_ }) -join ' | '
            Write-Output ("[{0}] Id={1} {2}" -f $e.TimeCreated, $e.Id, $data)
        }
    } else {
        Write-Output 'No nvlddmkm events found (this is good).'
    }

    Write-Section 'END'
}

if ($OutFile) {
    $report | Out-File -FilePath $OutFile -Encoding UTF8
    Write-Host "Written to $OutFile"
} else {
    $report
}
