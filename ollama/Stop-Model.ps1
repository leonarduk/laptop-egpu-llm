<#
.SYNOPSIS
    Unload a model to free its VRAM, without stopping the Ollama server.

.DESCRIPTION
    Ollama keeps a model resident for five minutes after its last request.
    That is helpful until you want to load something bigger, at which point
    the previous model is simply holding memory you need - and on 7.93 GB of
    internal card, one resident 6 GB model is the difference between the next
    load fitting and not.

    Unloading is a normal request with `keep_alive: 0`, not a kill: the
    server stays up and the model reloads on next use. Nothing is deleted.

.PARAMETER Model
    Model to unload. Ignored when -All is given.

.PARAMETER All
    Unload every currently resident model.

.PARAMETER Endpoint
    Ollama endpoint (default http://localhost:11434).

.OUTPUTS
    Exit 0 - unloaded, or there was nothing resident.
    Exit 2 - Ollama unreachable, or the unload was rejected.

.EXAMPLE
    .\Stop-Model.ps1 -All

.EXAMPLE
    .\Stop-Model.ps1 -Model qwen2.5-coder:14b
#>
[CmdletBinding(DefaultParameterSetName = 'One')]
param(
    [Parameter(ParameterSetName = 'One', Mandatory, Position = 0)]
    [string]$Model,

    [Parameter(ParameterSetName = 'All', Mandatory)]
    [switch]$All,

    [string]$Endpoint = 'http://localhost:11434'
)

function Invoke-Unload {
    param([string]$Name)
    # keep_alive 0 tells the server to drop it as soon as this returns. An
    # empty prompt means no generation happens - this is an eviction, not a
    # request that happens to be short.
    $body = @{ model = $Name; prompt = ''; keep_alive = 0 } | ConvertTo-Json
    Invoke-RestMethod -Uri "$Endpoint/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 30 | Out-Null
}

try {
    $ps = Invoke-RestMethod -Uri "$Endpoint/api/ps" -TimeoutSec 10
} catch {
    Write-Host "Ollama is not reachable at $Endpoint ($($_.Exception.Message))." -ForegroundColor Red
    exit 2
}

$resident = @($ps.models | ForEach-Object { $_.name })
if ($resident.Count -eq 0) {
    Write-Output 'Nothing resident - all VRAM is already free.'
    exit 0
}

$targets = if ($All) { $resident } else { @($Model) }

foreach ($name in $targets) {
    if ($resident -notcontains $name) {
        Write-Output "$name is not resident; nothing to unload."
        continue
    }
    Write-Output "Unloading $name ..."
    try {
        Invoke-Unload -Name $name
    } catch {
        Write-Host "Failed to unload ${name}: $($_.Exception.Message)" -ForegroundColor Red
        exit 2
    }
}

Start-Sleep -Milliseconds 500
try {
    $after = Invoke-RestMethod -Uri "$Endpoint/api/ps" -TimeoutSec 10
    $held = if ($after.models) { (($after.models | Measure-Object -Property size_vram -Sum).Sum) / 1GB } else { 0 }
    Write-Output ''
    Write-Output ('{0:N2} GB still held by {1} resident model(s).' -f $held, @($after.models).Count)
} catch {
    # The unload itself succeeded; failing to re-read the state afterwards is
    # not a reason to report failure.
    Write-Output 'Unloaded (could not re-read resident state).'
}
exit 0
