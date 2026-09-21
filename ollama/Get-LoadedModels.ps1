<#
.SYNOPSIS
    Show which models are currently resident and how much VRAM they hold.

.DESCRIPTION
    Answers "why doesn't my model fit, there's nothing running?" - usually
    because something from ten minutes ago is still loaded. Ollama keeps a
    model resident for five minutes after its last request by default, so
    free VRAM is often lower than you expect.

    The useful column is Offload: what fraction of the model actually made it
    onto the GPU. Anything under 100% means layers are running on CPU, which
    is the slow-but-alive outcome - and a sign you are near the edge of what
    fits.

.PARAMETER Endpoint
    Ollama endpoint (default http://localhost:11434).

.OUTPUTS
    Exit 0 - listed (including "nothing resident").
    Exit 2 - Ollama unreachable.

.EXAMPLE
    .\Get-LoadedModels.ps1
#>
[CmdletBinding()]
param(
    [string]$Endpoint = 'http://localhost:11434'
)

try {
    $ps = Invoke-RestMethod -Uri "$Endpoint/api/ps" -TimeoutSec 10
} catch {
    Write-Host "Ollama is not reachable at $Endpoint ($($_.Exception.Message))." -ForegroundColor Red
    exit 2
}

if (-not $ps.models -or $ps.models.Count -eq 0) {
    Write-Output 'Nothing resident - all VRAM is free for the next load.'
    exit 0
}

$rows = foreach ($m in $ps.models) {
    $total = [double]$m.size
    $vram = [double]$m.size_vram
    [PSCustomObject]@{
        Model    = $m.name
        TotalGB  = '{0:N2}' -f ($total / 1GB)
        VramGB   = '{0:N2}' -f ($vram / 1GB)
        Offload  = if ($total -gt 0) { '{0:N0}%' -f (100 * $vram / $total) } else { 'n/a' }
        Expires  = if ($m.expires_at) { ([datetime]$m.expires_at).ToString('HH:mm:ss') } else { '' }
    }
}
$rows | Format-Table -AutoSize

$heldGB = (($ps.models | Measure-Object -Property size_vram -Sum).Sum) / 1GB
Write-Output ('{0:N2} GB of VRAM is held by {1} resident model(s).' -f $heldGB, $ps.models.Count)

if ($ps.models | Where-Object { [double]$_.size -gt 0 -and [double]$_.size_vram -lt [double]$_.size }) {
    Write-Output ''
    Write-Output 'At least one model is partly on CPU. It will run, slowly. If that is'
    Write-Output 'not deliberate, free VRAM (Stop-Model.ps1) or drop to a smaller model.'
}

Write-Output ''
Write-Output 'Free it before loading something bigger:  .\Stop-Model.ps1 -All'
exit 0
