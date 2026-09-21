<#
.SYNOPSIS
    Load an Ollama model only if it fits in the VRAM actually attached.

.DESCRIPTION
    `ollama run` will happily accept a model far larger than your GPUs can
    hold. With the eGPU detached that spills into system memory and hangs the
    machine hard enough to need a reboot. This is the same command with the
    check in front of it: Test-ModelFits.ps1 decides, and nothing loads
    unless it passes.

    The check is a separate process on purpose - it is the tested
    implementation, and duplicating its logic here would let the two drift
    into disagreeing about what is safe.

    There is deliberately no -Force. If you believe a model fits because the
    runtime splits proportionally rather than evenly, say so with
    -Strategy Proportional, which is a claim the check can act on. "Load it
    anyway" is not.

.PARAMETER Model
    Model to load, e.g. "qwen2.5-coder:7b".

.PARAMETER Prompt
    Run one prompt and return the answer instead of opening a chat session.

.PARAMETER Pull
    Pull the model first if it is not present. Pulling costs disk, not VRAM,
    so it happens before the fit check - which then has a size to work with
    rather than refusing an unknown model.

.PARAMETER Strategy
    Passed to Test-ModelFits.ps1. Conservative (default) assumes an even
    split across cards; Proportional assumes the summed ceiling.

.PARAMETER HeadroomPercent
    Passed to Test-ModelFits.ps1 (default 20). Use 35-50 for long context,
    where the KV cache is substantial.

.PARAMETER Endpoint
    Ollama endpoint (default http://localhost:11434).

.OUTPUTS
    Exit 0 - loaded (or the prompt was answered).
    Exit 1 - refused: it does not fit. Nothing was loaded.
    Exit 2 - refused: the fit could not be established. Nothing was loaded.

.EXAMPLE
    .\Start-Model.ps1 -Model qwen2.5-coder:7b

.EXAMPLE
    .\Start-Model.ps1 -Model qwen3.8-64k:latest -HeadroomPercent 40
    The wider margin a long-context run deserves. Refuses unless the eGPU is
    attached, which is the point.

.EXAMPLE
    .\Start-Model.ps1 -Model qwen2.5-coder:7b -Prompt "Explain this stack trace"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Model,

    [string]$Prompt,
    [switch]$Pull,

    [ValidateSet('Conservative', 'Proportional', 'Even')]
    [string]$Strategy = 'Conservative',
    [int]$HeadroomPercent = 20,
    [string]$Endpoint = 'http://localhost:11434'
)

$checkScript = Join-Path (Split-Path $PSScriptRoot -Parent) 'diagnostics\Test-ModelFits.ps1'
if (-not (Test-Path $checkScript)) {
    Write-Host "REFUSED: cannot find $checkScript, so the fit cannot be checked." -ForegroundColor Red
    exit 2
}

if ($Pull) {
    Write-Host "==> Pulling $Model (disk only; nothing is loaded into VRAM yet)"
    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        Write-Host "REFUSED: ollama pull failed ($LASTEXITCODE)." -ForegroundColor Red
        exit 2
    }
}

Write-Host "==> Checking $Model fits before loading it"
& pwsh -NoProfile -File $checkScript -Model $Model -Strategy $Strategy -HeadroomPercent $HeadroomPercent -Endpoint $Endpoint
$fit = $LASTEXITCODE
if ($fit -ne 0) {
    Write-Host ''
    Write-Host "NOT LOADING $Model - the fit check exited $fit." -ForegroundColor Red
    if ($fit -eq 1) {
        Write-Host 'Attach the eGPU (cold boot, Restart not Shut down), free VRAM with' -ForegroundColor Red
        Write-Host 'Stop-Model.ps1, or pick a smaller model - see docs/model-picker.md.' -ForegroundColor Red
    }
    exit $fit
}

Write-Host ''
Write-Host "==> Fit check passed; loading $Model"
if ($Prompt) {
    & ollama run $Model $Prompt
} else {
    & ollama run $Model
}
exit $LASTEXITCODE
