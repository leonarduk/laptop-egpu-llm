<#
.SYNOPSIS
    Time a model on a fixed prompt and report tokens/sec and GPU offload.

.DESCRIPTION
    The README says benchmarks are pending. This is the thing that produces
    them, and it reports the number that actually explains a slow model:
    how much of it made it onto the GPU. A model 80% offloaded is not
    "a bit slower" - the CPU layers dominate, and tokens/sec falls off a
    cliff. Reporting speed without offload invites the wrong conclusion.

    Loads the model, so it runs the same fit check Start-Model.ps1 does
    first. A benchmark that hangs the machine is not a benchmark.

    Numbers come from Ollama's own timings rather than a stopwatch:

      prompt eval  - prompt_eval_count / prompt_eval_duration
      generation   - eval_count / eval_duration

    Those are separate because they scale differently: prompt evaluation is
    compute-bound and parallel, generation is memory-bandwidth-bound and
    sequential. Quoting one figure for "speed" hides which one is your
    bottleneck.

.PARAMETER Model
    Model to benchmark.

.PARAMETER Prompt
    Prompt to use. The default is a small code-shaped task, so the number
    reflects the work this machine is actually for.

.PARAMETER Tokens
    Cap on tokens to generate (default 200). Enough to get a stable rate
    without waiting through a long answer.

.PARAMETER Repeat
    Runs to average (default 1). The first run after a load pays for warm-up;
    use 2-3 and read the last figure if you care about steady state.

.PARAMETER Strategy
    Passed to the fit check. Conservative (default) assumes an even split.

.PARAMETER HeadroomPercent
    Passed to the fit check (default 20).

.PARAMETER Endpoint
    Ollama endpoint (default http://localhost:11434).

.OUTPUTS
    Exit 0 - benchmarked.
    Exit 1 - refused: the model does not fit. Nothing was loaded.
    Exit 2 - refused or failed: fit not establishable, or the request failed.

.EXAMPLE
    .\Measure-ModelSpeed.ps1 -Model qwen2.5-coder:7b

.EXAMPLE
    .\Measure-ModelSpeed.ps1 -Model qwen2.5-coder:7b -Repeat 3
    Three runs; read the last for steady-state, the first for cold-load cost.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Model,

    [string]$Prompt = 'Write a Python function that merges two sorted lists into one sorted list, with a docstring.',
    [int]$Tokens = 200,
    [int]$Repeat = 1,

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

Write-Host "==> Checking $Model fits before benchmarking it"
& pwsh -NoProfile -File $checkScript -Model $Model -Strategy $Strategy -HeadroomPercent $HeadroomPercent -Endpoint $Endpoint
$fit = $LASTEXITCODE
if ($fit -ne 0) {
    Write-Host ''
    Write-Host "NOT BENCHMARKING $Model - the fit check exited $fit." -ForegroundColor Red
    exit $fit
}

$body = @{
    model   = $Model
    prompt  = $Prompt
    stream  = $false
    options = @{ num_predict = $Tokens }
} | ConvertTo-Json

$results = @()
for ($i = 1; $i -le $Repeat; $i++) {
    Write-Host ''
    Write-Host "==> Run $i of $Repeat"
    try {
        $r = Invoke-RestMethod -Uri "$Endpoint/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    } catch {
        Write-Host "Request failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 2
    }

    # Durations are nanoseconds.
    $genRate = if ($r.eval_duration -gt 0) { $r.eval_count / ($r.eval_duration / 1e9) } else { 0 }
    $promptRate = if ($r.prompt_eval_duration -gt 0) { $r.prompt_eval_count / ($r.prompt_eval_duration / 1e9) } else { 0 }

    $results += [PSCustomObject]@{
        Run          = $i
        PromptTokPerS = '{0:N1}' -f $promptRate
        GenTokPerS    = '{0:N1}' -f $genRate
        GenTokens     = $r.eval_count
        TotalSec      = '{0:N1}' -f ($r.total_duration / 1e9)
    }
    Write-Host ('    generation {0:N1} tok/s, prompt {1:N1} tok/s, {2:N1}s total' -f $genRate, $promptRate, ($r.total_duration / 1e9))
}

Write-Host ''
$results | Format-Table -AutoSize

# Offload is read after the runs, while the model is still resident - this is
# the figure that explains a disappointing rate.
try {
    $ps = Invoke-RestMethod -Uri "$Endpoint/api/ps" -TimeoutSec 10
    $live = $ps.models | Where-Object { $_.name -eq $Model } | Select-Object -First 1
    if ($live) {
        $total = [double]$live.size
        $vram = [double]$live.size_vram
        $pct = if ($total -gt 0) { 100 * $vram / $total } else { 0 }
        Write-Output ('GPU offload: {0:N0}% ({1:N2} GB of {2:N2} GB in VRAM)' -f $pct, ($vram / 1GB), ($total / 1GB))
        if ($pct -lt 99) {
            Write-Output 'Part of this model is on CPU, which is what is capping the rate above.'
            Write-Output 'Free VRAM (Stop-Model.ps1 -All), attach the eGPU, or use a smaller quant.'
        }
    }
} catch {
    Write-Output 'Could not read GPU offload afterwards.'
}

Write-Output ''
Write-Output 'Generation is memory-bandwidth-bound and prompt eval is compute-bound,'
Write-Output 'so compare like with like when putting these in the README.'
exit 0
