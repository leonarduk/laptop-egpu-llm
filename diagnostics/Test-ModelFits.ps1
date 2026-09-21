<#
.SYNOPSIS
    Refuse to load a local model that does not fit in the GPU memory actually
    present right now. Run it before starting a model, not after.

.DESCRIPTION
    A model bigger than available VRAM does not degrade gracefully here. With
    the eGPU detached, asking for a 27B model (11.3 GB against 7.93 GB of
    internal card) spilled into system memory and hung the machine hard
    enough to need a reboot. This answers the question beforehand, cheaply:
    does what I am about to load actually fit?

    It checks the thing that breaks - the fit - rather than looking for a
    named device. An absent eGPU is caught because total free VRAM drops, but
    so is a model already resident holding memory, a driver that fell back to
    one card (see Test-DriverConflict.ps1), or a larger model pulled later.
    Looking only for the eGPU would catch the first of those and miss the
    rest.

    Nothing here loads a model. It reads nvidia-smi and asks the runtime for
    the size of something already pulled. That is the whole point: the
    question has to be answered before anything can allocate.

.PARAMETER Model
    Models to check, e.g. "qwen3.8-64k:latest". Defaults to every model
    Ollama has pulled, which answers "what can I run right now?".

.PARAMETER Strategy
    How to combine VRAM across cards, which matters because the cards here
    are asymmetric (7.93 GB + 15.90 GB):

      Conservative (default) - assume the runtime may split evenly, so the
        ceiling is (number of GPUs) x (smallest free). For this pair that is
        ~15.9 GB, not 23.8 GB. LM Studio's default Strategy is "Split
        evenly", and per docs/lmstudio-multi-gpu.md that caps you at twice
        the smaller card however big the other one is.

      Proportional - assume layers are placed in proportion to free memory,
        so the ceiling is the sum. Ollama/llama.cpp does this by default.
        Use it when you know the runtime is placing proportionally.

      Even - force the even-split ceiling even on a single GPU.

    Conservative is the default because guessing high is what hangs the
    machine, and guessing low only makes you check twice.

.PARAMETER HeadroomPercent
    Extra VRAM required beyond the model's file size (default 20).

    THE FILE SIZE IS NOT THE WHOLE FOOTPRINT. The KV cache lives in VRAM
    beside the weights and grows with context length - and, per
    docs/lmstudio-multi-gpu.md, is multiplied by n_slots, so four parallel
    slots quadruple it. A 64k-context run wants a much wider margin than the
    default; 35-50% is more honest there, or quantise the KV cache to Q8_0.
    This script does not try to compute the KV cache exactly, because a
    confident wrong number would be worse than an explicit margin.

.PARAMETER Endpoint
    Ollama endpoint (default http://localhost:11434).

.OUTPUTS
    Exit 0 - everything checked fits.
    Exit 1 - at least one model does not. Do not load it.
    Exit 2 - the question could not be answered (no nvidia-smi, no GPU,
             runtime unreachable, model not pulled). Also not safe to load.

.EXAMPLE
    .\Test-ModelFits.ps1
    What can I run right now, with the hardware currently attached?

.EXAMPLE
    .\Test-ModelFits.ps1 -Model qwen3.8-64k:latest -HeadroomPercent 40
    Check one model with the wider margin a long-context run deserves.

.EXAMPLE
    .\Test-ModelFits.ps1 -Model qwen3.8-64k:latest -Strategy Proportional
    Check against the summed ceiling, having confirmed the runtime places
    layers proportionally rather than evenly.
#>
[CmdletBinding()]
param(
    [string[]]$Model,
    [ValidateSet('Conservative', 'Proportional', 'Even')]
    [string]$Strategy = 'Conservative',
    [int]$HeadroomPercent = 20,
    [string]$Endpoint = 'http://localhost:11434'
)

function Write-Section {
    param([string]$Title)
    Write-Output ''
    Write-Output ('=' * 70)
    Write-Output "  $Title"
    Write-Output ('=' * 70)
}

function Stop-Unsafe {
    param([string]$Message, [int]$Code = 1)
    Write-Output ''
    Write-Host "REFUSED: $Message" -ForegroundColor Red
    exit $Code
}

# --- what is actually attached -------------------------------------------
Write-Section 'GPUs PRESENT'

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Stop-Unsafe 'nvidia-smi not found, so free VRAM cannot be established. Not loading anything onto an unknown GPU setup.' 2
}

$gpus = @()
try {
    $raw = & nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi exited $LASTEXITCODE : $raw" }
    foreach ($line in $raw) {
        $f = $line -split '\s*,\s*'
        if ($f.Count -ge 4) {
            $gpus += [PSCustomObject]@{
                Index   = [int]$f[0]
                Name    = $f[1]
                TotalGB = [double]$f[2] / 1024
                FreeGB  = [double]$f[3] / 1024
            }
        }
    }
} catch {
    Stop-Unsafe "could not read nvidia-smi ($($_.Exception.Message)). Not loading anything onto an unknown GPU setup." 2
}

if ($gpus.Count -eq 0) {
    Stop-Unsafe 'nvidia-smi reported no GPUs at all. Run Get-GpuState.ps1 - this is a driver or enclosure problem, not a sizing one.' 2
}

$gpus | Select-Object Index, Name,
    @{ n = 'TotalGB'; e = { '{0:N2}' -f $_.TotalGB } },
    @{ n = 'FreeGB';  e = { '{0:N2}' -f $_.FreeGB  } } |
    Format-Table -AutoSize | Out-String -Stream | Where-Object { $_.Trim() }

$sumFree = ($gpus | Measure-Object -Property FreeGB -Sum).Sum
$minFree = ($gpus | Measure-Object -Property FreeGB -Minimum).Minimum
$evenCeiling = $minFree * $gpus.Count

switch ($Strategy) {
    'Proportional' { $ceiling = $sumFree }
    'Even'         { $ceiling = $evenCeiling }
    default        { $ceiling = [Math]::Min($sumFree, $evenCeiling) }
}

Write-Output ''
Write-Output ('Free, summed            : {0:N2} GB' -f $sumFree)
if ($gpus.Count -gt 1) {
    Write-Output ('Free, even-split ceiling: {0:N2} GB  ({1} x {2:N2} GB, the smallest card)' -f $evenCeiling, $gpus.Count, $minFree)
    if ($evenCeiling -lt $sumFree - 0.01) {
        Write-Output ''
        Write-Output 'These cards are asymmetric. An even split wastes the larger one -'
        Write-Output 'see docs/lmstudio-multi-gpu.md for changing Strategy away from'
        Write-Output '"Split evenly" so the bigger card carries proportionally more.'
    }
}
Write-Output ''
Write-Output ('Budget used ({0}): {1:N2} GB' -f $Strategy, $ceiling)

# The eGPU is the only way this machine gets past ~7.9 GB, so a single card
# is worth calling out with the fix that actually works - hot-plugging does
# not (Code 12 is PCIe address-space exhaustion, allocated at POST).
if ($gpus.Count -eq 1) {
    Write-Output ''
    Write-Output 'Only one GPU is present. If the eGPU should be attached:'
    Write-Output '  - Hot-plugging will not fix it. Cold boot with the enclosure attached.'
    Write-Output '  - On Windows 11 that means Restart, not Shut down: with Fast Startup,'
    Write-Output '    "Shut down" is a hybrid hibernate and does not force a full POST.'
    Write-Output '  - If it is attached and still missing, run Test-DriverConflict.ps1;'
    Write-Output '    two NVIDIA driver versions leave one card on Code 31.'
}

# --- what each model needs ------------------------------------------------
Write-Section 'MODELS'

$tags = $null
try {
    $tags = Invoke-RestMethod -Uri "$Endpoint/api/tags" -TimeoutSec 10
} catch {
    Stop-Unsafe "Ollama is not reachable at $Endpoint ($($_.Exception.Message)), so model sizes cannot be established." 2
}

if (-not $Model -or $Model.Count -eq 0) {
    $Model = @($tags.models | Sort-Object size -Descending | ForEach-Object { $_.name })
    if ($Model.Count -eq 0) { Stop-Unsafe "Ollama at $Endpoint has no models pulled." 2 }
}

$factor = 1 + ($HeadroomPercent / 100)
$results = @()
foreach ($name in $Model) {
    $entry = $tags.models | Where-Object { $_.name -eq $name } | Select-Object -First 1
    if (-not $entry) {
        Stop-Unsafe "'$name' is not pulled on this machine, so its size is unknown (ollama pull $name)." 2
    }
    $sizeGB = [double]$entry.size / 1GB
    $needGB = $sizeGB * $factor
    $results += [PSCustomObject]@{
        Model   = $name
        SizeGB  = '{0:N2}' -f $sizeGB
        NeedGB  = '{0:N2}' -f $needGB
        Verdict = if ($needGB -le $ceiling) { 'fits' } else { 'TOO BIG' }
        ShortGB = if ($needGB -le $ceiling) { '' } else { '{0:N2}' -f ($needGB - $ceiling) }
        _need   = $needGB
    }
}

$results | Select-Object Model, SizeGB, NeedGB, Verdict, ShortGB |
    Format-Table -AutoSize | Out-String -Stream | Where-Object { $_.Trim() }

Write-Output ''
Write-Output ("Need = file size + {0}% headroom, against a {1:N2} GB budget." -f $HeadroomPercent, $ceiling)

$tooBig = @($results | Where-Object { $_.Verdict -eq 'TOO BIG' })

# An explicit -Model list is a request to load those; refuse if any is too
# big. With no list this is a survey of everything pulled, where some models
# being too big is the expected answer, not a failure.
if ($PSBoundParameters.ContainsKey('Model') -and $tooBig.Count -gt 0) {
    $names = ($tooBig | ForEach-Object { $_.Model }) -join ', '
    Stop-Unsafe "$names will not fit in $('{0:N2}' -f $ceiling) GB. Do not load it - this is the configuration that hangs the machine." 1
}

if ($tooBig.Count -gt 0) {
    Write-Output ''
    Write-Output ("{0} of {1} pulled models do not fit right now." -f $tooBig.Count, $results.Count)
}

Write-Output ''
Write-Host 'OK - nothing requested exceeds the VRAM present.' -ForegroundColor Green
exit 0
