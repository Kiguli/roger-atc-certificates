<#
Semantic-equivalence campaign (floors 0.90 / 0.95) over both ATC corpora.

Phase 1 = certificate splits (what the certified G curves and the figure need),
phase 2 = ablation splits (held-out validation of the semantic certificates,
and reusable for later documents).  Within each phase the models run cheapest
first, so partial results are useful early and the 1.5B-parameter Whisper-ATC
never blocks the others.

Everything is logged to _sem_campaign.log (append; one line per utterance from
the runner itself, plus the per-combo banners below).  Each combo is
resumable: run this script again and finished utterances are skipped.

  ./run_all_atc_sem.ps1                     # everything
  ./run_all_atc_sem.ps1 -Phase certificate  # just phase 1
#>
param(
  [ValidateSet("all", "certificate", "ablation")]
  [string]$Phase = "all"
)

$ErrorActionPreference = "Continue"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE
$PY = Join-Path $HERE "..\.venv\Scripts\python.exe"
$LOG = Join-Path $HERE "_sem_campaign.log"

function Log($msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $LOG -Value $line -Encoding utf8
}

# cheapest first; whatc (whisper-large-v3 fine-tune) is ~50x slower per decode
$MODELS = @("qn", "ds2", "wh", "whatc")
$CORPORA = @("atcosim", "atco2")
$ROLES = if ($Phase -eq "all") { @("certificate", "ablation") } else { @($Phase) }

Log "=== semantic campaign start (phase=$Phase) ==="
Log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
$t0 = Get-Date

foreach ($role in $ROLES) {
  foreach ($m in $MODELS) {
    foreach ($ds in $CORPORA) {
      $tag = "$m/$ds/$role"
      Log "--- BEGIN $tag ---"
      $ts = Get-Date
      & $PY run_atc_sem.py $m $ds $role 1000 2>&1 |
        ForEach-Object { Add-Content -Path $LOG -Value $_ -Encoding utf8 }
      $code = $LASTEXITCODE
      $mins = [math]::Round(((Get-Date) - $ts).TotalMinutes, 1)
      if ($code -ne 0) {
        Log "--- FAILED $tag (exit $code, ${mins}m) -- continuing to next combo ---"
      } else {
        Log "--- DONE $tag (${mins}m) ---"
      }
    }
  }
  Log "=== phase '$role' complete ==="
}

$hrs = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
Log "=== semantic campaign finished in ${hrs}h ==="
