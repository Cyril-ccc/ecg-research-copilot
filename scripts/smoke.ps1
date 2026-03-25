$ErrorActionPreference = "Stop"

function Assert($cond, $msg) {
  if (-not $cond) {
    Write-Error "ASSERT FAILED: $msg"
    exit 1
  }
}

Write-Host "[smoke] docker compose up..."
docker compose up --build -d | Out-Null

# 等待 /health 可用（最多尝试 30 次，每次 1 秒）
$ok = $false
for ($i=0; $i -lt 30; $i++) {
  try {
    $resp = Invoke-RestMethod -Method GET -Uri "http://localhost:8000/health" -TimeoutSec 2
    if ($resp.ok -eq $true) { $ok = $true; break }
  } catch {}
  Start-Sleep -Seconds 1
}
Assert $ok "API /health not ready"

Write-Host "[smoke] create run..."
$runBody = @{
  question = "smoke test run"
  params   = @{ k_threshold = 5.5; window_hours = 6 }
} | ConvertTo-Json -Depth 5

$runResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/runs" -ContentType "application/json" -Body $runBody
Assert ($runResp.run_id -ne $null -and $runResp.run_id.Length -gt 10) "POST /runs did not return run_id"
$runId = $runResp.run_id
Write-Host "[smoke] run_id = $runId"

Write-Host "[smoke] run safe sql..."
$sqlOkBody = @{ sql = "SELECT 1 as one"; limit = 10; run_id = $runId } | ConvertTo-Json -Depth 5
$sqlOkResp = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/tools/run_sql" -ContentType "application/json" -Body $sqlOkBody
Assert ($sqlOkResp.ok -eq $true) "run_sql ok=false"
Assert ($sqlOkResp.rows[0].one -eq 1) "run_sql did not return one=1"

Write-Host "[smoke] run unsafe sql (should fail)..."
$failed = $false
try {
  $sqlBadBody = @{ sql = "DROP TABLE runs"; limit = 10; run_id = $runId } | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Method POST -Uri "http://localhost:8000/tools/run_sql" -ContentType "application/json" -Body $sqlBadBody | Out-Null
} catch {
  $failed = $true
}
Assert $failed "unsafe SQL was not rejected"

Write-Host "[smoke] check audit logs..."
# 检查最近是否有 run_sql_request（allowed=false）记录
$cntBad = docker compose exec -T db psql -U ecg -d ecg -t -A -c "select count(*) from audit_logs where action='run_sql_request' and payload->>'allowed'='false';"
$cntBad = [int]$cntBad.Trim()
Assert ($cntBad -ge 1) "audit_logs missing rejected run_sql_request"

# 检查 runs 表里能查到这条 run
$cntRun = docker compose exec -T db psql -U ecg -d ecg -t -A -c "select count(*) from runs where run_id='$runId'::uuid;"
$cntRun = [int]$cntRun.Trim()
Assert ($cntRun -eq 1) "runs table missing created run_id"

Write-Host "[smoke] PASS"
exit 0