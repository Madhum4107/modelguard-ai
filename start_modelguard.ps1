$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "           ModelGuard AI — ML Observability Lab v1.0.0" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

$venv = Join-Path $PSScriptRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$requirements = Join-Path $PSScriptRoot "requirements.txt"
$logOut = Join-Path $PSScriptRoot "modelguard_server.out.log"
$logErr = Join-Path $PSScriptRoot "modelguard_server.err.log"

if (-not (Test-Path $python)) {
    Write-Host "[1/4] Creating local Python environment..." -ForegroundColor Yellow
    & python -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $python)) {
        throw "Could not create .venv. Make sure Python 3.11+ is installed and available in PATH."
    }
} else {
    Write-Host "[1/4] Local Python environment found." -ForegroundColor Green
}

Write-Host "[2/4] Verifying Python packages..." -ForegroundColor Yellow
& $python -m pip install -r $requirements --disable-pip-version-check --no-input
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check your internet connection and modelguard_server.out.log / modelguard_server.err.log." }

# Pick an available loopback port. This avoids stale local services on 8000.
$port = 8000
while ($port -le 8010) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
        $listener.Start()
        $listener.Stop()
        break
    } catch {
        if ($listener) { try { $listener.Stop() } catch {} }
        $port++
    }
}
if ($port -gt 8010) { throw "Ports 8000-8010 are busy. Close the local service using them and run again." }

Remove-Item $logOut,$logErr -Force -ErrorAction SilentlyContinue
Write-Host "[3/4] Starting API + UI on http://127.0.0.1:$port ..." -ForegroundColor Yellow
$server = Start-Process -FilePath $python `
    -ArgumentList @("-m","uvicorn","backend.main:app","--host","127.0.0.1","--port",$port) `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Milliseconds 250
        if ($server.HasExited) {
            $details = if (Test-Path $logOut) { Get-Content $logOut -Raw } else { "No stdout log was created." }
            throw "ModelGuard server stopped unexpectedly (exit code $($server.ExitCode)).`n$details"
        }
        try {
            $health = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
    }
    if (-not $ready) {
        $details = if (Test-Path $logOut) { Get-Content $logOut -Raw } else { "No stdout log was created." }
        throw "ModelGuard API did not become ready on port $port.`n$details"
    }

    Write-Host "[4/4] ModelGuard is READY." -ForegroundColor Green
    Write-Host "UI:       http://127.0.0.1:$port" -ForegroundColor Green
    Write-Host "API docs: http://127.0.0.1:$port/docs" -ForegroundColor DarkCyan
    Write-Host "Logs:     $logOut / $logErr" -ForegroundColor Gray
    Write-Host "Keep this window open while using the app." -ForegroundColor Gray
    Start-Process "http://127.0.0.1:$port"
    Wait-Process -Id $server.Id
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
