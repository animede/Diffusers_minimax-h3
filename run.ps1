# Starts minimax-h3 on Windows (PowerShell). Default port 8611, bound to 127.0.0.1 only. ASCII only (see install.ps1).
#   .\run.bat                      # http://127.0.0.1:8611/  (runs this script with -ExecutionPolicy Bypass)
# Set environment variables one per line before starting:
#   $env:H3_PORT = "8612"          # change the port
#   $env:H3_HOST = "0.0.0.0"       # expose on the LAN (no authentication)
#   $env:H3_TURBO_LORA = "1"       # turbo LoRA (4 steps); other H3_* variables work the same way
#   .\run.bat
# With nothing set, H3_PROFILE=auto picks the configuration from the GPU's total VRAM.
Set-Location -Path $PSScriptRoot

# use the venv made by scripts\install.ps1 (venv\), or .venv\ if it was made with uv
$python = $null
foreach ($cand in @(".\venv\Scripts\python.exe", ".\.venv\Scripts\python.exe")) {
    if (Test-Path $cand) { $python = $cand; break }
}
if ($null -eq $python) {
    Write-Host "venv not found (neither venv\Scripts\python.exe nor .venv\Scripts\python.exe exists)." -ForegroundColor Yellow
    Write-Host "Run the installer first:"
    Write-Host "    .\scripts\install.bat"
    Write-Host "(if install.bat stopped with an error, check its output; Python 3.12 and git are required)"
    exit 1
}

$port = if ($env:H3_PORT) { $env:H3_PORT } else { "8611" }
$bind = if ($env:H3_HOST) { $env:H3_HOST } else { "127.0.0.1" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
& $python -u -m uvicorn app:app --host $bind --port $port
