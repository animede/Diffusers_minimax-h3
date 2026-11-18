# minimax-h3 one-shot installer for Windows (PowerShell). ASCII only: Windows PowerShell 5.1 reads
# scripts in the system code page unless they carry a BOM, so non-ASCII text here breaks parsing.
#
#   .\scripts\install.bat             # venv + torch (CUDA 12.8) + requirements.txt   (runs this script with -ExecutionPolicy Bypass)
#   .\scripts\install.bat -LocalLLM   # also fetch the prebuilt llama-server.exe for the built-in LLM
#
# - Needs Python 3.12 (py launcher) and git. No compiler. If missing:
#     winget install --id Git.Git -e --source winget
#     winget install --id Python.Python.3.12 -e --source winget
# - torch / torchvision / torchaudio come from the PyTorch cu128 index (PyPI's Windows torch is CPU-only).
#   torchao uses PyPI's pure-Python wheel (0.17.0); the C++ extensions are not needed.
# - SageAttention has no Windows build and is skipped; H3_PROFILE=auto sets H3_ATTN_BACKEND=default at startup.
param(
    [switch]$LocalLLM
)
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found (needed to install diffusers pinned to a commit). Install it with: winget install --id Git.Git -e --source winget  (then reopen PowerShell)"
}

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "==> creating venv (Python 3.12)"
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv venv
        if ($LASTEXITCODE -eq 0) { $created = $true }
    }
    if (-not $created) {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($ver -ne "3.12") { throw "Python 3.12 not found (found python $ver). Install it with: winget install --id Python.Python.3.12 -e --source winget  (then reopen PowerShell)" }
        & python -m venv venv
    }
}
$python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "venv creation failed ($python is missing). Run 'py -3.12 -m venv venv' by hand and check its output" }

& $python -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "==> torch 2.9.0 (CUDA 12.8) from the PyTorch index"
& $python -m pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "==> requirements.txt"
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed" }

if ($LocalLLM) {
    Write-Host "==> built-in LLM: fetching the prebuilt llama-server.exe"
    & $python scripts\fetch_llama_server.py
}

Write-Host ""
Write-Host "==> check"
& $python scripts\check_install.py
Write-Host ""
Write-Host "Done. Next, prepare the models (low-VRAM / low-RAM PCs, builds the int8 cache shard by shard):"
Write-Host "    .\venv\Scripts\python.exe scripts\prepare_lowvram.py"
Write-Host "  (48GB+ VRAM and 64GB+ RAM: .\venv\Scripts\python.exe scripts\download_t2va.py also works)"
Write-Host "Start the server:             .\run.bat   (http://127.0.0.1:8611/)"
