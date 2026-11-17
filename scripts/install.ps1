# minimax-h3 のワンショットインストール(Windows、PowerShell)。
#
#   .\scripts\install.ps1             # venv を作って依存を入れる(CUDA 12.8 用 torch + requirements.txt)
#   .\scripts\install.ps1 -LocalLLM   # さらに内蔵 LLM 用のビルド済み llama-server.exe を取得
#
# - Python 3.12(py ランチャー)と git が必要。コンパイラは不要。無ければ winget で:
#     winget install --id Git.Git -e
#     winget install --id Python.Python.3.12 -e
# - torch / torchvision / torchaudio は PyTorch の cu128 index から入れる(PyPI の Windows 版 torch は CPU 専用)。
#   torchao は PyPI の純 Python wheel(0.17.0)で足りる(C++ 拡張は使わない)。
# - SageAttention は Windows 向けのビルドが無いので入れない。起動時に H3_PROFILE=auto が
#   H3_ATTN_BACKEND=default を自動で入れる。
# - 実行ポリシーで止まる場合: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
param(
    [switch]$LocalLLM
)
$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path $PSScriptRoot "..")

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git が見つかりません(diffusers をコミット固定で入れるのに必要)。winget install --id Git.Git -e で入れて PowerShell を開き直してください"
}

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "==> venv を作成(Python 3.12)"
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv venv
        if ($LASTEXITCODE -eq 0) { $created = $true }
    }
    if (-not $created) {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($ver -ne "3.12") { throw "Python 3.12 が見つかりません(winget install --id Python.Python.3.12 -e で入れて PowerShell を開き直してください。見つかった python は $ver)" }
        & python -m venv venv
    }
}
$python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "venv の作成に失敗しました($python が無い)。py -3.12 -m venv venv を手で実行して出力を確認してください" }

& $python -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "pip の更新に失敗" }

Write-Host "==> torch 2.9.0 (CUDA 12.8) を PyTorch の index から"
& $python -m pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "torch のインストールに失敗" }

Write-Host "==> requirements.txt をインストール"
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt のインストールに失敗" }

if ($LocalLLM) {
    Write-Host "==> 内蔵 LLM: ビルド済み llama-server.exe を取得"
    & $python scripts\fetch_llama_server.py
}

Write-Host ""
Write-Host "==> 確認"
$check = @'
import sys
sys.path.insert(0, ".")
import torch, diffusers, transformers, torchao
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, "torchao", torchao.__version__)
from core import local_llm
exe = local_llm.find_binary()
print("built-in LLM (llama-server):", exe if exe else "not fetched yet -> fetched automatically on first fallback, or install.ps1 -LocalLLM")
'@
& $python -c $check
Write-Host ""
Write-Host "完了。次はモデルの取得:  .\venv\Scripts\python.exe scripts\download_t2va.py"
Write-Host "起動:                    .\run.ps1   (http://127.0.0.1:8611/)"
