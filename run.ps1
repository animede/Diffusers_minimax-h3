# minimax-h3 を Windows(PowerShell)で起動する。既定ポート 8611、127.0.0.1 のみで待ち受け。
#   .\run.ps1                     # http://127.0.0.1:8611/  何も指定しなければ H3_PROFILE=auto が GPU の総 VRAM から構成を選ぶ
# 環境変数は起動前に 1 行ずつ設定する:
#   $env:H3_PORT = "8612"         # ポート変更
#   $env:H3_HOST = "0.0.0.0"      # LAN に公開(認証なし)
#   $env:H3_TURBO_LORA = "1"      # turbo LoRA(4 ステップ)。他の H3_* も同様
#   .\run.ps1
# 実行ポリシーで止まる場合: powershell -ExecutionPolicy Bypass -File .\run.ps1
Set-Location -Path $PSScriptRoot
$port = if ($env:H3_PORT) { $env:H3_PORT } else { "8611" }
$bind = if ($env:H3_HOST) { $env:H3_HOST } else { "127.0.0.1" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
& ".\venv\Scripts\python.exe" -u -m uvicorn app:app --host $bind --port $port
