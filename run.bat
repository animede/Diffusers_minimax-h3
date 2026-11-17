@echo off
rem Starts minimax-h3 on Windows. Runs run.ps1 regardless of the PowerShell execution policy.
rem Set environment variables first, e.g.  set H3_TURBO_LORA=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
