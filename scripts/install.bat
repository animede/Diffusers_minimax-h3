@echo off
rem minimax-h3 installer for Windows. Runs scripts\install.ps1 regardless of the PowerShell execution policy.
rem   scripts\install.bat             basic install (venv + torch cu128 + requirements.txt)
rem   scripts\install.bat -LocalLLM   also fetch the prebuilt llama-server.exe for the built-in LLM
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
