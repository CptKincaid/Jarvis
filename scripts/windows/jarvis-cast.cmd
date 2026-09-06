@echo off
rem Jarvis cast poller -- HIS FILE, on HIS machine. Nothing installs this.
rem Copy it to:
rem   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\jarvis-cast.cmd
rem It starts cast-poll.ps1 hidden, as him, at logon.
start "" /min powershell -NoProfile -WindowStyle Hidden ^
  -ExecutionPolicy Bypass -File "%USERPROFILE%\jarvis\cast-poll.ps1"
