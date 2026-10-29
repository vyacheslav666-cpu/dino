@echo off
chcp 65001 >nul
rem Запускает веб-сервер из папки проекта и открывает просмотр DQN в браузере.
cd /d "%~dp0"
set PORT=8765
set URL=http://localhost:%PORT%/rl/web/

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

rem Сервер уже запущен — просто открыть страницу.
netstat -ano | findstr /r /c:"127.0.0.1:%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  start "" "%URL%"
  exit /b
)

start "" /b powershell -NoProfile -Command "Start-Sleep 1; Start-Process '%URL%'"
echo Динозавр: %URL%
echo Чтобы остановить сервер, закройте это окно.
echo.
"%PY%" -m http.server %PORT% --bind 127.0.0.1
pause
