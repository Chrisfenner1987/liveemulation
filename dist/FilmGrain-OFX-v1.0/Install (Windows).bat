@echo off
REM Grain Match OFX - Windows installer.
REM RIGHT-CLICK this file and choose "Run as administrator".
setlocal
set "DEST=C:\Program Files\Common Files\OFX\Plugins"

echo ===============================================
echo  Grain Match - OFX plugin installer
echo ===============================================
echo.

if not exist "%~dp0FilmGrain.ofx.bundle" (
  echo ERROR: FilmGrain.ofx.bundle was not found next to this script.
  echo Keep this .bat in the same folder as the .ofx.bundle.
  pause
  exit /b 1
)

echo Installing to: %DEST%
mkdir "%DEST%" 2>nul
xcopy /E /I /Y "%~dp0FilmGrain.ofx.bundle" "%DEST%\FilmGrain.ofx.bundle" >nul
if %errorlevel%==0 (
  echo.
  echo Installed. Quit and reopen DaVinci Resolve.
  echo Find it under:  OpenFX  ^>  Fenner  ^>  Film Grain ^(Stochastic^)
) else (
  echo.
  echo Install FAILED. Make sure you ran this as Administrator.
)
echo.
pause
