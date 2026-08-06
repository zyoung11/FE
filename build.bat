@echo off
call "%~dp0build-kernel.bat" >nul 2>&1
odin build . -o:speed
if errorlevel 1 (
    echo [-] build failed
    exit /b 1
)
echo [+] FE.exe built
