@echo off
setlocal

where pythonw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pythonw "%~dp0..\insta360_frame_extractor_gui.py"
    exit /b %ERRORLEVEL%
)

where pyw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pyw -3 "%~dp0..\insta360_frame_extractor_gui.py"
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python "%~dp0..\insta360_frame_extractor_gui.py"
    exit /b %ERRORLEVEL%
)

py -3 "%~dp0..\insta360_frame_extractor_gui.py"
