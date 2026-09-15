@echo off
REM Koakumix desktop shell -- double-click this file, or call it with extra options:
REM     koakumix-desktop.cmd --model ..\models\qwen3-0.6b-q8_0.gguf
REM It runs the module with the repository's own virtualenv from the repository root,
REM so it works no matter which directory it is invoked from (and does not depend on
REM an editable install having succeeded).
setlocal
set "ROOT=%~dp0.."
set "PY="
if exist "%ROOT%\.venv-test\Scripts\python.exe" set "PY=%ROOT%\.venv-test\Scripts\python.exe"
if not defined PY if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY (
    echo Koakumix: no virtualenv found next to "%ROOT%".
    echo Create one first:
    echo     python -m venv .venv-test
    echo     .venv-test\Scripts\pip install -e ".[desktop]"
    pause
    exit /b 2
)
pushd "%ROOT%"
"%PY%" -m harness_workbench.desktop %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
