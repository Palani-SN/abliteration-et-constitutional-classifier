@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM SETUP_ENV.BAT
REM Creates (if needed) and provisions the "eip" conda environment.
REM PyTorch (CUDA 12.4) is installed manually first, then the remaining
REM dependencies are installed from reqs.txt.
REM =============================================================================

set ENV_NAME=eip
set PYTHON_VERSION=3.11.6
set SCRIPT_DIR=%~dp0
set REQS_FILE=%SCRIPT_DIR%reqs.txt

if not exist "%REQS_FILE%" (
    echo [ERROR] Could not find reqs.txt at "%REQS_FILE%".
    exit /b 1
)

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found on PATH. Install Miniconda/Miniforge and retry.
    exit /b 1
)

for /f "delims=" %%i in ('conda info --base') do set CONDA_BASE=%%i
if not exist "%CONDA_BASE%\Scripts\activate.bat" (
    echo [ERROR] Could not locate activate.bat under "%CONDA_BASE%".
    exit /b 1
)

conda env list | findstr /b /c:"%ENV_NAME% " >nul
if errorlevel 1 (
    echo [1/4] Creating conda environment "%ENV_NAME%" ^(python %PYTHON_VERSION%^) ...
    call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
) else (
    echo [1/4] Conda environment "%ENV_NAME%" already exists - skipping creation.
)

echo [2/4] Activating "%ENV_NAME%" ...
call "%CONDA_BASE%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate "%ENV_NAME%".
    exit /b 1
)

echo [3/4] Installing PyTorch ^(CUDA 12.4^) ...
pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    exit /b 1
)

echo [4/4] Installing remaining dependencies from reqs.txt ...
pip install -r "%REQS_FILE%"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

echo.
echo Verifying installation ...
python -c "import torch, transformers, bitsandbytes, plotly, pandas, openai, scipy; print(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()}'); print(f'transformers {transformers.__version__}'); print(f'bitsandbytes {bitsandbytes.__version__}'); print(f'plotly {plotly.__version__}'); print(f'pandas {pandas.__version__}'); print(f'openai {openai.__version__}'); print(f'scipy {scipy.__version__}')"
if errorlevel 1 (
    echo [ERROR] Verification import failed.
    exit /b 1
)

echo.
echo Environment "%ENV_NAME%" is ready.
echo NOTE: If tiiuae/Falcon3-1B-Instruct or a gated model you swap in requires
echo       Hugging Face access, authenticate first ^(huggingface-cli login^).
echo NOTE: The judge requires Ollama + gemma4:e4b. Run:
echo         ollama serve
echo         ollama pull gemma4:e4b
endlocal
