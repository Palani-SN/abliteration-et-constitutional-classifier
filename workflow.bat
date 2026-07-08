@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM WORKFLOW.BAT
REM Runs the full refusal-direction / abliteration pipeline end to end, in order:
REM   1. collect_activations.py - collect per-layer last-token activations for the
REM                                harmfull/harmless train+test prompt sets (loaded
REM                                via load_datasets.PromptSets from dataset/*.xlsx,
REM                                which dataset/store_datasets.py must have already
REM                                curated)
REM   2. compute_direction.py   - compute per-layer mean-difference directions and
REM                                save the single selected layer's direction to
REM                                activations/direction.pt, plus the Cohen's-d
REM                                selected {layers}x{dims} signature (+ empirically
REM                                fit gate_threshold) to activations/signature.pt
REM   3. signature_report.py    - validate the signature (per-layer Cohen's d sweep,
REM                                train-vs-OOD classification coherence, and an
REM                                all-dims/all-layers baseline comparison), writing
REM                                activations/signature_report.html + signature_stats.json
REM   4. abliterate.py          - apply the direction as a runtime ablation hook (no
REM                                model is saved to disk) and print a quick
REM                                LLM-as-judge sanity check on a few OOD prompts
REM   5. classify.py            - two-stage Constitutional-Classifiers++ sanity check
REM                                (FastGate activation probe + Falcon3 ExchangeClassifier)
REM                                over top_n harmful/harmless OOD test prompts
REM   6. verify.py              - batch-compare original vs ablated responses AND run
REM                                the Stage 1/2 classifier on the same held-out OOD
REM                                test prompts, writing
REM                                results/<timestamp>/{harmless,harmfull}.xlsx
REM   7. comparison_report.py   - reads the latest results/<timestamp>/*.xlsx and
REM                                writes a consolidated HTML comparing judgement and
REM                                latency (*_ts) across Original / Abliterated /
REM                                Constitutional Classifier++ to
REM                                results/<timestamp>/comparison_report.html
REM Run setup_env.bat once before this script if the "eip" conda
REM environment has not been created yet.
REM =============================================================================

set ENV_NAME=eip

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda was not found on PATH. Run setup_env.bat first.
    exit /b 1
)

for /f "delims=" %%i in ('conda info --base') do set CONDA_BASE=%%i
call "%CONDA_BASE%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment "%ENV_NAME%". Run setup_env.bat first.
    exit /b 1
)

for /f %%t in ('powershell -NoProfile -Command "[long](Get-Date).Ticks"') do set START_TICKS=%%t

echo ============================================================
echo STAGE 1/7: Collecting activations
echo ============================================================
python collect_activations.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 2/7: Computing the refusal direction and signature
echo ============================================================
python compute_direction.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 3/7: Validating the signature (report + baseline comparison)
echo ============================================================
python signature_report.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 4/7: Abliterating the model (runtime ablation + quick judge check)
echo ============================================================
python abliterate.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 5/7: Two-stage classifier sanity check (FastGate + ExchangeClassifier)
echo ============================================================
python classify.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 6/7: Verifying generalization on held-out OOD prompts
echo ============================================================
python verify.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo STAGE 7/7: Building the comparison report (latest results/ run)
echo ============================================================
python comparison_report.py
if errorlevel 1 goto :fail

call :elapsed
echo.
echo ============================================================
echo PIPELINE COMPLETE
echo   - refusal direction:    activations\direction.pt
echo   - signature + gate:     activations\signature.pt
echo   - signature report:     activations\signature_report.html
echo   - verification reports: results\^<timestamp^>\{harmless,harmfull}.xlsx
echo   - comparison report:    results\^<timestamp^>\comparison_report.html
echo   - total time taken:     %ELAPSED%
echo ============================================================
endlocal
exit /b 0

:fail
call :elapsed
echo.
echo [ERROR] Pipeline stopped due to the error above.
echo   - time elapsed before failure: %ELAPSED%
endlocal
exit /b 1

:elapsed
for /f %%t in ('powershell -NoProfile -Command "[long](Get-Date).Ticks"') do set END_TICKS=%%t
for /f %%e in ('powershell -NoProfile -Command "[TimeSpan]::FromTicks(%END_TICKS%-%START_TICKS%).ToString(\"hh\:mm\:ss\")"') do set ELAPSED=%%e
exit /b 0
