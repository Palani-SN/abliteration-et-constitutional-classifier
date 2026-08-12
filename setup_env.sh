#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# setup_env.sh
# Bootstraps this repo's "eip" conda environment on a Linux GPU instance:
#   1. Miniforge, only if conda isn't already on PATH (never overrides an
#      existing conda/miniforge install -- just activates on top of it)
#   2. the "eip" conda environment (python 3.11.6)
#   3. PyTorch (CUDA 12.4, matching the pin in reqs.txt), then the rest of
#      reqs.txt
#   4. Ollama + gemma4:e4b (the LLM-as-Judge backend used throughout the
#      pipeline)
#   5. Hugging Face login, if $HF_TOKEN is exported (some models.yml entries
#      are gated) and a pre-download of the default/selected model
# =============================================================================

ENV_NAME="eip"
PYTHON_VERSION="3.11.6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS_FILE="${SCRIPT_DIR}/reqs.txt"
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-$HOME/miniforge3}"

if [ ! -f "$REQS_FILE" ]; then
    echo "[ERROR] Could not find reqs.txt at $REQS_FILE"
    exit 1
fi

echo "Checking for an NVIDIA GPU ..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "[WARN] nvidia-smi not found -- continuing, but torch.cuda.is_available() will likely be False."
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[1/6] conda not found -- installing Miniforge to ${MINIFORGE_PREFIX} ..."
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_PREFIX"
    rm -f /tmp/miniforge.sh
    export PATH="${MINIFORGE_PREFIX}/bin:${PATH}"
else
    echo "[1/6] conda already on PATH ($(command -v conda)) -- using the existing install, skipping Miniforge."
fi

CONDA_BASE="$(conda info --base)"
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] Could not locate conda.sh under ${CONDA_BASE}."
    exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# So future interactive shells (e.g. a new SSH session) can run `conda activate`
# directly without hitting "Run 'conda init' before 'conda activate'". Only
# affects shells opened after this; the current shell still needs the source
# line above. Non-fatal: harmless if it can't write the rc file.
conda init bash >/dev/null 2>&1 || true

create_env() {
    # -c conda-forge --override-channels: stay off Anaconda's "defaults" channel
    # (pkgs/main, pkgs/r) entirely, since it now requires a one-time interactive
    # ToS acceptance that would otherwise break this script on a fresh machine.
    # `pip` must be listed explicitly -- conda-forge's python package doesn't
    # bundle it, so without this the env ends up with no pip of its own and
    # `pip install` silently falls through PATH to some *other* env's pip.
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -c conda-forge --override-channels -y
}

env_is_valid() {
    local py
    py="$(conda run -n "$ENV_NAME" python --version 2>&1 | awk '{print $2}')"
    [ "$py" = "$PYTHON_VERSION" ] || return 1
    conda run -n "$ENV_NAME" python -m pip --version >/dev/null 2>&1 || return 1
}

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if env_is_valid; then
        echo "[2/6] Conda environment \"$ENV_NAME\" already exists with python $PYTHON_VERSION and pip - skipping creation."
    else
        echo "[2/6] Conda environment \"$ENV_NAME\" exists but is missing python $PYTHON_VERSION or pip -- recreating ..."
        conda env remove -n "$ENV_NAME" -y
        create_env
    fi
else
    echo "[2/6] Creating conda environment \"$ENV_NAME\" (python $PYTHON_VERSION) ..."
    create_env
fi

echo "[3/6] Activating \"$ENV_NAME\" ..."
conda activate "$ENV_NAME"

# Guard against activation "succeeding" but python/pip still resolving to some
# other env on PATH.
for bin in python pip; do
    resolved="$(command -v "$bin")"
    case "$resolved" in
        "$CONDA_PREFIX"/*) ;;
        *)
            echo "[ERROR] $bin resolved to $resolved, which is outside \$CONDA_PREFIX ($CONDA_PREFIX)."
            echo "        Activation did not take effect as expected -- aborting instead of installing into the wrong env."
            exit 1
            ;;
    esac
done

echo "[4/6] Installing PyTorch (CUDA 12.4) ..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "[5/6] Installing remaining dependencies from reqs.txt ..."
pip install -r "$REQS_FILE" --extra-index-url https://download.pytorch.org/whl/cu124

echo "[6/6] Ensuring Ollama + gemma4:e4b are ready ..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "      ollama not found -- installing ..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "      starting \"ollama serve\" in the background ..."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    disown 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -fsS http://localhost:11434/api/version >/dev/null 2>&1 && break
        sleep 1
    done
fi
ollama pull gemma4:e4b

if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to Hugging Face with \$HF_TOKEN ..."
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
fi

echo "Pre-downloading model weights (MODEL_KEY env var selects the models.yml"
echo "entry; unset defaults to the first entry, falcon_3_1b) ..."
TARGET_MODEL_ID="$(MODEL_KEY_ENV="${MODEL_KEY:-}" python -c "
import os, sys
sys.path.insert(0, '${SCRIPT_DIR}')
from models import resolve_model
key = os.environ.get('MODEL_KEY_ENV') or None
_, model_id = resolve_model(key)
print(model_id)
")"
hf download "$TARGET_MODEL_ID" || echo "[WARN] Could not pre-download $TARGET_MODEL_ID -- if it's gated, log in and accept its license at https://huggingface.co/$TARGET_MODEL_ID, then rerun: hf download $TARGET_MODEL_ID"

echo
echo "Verifying installation ..."
python -c "
from importlib.metadata import version
import torch, transformers, bitsandbytes, plotly, pandas, openai, scipy, yaml, accelerate, openpyxl
print(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')
print(f'transformers {transformers.__version__}')
print(f'bitsandbytes {bitsandbytes.__version__}')
print(f'plotly {plotly.__version__}')
print(f'pandas {pandas.__version__}')
print(f'openai {openai.__version__}')
print(f'scipy {scipy.__version__}')
print(f'PyYAML {yaml.__version__}')
print(f'accelerate {accelerate.__version__}')
print(f'openpyxl {openpyxl.__version__}')
"

echo
echo "Environment \"$ENV_NAME\" is ready."
echo "NOTE: gemma_1p1_2b, gemma_1p1_7b, and llama_3_8b in models.yml are gated on"
echo "      Hugging Face. Export HF_TOKEN before running this script to log in"
echo "      automatically, or run 'hf auth login' manually, after accepting each"
echo "      model's license on its Hugging Face page."
echo "NOTE: The judge requires Ollama + gemma4:e4b, already pulled above. If"
echo "      \"ollama serve\" isn't running in a future shell, start it with:"
echo "        ollama serve"
