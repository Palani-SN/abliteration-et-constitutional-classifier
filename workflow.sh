#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# workflow.sh
# Runs the full refusal-direction / abliteration pipeline end to end, in order:
#   1. collect_activations.py - collect per-layer last-token activations for the
#                                harmfull/harmless train+test prompt sets (loaded
#                                via load_datasets.PromptSets from dataset/*.xlsx,
#                                which dataset/store_datasets.py must have already
#                                curated)
#   2. compute_direction.py   - compute per-layer mean-difference directions and
#                                save the single selected layer's direction to
#                                activations/<model_key>/direction.pt, plus the
#                                Cohen's-d selected {layers}x{dims} signature
#                                (+ empirically fit gate_threshold) to
#                                activations/<model_key>/signature.pt
#   3. signature_report.py    - validate the signature (per-layer Cohen's d sweep,
#                                train-vs-OOD classification coherence, and an
#                                all-dims/all-layers baseline comparison), writing
#                                activations/<model_key>/signature_report.html +
#                                signature_stats.json
#   4. abliterate.py          - apply the direction as a runtime ablation hook (no
#                                model is saved to disk) and print a quick
#                                LLM-as-judge sanity check on a few OOD prompts
#   5. classify.py            - two-stage Constitutional-Classifiers++ sanity check
#                                (FastGate activation probe + ExchangeClassifier,
#                                reusing the same loaded model)
#                                over top_n harmful/harmless OOD test prompts
#   6. verify.py              - batch-compare original vs ablated responses AND run
#                                the Stage 1/2 classifier on the same held-out OOD
#                                test prompts, writing
#                                results/<model_key>/<timestamp>/{harmless,harmfull}.xlsx
#   7. comparison_report.py   - reads the latest results/<model_key>/<timestamp>/*.xlsx
#                                and writes a consolidated HTML comparing judgement and
#                                latency (*_ts) across Original / Abliterated /
#                                Constitutional Classifier++ to
#                                results/<model_key>/<timestamp>/comparison_report.html
#
# Run setup_env.sh once before this script if the "eip" conda environment has
# not been created yet.
#
# Usage: ./workflow.sh [model_key] [top_n]
#   model_key - a key from models.yml (e.g. qwen_3_1p7b). Defaults to the
#               first entry in models.yml when omitted. Every stage writes
#               under activations/<model_key>/ and results/<model_key>/, so
#               different models never share cached activations or results.
#   top_n     - caps verify.py (Stage 6) to top_n harmful + top_n harmless
#               OOD prompts, for a quick end-to-end smoke test. Defaults to
#               the full 100+100 held-out set when omitted.
# =============================================================================

ENV_NAME="eip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ARG=""
[ -n "${1:-}" ] && MODEL_ARG="--model $1"
TOPN_ARG=""
[ -n "${2:-}" ] && TOPN_ARG="--top_n $2"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda was not found on PATH. Run setup_env.sh first."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] Could not locate conda.sh under ${CONDA_BASE}."
    exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "[ERROR] Failed to activate conda environment \"$ENV_NAME\". Run setup_env.sh first."
    exit 1
fi

SECONDS=0
elapsed() { printf '%02d:%02d:%02d' $((SECONDS/3600)) $(((SECONDS%3600)/60)) $((SECONDS%60)); }

on_error() {
    local ec=$?
    echo
    echo "[ERROR] Pipeline stopped due to the error above."
    echo "  - time elapsed before failure: $(elapsed)"
    exit "$ec"
}
trap on_error ERR

echo "============================================================"
echo "STAGE 1/7: Collecting activations"
echo "============================================================"
python collect_activations.py $MODEL_ARG

echo
echo "============================================================"
echo "STAGE 2/7: Computing the refusal direction and signature"
echo "============================================================"
python compute_direction.py $MODEL_ARG

echo
echo "============================================================"
echo "STAGE 3/7: Validating the signature (report + baseline comparison)"
echo "============================================================"
python signature_report.py $MODEL_ARG

echo
echo "============================================================"
echo "STAGE 4/7: Abliterating the model (runtime ablation + quick judge check)"
echo "============================================================"
python abliterate.py $MODEL_ARG

echo
echo "============================================================"
echo "STAGE 5/7: Two-stage classifier sanity check (FastGate + ExchangeClassifier)"
echo "============================================================"
python classify.py $MODEL_ARG

echo
echo "============================================================"
echo "STAGE 6/7: Verifying generalization on held-out OOD prompts"
echo "============================================================"
python verify.py $MODEL_ARG $TOPN_ARG

echo
echo "============================================================"
echo "STAGE 7/7: Building the comparison report (latest results/ run)"
echo "============================================================"
python comparison_report.py $MODEL_ARG

trap - ERR
echo
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "  - refusal direction:    activations/<model_key>/direction.pt"
echo "  - signature + gate:     activations/<model_key>/signature.pt"
echo "  - signature report:     activations/<model_key>/signature_report.html"
echo "  - verification reports: results/<model_key>/<timestamp>/{harmless,harmfull}.xlsx"
echo "  - comparison report:    results/<model_key>/<timestamp>/comparison_report.html"
echo "  - total time taken:     $(elapsed)"
echo "============================================================"
