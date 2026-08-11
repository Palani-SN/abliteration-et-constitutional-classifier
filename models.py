import os
import sys
from pathlib import Path

import yaml

# =============================================================================
# MODEL REGISTRY
# Single source of truth for which Hugging Face model each pipeline stage
# uses, keyed by the short name every script's --model flag and every
# activations/<key>/, results/<key>/ folder is named after.
# =============================================================================

MODELS_FILE = Path(__file__).resolve().parent / "models.yml"


def load_models(path=MODELS_FILE):
    """Returns {key: model_id}, in the order models.yml defines them."""
    with open(path, "r", encoding="utf-8") as f:
        models = yaml.safe_load(f)
    if not models:
        raise ValueError(f"No models defined in {path}")
    return models


def resolve_model(key=None, path=MODELS_FILE):
    """Resolves a models.yml key to (key, model_id).

    key=None defaults to the first entry in models.yml.
    """
    models = load_models(path)
    if key is None:
        key = next(iter(models))
    if key not in models:
        available = ", ".join(models)
        raise KeyError(f"Unknown model key '{key}'. Available: {available}")
    return key, models[key]


def model_dirs(key, activations_root="activations", results_root="results"):
    """Per-model subfolders so different models never share cached
    activations/direction/signature or results."""
    return str(Path(activations_root) / key), str(Path(results_root) / key)


# =============================================================================
# HF_HUB_OFFLINE — cache-aware, per-model
#
# Every model-loading script forces HF_HUB_OFFLINE=1 to skip a Hugging Face
# Hub metadata call that intermittently access-violates inside
# socket.getaddrinfo on Windows once a model is already cached locally. But
# forcing it unconditionally also blocks a model's very first download (the
# whole point of offline mode). Since HF_HUB_OFFLINE is read by
# huggingface_hub once, at import time, the decision must be made — and this
# function called — before `import torch` / `from transformers import ...`.
# =============================================================================

def cli_model_arg(argv=None):
    """Lightweight scan for --model VALUE / --model=VALUE in argv, without
    pulling in argparse — used to resolve the model before transformers is
    imported, so configure_hf_offline_mode() can run first."""
    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    return None


def _hf_cache_dir():
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_cached(model_id):
    """True if model_id already has a downloaded snapshot in the local HF cache."""
    repo_dir = _hf_cache_dir() / ("models--" + model_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def configure_hf_offline_mode(model_id):
    """Sets HF_HUB_OFFLINE=1 only when model_id is already cached locally —
    keeps the Windows crash workaround for models we already have, without
    blocking a new model's first-ever download. No-op if HF_HUB_OFFLINE is
    already set explicitly in the environment.
    """
    if "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1" if is_model_cached(model_id) else "0"
