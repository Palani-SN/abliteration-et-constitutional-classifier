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

MODELS_FILE = Path(__file__).resolve().parent.parent / "models.yml"  # kept at repo root, one level up from pipeline/


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


# =============================================================================
# device_map="auto" memory budget
#
# Left to its own defaults, device_map="auto" plans device placement against
# each device's raw capacity, packing a model onto the GPU right up to the
# edge if it technically fits. That leaves no slack for transient load-time
# allocations (e.g. tie_weights() comparing tied parameters) or generation-
# time activations, and a model that "fits" on paper OOMs in practice. See
# recurrentgemma_9b: ~18GB of bf16 weights on a 24GB GPU reporting only
# ~22GB usable, packed almost entirely onto the GPU, OOM'd on a 1GB
# allocation with 320MB free. Reserving a slice of both devices up front
# makes accelerate offload proactively instead.
# =============================================================================

def safe_max_memory(gpu_headroom_gib=2.0, cpu_fraction=0.85):
    """max_memory dict for from_pretrained(..., device_map="auto"), reserving
    headroom below raw capacity on both devices. Call only after `import
    torch` — deliberately not imported at module level here, since this
    module is imported before torch/transformers elsewhere in the pipeline
    (see configure_hf_offline_mode above) and must stay import-light.

    Returns None (accelerate's own default planning) if no CUDA device is
    available, e.g. on a local CPU-only dev machine.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    import psutil

    total_gpu_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_budget = max(total_gpu_gib - gpu_headroom_gib, 1.0)

    available_cpu_gib = psutil.virtual_memory().available / (1024 ** 3)
    cpu_budget = max(available_cpu_gib * cpu_fraction, 1.0)

    return {0: f"{gpu_budget:.1f}GiB", "cpu": f"{cpu_budget:.1f}GiB"}
