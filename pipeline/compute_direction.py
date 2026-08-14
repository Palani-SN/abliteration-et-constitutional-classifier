import argparse
import hashlib
from pathlib import Path

# load_datasets (pandas/pyarrow) must be imported before torch — importing torch
# first causes a deterministic access-violation crash inside pyarrow on Windows.
from load_datasets import PromptSets

import torch

from models import resolve_model
from utils.visualize import plot_activation_analysis

# =============================================================================
# COMPUTE THE REFUSAL DIRECTION FROM COLLECTED ACTIVATIONS
#
# Mirrors remove-refusals-with-transformers/compute_refusal_dir.py: a single
# mean-diff direction (harmful_mean - harmless_mean, normalized) taken from one
# fixed layer — layer_idx = int(num_layers * mult_factor), no statistical
# ranking or causal search across candidates. mult_factor=0.6 matches their
# script's default and the classic "refusal direction" finding that this
# depth range is usually sufficient on its own.
#
# The per-layer mean-diff/norms used for the HTML visualization are still
# computed across every layer (cheap — no generation involved), the formula
# just picks which single layer's direction actually gets saved for use by
# abliterate.py/verify.py.
#
# "refuse" = harmfull prompts (model should refuse), "accept" = harmless
# prompts (model should comply) — activations come from collect_activations.py,
# which stores them under activations/harmfull_train/ and activations/harmless_train/.
#
# SIGNATURE (early, pre-generation classification signal)
# Separate from the ablation direction above. Takes the last `sig_band` layers
# (nearest the output, excluding the final layer itself) and ranks d_model
# dimensions by Cohen's d — (refuse_mean - accept_mean) / pooled_std, averaged
# across the layer band — NOT raw mean-diff magnitude. Raw magnitude picks up
# "rogue"/outlier dimensions that carry huge activation values on nearly every
# prompt regardless of content (a well-known LLM phenomenon); those dominate a
# flattened cosine similarity without actually discriminating refuse vs accept.
# Cohen's d normalizes by each dimension's own spread, so a dim only gets
# picked if it is reliably different between the two groups relative to its
# noise, not just loud. Keeps the top/bottom `sig_top_n`/`sig_bottom_n` dims
# (most positive and most negative Cohen's d). The saved matrix holds
# refuse_mean (not raw_diff) at that {layers} x {dims} cross-product, plus the
# layer/dim index lists needed to re-slice the same coordinates from a live
# forward pass at inference time, plus a gate_threshold. The threshold is NOT
# the midpoint of the refuse/accept mean scores — that assumes both groups
# have similar spread, which they don't (refuse clusters tightly near 1.0,
# accept is wide with a long upper tail), so the mean-midpoint sits well
# inside accept's tail and creates avoidable false positives. Instead it's the
# accuracy-maximizing cut point found by exact search over the train score
# distribution (see _best_threshold) — an empirically grounded cutoff,
# not a guessed constant or an assumption about the distributions' shape.
# =============================================================================


class DirectionComputer:

    PROMPT_COLUMN = "text"

    def __init__(self, dataset_path="dataset/", activations_dir="activations",
                 observations_dir=None, out_path=None, sig_path=None, mult_factor=0.6, top_n=None,
                 sig_band=6, sig_top_n=3, sig_bottom_n=3):

        self.activations_dir = Path(activations_dir)
        # observations_dir holds the small, human-facing outputs (direction,
        # signature, reports) separately from the bulky raw per-prompt
        # activation cache under activations_dir — so a paper/analysis
        # workflow only needs to copy observations/<model_key>/ off the GPU
        # box, not the much larger activations/<model_key>/ tree.
        self.observations_dir = Path(observations_dir) if observations_dir else self.activations_dir
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = Path(out_path) if out_path else self.observations_dir / "direction.pt"
        self.sig_path = Path(sig_path) if sig_path else self.observations_dir / "signature.pt"
        self.mult_factor = mult_factor
        self.top_n = top_n

        self.sig_band = sig_band
        self.sig_top_n = sig_top_n
        self.sig_bottom_n = sig_bottom_n

        self.prompt_sets = PromptSets(dataset_path)

    @staticmethod
    def _prompt_id(prompt):
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def _load_prompts(self, category, split):
        df = self.prompt_sets.get(category, split)
        prompts = df[self.PROMPT_COLUMN] if self.top_n is None else df[self.PROMPT_COLUMN].head(self.top_n)
        return prompts.tolist()

    def _load_activation_set(self, category, split):
        # Look up each prompt's activation by content hash, in prompt order, so a
        # changed prompt set can never be silently paired with stale activations.
        name = f"{category}_{split}"
        set_dir = self.activations_dir / name
        acts = []
        for prompt in self._load_prompts(category, split):
            path = set_dir / f"{self._prompt_id(prompt)}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"No cached activation for a prompt in '{name}' (expected {path}). "
                    f"Prompts have likely changed since collect_activations.py was last run "
                    f"— re-run it before computing directions."
                )
            acts.append(torch.load(path).float())
        return torch.stack(acts, dim=0)  # [num_prompts, num_layers+1, d_model]

    @staticmethod
    def _compute_mean_diff(refuse_acts, accept_acts):
        refuse_mean = refuse_acts.mean(dim=0)  # [num_layers+1, d_model]
        accept_mean = accept_acts.mean(dim=0)  # [num_layers+1, d_model]

        raw_diff = refuse_mean - accept_mean  # [num_layers+1, d_model]
        norms = raw_diff.norm(dim=-1)  # [num_layers+1]
        directions = raw_diff / norms.unsqueeze(-1)  # unit vectors [num_layers+1, d_model]

        return refuse_mean, accept_mean, raw_diff, norms, directions

    def _save_direction(self, direction, layer_idx, norm):
        torch.save({
            "direction":   direction,     # [D]  unit vector, the one direction to ablate
            "layer_idx":   layer_idx,     # activation position it was taken from
            "mult_factor": self.mult_factor,
            "norm":        norm,
        }, self.out_path)
        print(f"\nSaved -> {self.out_path}")
        print(f"  direction: {tuple(direction.shape)}  (layer {layer_idx}, mult_factor={self.mult_factor}, norm={norm:.4f})")

    @staticmethod
    def _flatten_cosine_scores(acts, layers, dims, ref_unit):
        # acts: [N, num_layers+1, D] -> per-sample cosine similarity of the
        # flattened {layers} x {dims} block against the flattened reference.
        block = acts[:, layers][:, :, dims]  # [N, len(layers), len(dims)]
        flat = block.flatten(start_dim=1)  # [N, len(layers)*len(dims)]
        flat_unit = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return flat_unit @ ref_unit  # [N]

    @staticmethod
    def _best_threshold(refuse_scores, accept_scores):
        # Midpoint-of-means is wrong here: refuse/accept score distributions
        # have very different variances (refuse is a tight cluster near 1.0,
        # accept is wide with a long upper tail), so the accuracy-maximizing
        # cut point is not their mean midpoint. Exact search: every midpoint
        # between consecutive sorted combined scores is a candidate threshold;
        # pick whichever maximizes train accuracy. O(N^2) on N~500 is trivial.
        combined = torch.cat([refuse_scores, accept_scores])
        sorted_unique = torch.unique(combined)
        mids = (sorted_unique[:-1] + sorted_unique[1:]) / 2
        candidates = torch.cat([sorted_unique[:1] - 1e-6, mids, sorted_unique[-1:] + 1e-6])

        n_total = refuse_scores.numel() + accept_scores.numel()
        best_t, best_acc = candidates[0].item(), -1.0
        for t in candidates.tolist():
            tp = (refuse_scores >= t).sum().item()
            tn = (accept_scores < t).sum().item()
            acc = (tp + tn) / n_total
            if acc > best_acc:
                best_acc = acc
                best_t = t
        return best_t

    def _compute_signature(self, refuse_acts, accept_acts, refuse_mean, num_layers):
        layer_start = num_layers - self.sig_band
        layers = list(range(layer_start, num_layers))
        layer_t = torch.tensor(layers)

        refuse_band = refuse_acts[:, layer_t]  # [N, sig_band, D]
        accept_band = accept_acts[:, layer_t]

        diff = refuse_band.mean(dim=0) - accept_band.mean(dim=0)  # [sig_band, D]
        pooled_std = ((refuse_band.std(dim=0) ** 2 + accept_band.std(dim=0) ** 2) / 2).sqrt().clamp(min=1e-8)
        cohend = (diff / pooled_std).mean(dim=0)  # [D], averaged across the layer band

        order = torch.argsort(cohend, descending=True)  # rank by real separation, not raw magnitude
        top_idx = order[:self.sig_top_n].tolist()
        bottom_idx = order[-self.sig_bottom_n:].tolist()
        dims = sorted(top_idx + bottom_idx)  # re-sorted by index, not by rank
        dim_t = torch.tensor(dims)

        matrix = refuse_mean[layer_t][:, dim_t]  # [sig_band, top_n+bottom_n], from refuse_mean

        ref_flat = matrix.flatten()
        ref_unit = ref_flat / ref_flat.norm().clamp(min=1e-8)
        refuse_scores = self._flatten_cosine_scores(refuse_acts, layer_t, dim_t, ref_unit)
        accept_scores = self._flatten_cosine_scores(accept_acts, layer_t, dim_t, ref_unit)
        gate_threshold = self._best_threshold(refuse_scores, accept_scores)

        return layers, dims, matrix, gate_threshold

    def _save_signature(self, layers, dims, matrix, gate_threshold):
        torch.save({
            "layers":         layers,          # ascending, matches matrix rows
            "dims":           dims,            # ascending, matches matrix cols
            "matrix":         matrix,          # [len(layers), len(dims)] block from refuse_mean
            "top_n":          self.sig_top_n,
            "bottom_n":       self.sig_bottom_n,
            "gate_threshold": gate_threshold,  # midpoint of refuse/accept train flatten-cosine scores
        }, self.sig_path)
        print(f"\nSaved -> {self.sig_path}")
        print(f"  signature: {tuple(matrix.shape)}  (layers={layers}, dims={dims}, gate_threshold={gate_threshold:.4f})")

    def run(self):
        refuse_acts = self._load_activation_set("harmfull", "train")
        accept_acts = self._load_activation_set("harmless", "train")

        print(f"harmfull_train activations: {tuple(refuse_acts.shape)}")
        print(f"harmless_train activations: {tuple(accept_acts.shape)}")

        num_layers = refuse_acts.shape[1] - 1  # exclude the embedding position

        refuse_mean, accept_mean, raw_diff, norms, directions = self._compute_mean_diff(refuse_acts, accept_acts)

        print("\nlayer | diff-of-means norm")
        print("------|--------------------")
        for layer in range(num_layers + 1):
            print(f"{layer:5d} | {norms[layer]:.4f}")

        layer_idx = int(num_layers * self.mult_factor)
        direction = directions[layer_idx]
        print(f"\nlayer_idx = int({num_layers} * {self.mult_factor}) = {layer_idx}")

        self._save_direction(direction, layer_idx, norms[layer_idx].item())

        sig_layers, sig_dims, sig_matrix, gate_threshold = self._compute_signature(
            refuse_acts=refuse_acts, accept_acts=accept_acts, refuse_mean=refuse_mean, num_layers=num_layers,
        )
        self._save_signature(sig_layers, sig_dims, sig_matrix, gate_threshold)

        plot_activation_analysis(
            refuse_mean, accept_mean, raw_diff, norms,
            out_path=self.observations_dir / "activation_analysis.html",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key from models.yml (default: first entry)")
    args = parser.parse_args()

    model_key, model_id = resolve_model(args.model)
    print(f"Using model '{model_key}' -> {model_id}")

    computer = DirectionComputer(
        dataset_path="dataset/",
        activations_dir=f"activations/{model_key}",
        observations_dir=f"observations/{model_key}",
        mult_factor=0.6,
        top_n=None,
    )
    computer.run()
