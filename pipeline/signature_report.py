import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats
import plotly.graph_objects as go
import plotly.io as pio

from models import resolve_model

# =============================================================================
# SIGNATURE DIMENSION ANALYSIS (adapted from analyze_signature.py's reference
# methodology for THIS project's data: 2048-dim, layers/dims read from
# activations/<model_key>/signature.pt instead of hardcoded, and prompt
# activations globbed directly from activations/<model_key>/<category>_<split>/
# instead of a prompts/*.json manifest. Model selected dynamically via
# --model <key> from models.yml.)
#
# Checks whether the Cohen's-d-selected {layers} x {dims} from
# compute_direction.py forms a statistically reliable fingerprint for refusal —
# the "first step of quick verification" before considering a larger dim set.
#
# Per-layer analysis: cosine similarity distributions (refuse-refuse vs
# refuse-accept) with Cohen's d and Mann-Whitney U test, swept across every
# layer (not just the selected band) so the band can be judged in context.
#
# Combined analysis: averages the per-layer cosine similarity scores across
# the signature's layer band (score averaging, not vector concatenation).
#
# OOD classification: builds a mean refuse signature from harmfull_train, then
# classifies each harmfull_test/harmless_test ("OOD") prompt by cosine
# similarity threshold (midpoint of train-set refuse/accept mean scores) and
# reports accuracy, precision, recall, F1.
#
# Outputs:
#   observations/<model_key>/signature_stats.json    per-layer + combined + OOD statistics
#   observations/<model_key>/signature_report.html   five-panel interactive report
# =============================================================================

ACTIVATIONS_DIR = "activations"    # rebound in __main__ to activations/<model_key>
OBSERVATIONS_DIR = "observations"  # rebound in __main__ to observations/<model_key>


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_activation_set(name):
    """Returns [N_prompts, N_layers, d_model] float32 tensor from cached .pt files."""
    set_dir = Path(ACTIVATIONS_DIR) / name
    acts = [torch.load(f).float() for f in sorted(set_dir.glob("*.pt"))]
    if not acts:
        raise FileNotFoundError(f"No cached activations found under {set_dir}")
    return torch.stack(acts, dim=0)  # [N, L, D]


def cosine_sims(a, b=None):
    """
    a: [M, K]  b: [N, K]  (b=None -> a vs a, diagonal excluded)
    Returns all pairwise cosine similarities as a flat array.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    if b is None:
        mat = a_norm @ a_norm.T
        i, j = np.triu_indices(len(a), k=1)
        return mat[i, j]
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm @ b_norm.T).flatten()


def cosine_sim_to_ref(queries, reference):
    """Cosine similarity of each row in queries against a single reference vector."""
    ref_norm = reference / (np.linalg.norm(reference) + 1e-8)
    q_norm   = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    return q_norm @ ref_norm  # [N]


def cohens_d(x, y):
    pooled_std = (x.std() + y.std()) / 2 + 1e-9
    return float((x.mean() - y.mean()) / pooled_std)


def best_threshold(refuse_scores, accept_scores):
    """
    Exact search for the accuracy-maximizing cut point: every midpoint between
    consecutive sorted combined scores is a candidate threshold. Midpoint-of-
    means is not used here because refuse/accept score distributions have very
    different variances (refuse is tight near 1.0, accept is wide with a long
    upper tail) — the mean midpoint sits inside accept's tail and creates
    avoidable false positives. O(N^2) on N~500 train prompts is trivial.
    """
    combined = np.concatenate([refuse_scores, accept_scores])
    sorted_unique = np.unique(combined)
    mids = (sorted_unique[:-1] + sorted_unique[1:]) / 2
    candidates = np.concatenate([sorted_unique[:1] - 1e-6, mids, sorted_unique[-1:] + 1e-6])

    n_total = len(refuse_scores) + len(accept_scores)
    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        tp = (refuse_scores >= t).sum()
        tn = (accept_scores < t).sum()
        acc = (tp + tn) / n_total
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t)


def _fmt_p(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


# ── Per-layer analysis ────────────────────────────────────────────────────────

def analyse(refuse_acts, accept_acts, sig_dims):
    """Returns per-layer stat dicts for every layer (0 = embedding)."""
    N_layers = refuse_acts.shape[1]
    results = []
    for layer in range(N_layers):
        ref_sig = refuse_acts[:, layer, :][:, sig_dims].numpy()
        acc_sig = accept_acts[:, layer, :][:, sig_dims].numpy()

        rr = cosine_sims(ref_sig)
        ra = cosine_sims(ref_sig, acc_sig)

        _, p = stats.mannwhitneyu(rr, ra, alternative="greater")
        d    = cohens_d(rr, ra)

        results.append({
            "layer":     layer,
            "rr_mean":   float(rr.mean()),
            "rr_std":    float(rr.std()),
            "ra_mean":   float(ra.mean()),
            "ra_std":    float(ra.std()),
            "cohens_d":  d,
            "mw_pvalue": float(p),
            "rr_sims":   rr.tolist(),
            "ra_sims":   ra.tolist(),
        })
    return results


# ── Combined multi-layer analysis (score averaging) ───────────────────────────

def analyse_combined(refuse_acts, accept_acts, sig_dims, sig_layers):
    """
    Score averaging: compute cosine similarity independently at each layer in
    sig_layers, then average those scores per pair. Each layer casts one
    independent vote — the average preserves every layer's discriminative
    signal rather than diluting it in a joint vector space (which is what
    concatenation does and why it underperforms the best single layer).
    """
    rr_per_layer = []
    ra_per_layer = []

    for layer in sig_layers:
        ref_sig = refuse_acts[:, layer, :][:, sig_dims].numpy()
        acc_sig = accept_acts[:, layer, :][:, sig_dims].numpy()
        rr_per_layer.append(cosine_sims(ref_sig))
        ra_per_layer.append(cosine_sims(ref_sig, acc_sig))

    rr = np.stack(rr_per_layer, axis=0).mean(axis=0)
    ra = np.stack(ra_per_layer, axis=0).mean(axis=0)

    _, p = stats.mannwhitneyu(rr, ra, alternative="greater")
    d    = cohens_d(rr, ra)

    return {
        "layers":    sig_layers,
        "dims":      sig_dims,
        "rr_mean":   float(rr.mean()),
        "rr_std":    float(rr.std()),
        "ra_mean":   float(ra.mean()),
        "ra_std":    float(ra.std()),
        "cohens_d":  d,
        "mw_pvalue": float(p),
        "rr_sims":   rr.tolist(),
        "ra_sims":   ra.tolist(),
    }


# ── OOD classification ────────────────────────────────────────────────────────

def _classification_metrics(refuse_scores, accept_scores, threshold):
    refuse_preds = refuse_scores >= threshold
    accept_preds = accept_scores >= threshold

    tp = int(refuse_preds.sum())
    fn = int((~refuse_preds).sum())
    fp = int(accept_preds.sum())
    tn = int((~accept_preds).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + tn + fp + fn)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":      round(accuracy,  4),
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "f1":            round(f1,        4),
        "refuse_scores": refuse_scores.tolist(),
        "accept_scores": accept_scores.tolist(),
    }


def classify_ood(refuse_train_acts, accept_train_acts, refuse_ood_acts, accept_ood_acts,
                 sig_dims, sig_layers):
    """
    Builds the mean refuse signature from training data (at each layer in
    sig_layers, take the mean sig_dims vector across all refuse_train prompts).
    Scores every prompt (train AND OOD) by averaging its per-layer cosine
    similarity to that reference. Threshold = accuracy-maximizing cut point
    over the TRAIN-set scores only (see best_threshold) — never derived from
    OOD data, since a deployed gate has no access to OOD labels at calibration
    time. Both splits are then classified with that same train-derived
    threshold, so the two can be visually compared for coherence (does OOD
    look like more of the same distribution, or has it drifted?).
    """
    ref_means = []
    for layer in sig_layers:
        sig = refuse_train_acts[:, layer, :][:, sig_dims].numpy()
        ref_means.append(sig.mean(axis=0))  # [K]

    def score_prompts(acts):
        per_layer = []
        for idx, layer in enumerate(sig_layers):
            sig  = acts[:, layer, :][:, sig_dims].numpy()  # [N, K]
            ref  = ref_means[idx]
            sims = cosine_sim_to_ref(sig, ref)             # [N]
            per_layer.append(sims)
        return np.stack(per_layer, axis=0).mean(axis=0)    # [N]

    refuse_train_scores = score_prompts(refuse_train_acts)
    accept_train_scores = score_prompts(accept_train_acts)
    refuse_ood_scores   = score_prompts(refuse_ood_acts)
    accept_ood_scores   = score_prompts(accept_ood_acts)

    threshold = best_threshold(refuse_train_scores, accept_train_scores)

    return {
        "threshold": threshold,
        "train": _classification_metrics(refuse_train_scores, accept_train_scores, threshold),
        "ood":   _classification_metrics(refuse_ood_scores, accept_ood_scores, threshold),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def build_html(results, combined, ood, sig_dims, sig_layers, baseline_combined=None, baseline_ood=None):
    layers = [r["layer"] for r in results]
    best   = max(results, key=lambda r: r["cohens_d"])

    # ── Figure 1: mean cosine sim per layer ──────────────────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=layers, y=[r["rr_mean"] for r in results],
        error_y=dict(type="data", array=[r["rr_std"] for r in results], visible=True),
        mode="lines+markers", name="refuse-refuse",
        line=dict(color="royalblue", width=2), marker=dict(size=6),
        hovertemplate="layer %{x}<br>mean=%{y:.3f}<extra>refuse-refuse</extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=layers, y=[r["ra_mean"] for r in results],
        error_y=dict(type="data", array=[r["ra_std"] for r in results], visible=True),
        mode="lines+markers", name="refuse-accept",
        line=dict(color="tomato", width=2), marker=dict(size=6),
        hovertemplate="layer %{x}<br>mean=%{y:.3f}<extra>refuse-accept</extra>",
    ))
    fig1.add_vrect(
        x0=min(sig_layers) - 0.5, x1=max(sig_layers) + 0.5,
        fillcolor="gold", opacity=0.12, line_width=0,
        annotation_text=f"layers {min(sig_layers)}-{max(sig_layers)}",
        annotation_position="top left",
    )
    fig1.update_layout(
        title_text=f"Mean cosine similarity per layer — dims {sig_dims}",
        xaxis_title="Layer", yaxis_title="Cosine similarity",
        height=420, hovermode="x unified", legend=dict(x=0.01, y=0.99),
    )

    # ── Figure 2: Cohen's d per layer ────────────────────────────────────────
    ds = [r["cohens_d"] for r in results]
    fig2 = go.Figure(go.Bar(
        x=layers, y=ds,
        marker_color=["royalblue" if d >= 0 else "tomato" for d in ds],
        hovertemplate="layer %{x}<br>Cohen's d=%{y:.3f}<extra></extra>",
    ))
    fig2.add_vrect(
        x0=min(sig_layers) - 0.5, x1=max(sig_layers) + 0.5,
        fillcolor="gold", opacity=0.12, line_width=0,
    )
    fig2.update_layout(
        title_text=f"Cohen's d per layer — dims {sig_dims}",
        xaxis_title="Layer", yaxis_title="Cohen's d", height=350,
    )

    # ── Figure 3: per-layer histogram with dropdown ───────────────────────────
    fig3 = go.Figure()
    N_layers = len(results)
    for i, r in enumerate(results):
        visible = i == 0
        for sims, label, color in [
            (r["rr_sims"], "refuse-refuse", "royalblue"),
            (r["ra_sims"], "refuse-accept", "tomato"),
        ]:
            fig3.add_trace(go.Histogram(
                x=sims, name=label, marker_color=color, opacity=0.65,
                nbinsx=40, visible=visible, showlegend=(i == 0),
                hovertemplate=f"{label}: %{{x:.3f}}<extra></extra>",
            ))

    buttons = []
    for i, r in enumerate(results):
        vis = [False] * (N_layers * 2)
        vis[i * 2] = True
        vis[i * 2 + 1] = True
        buttons.append(dict(
            label=f"Layer {r['layer']}",
            method="update",
            args=[
                {"visible": vis},
                {"title": (
                    f"Layer {r['layer']} — cosine sim distribution  |  "
                    f"Cohen's d={r['cohens_d']:.2f}  |  MW p={_fmt_p(r['mw_pvalue'])}  |  "
                    f"rr={r['rr_mean']:.3f}±{r['rr_std']:.3f}  "
                    f"ra={r['ra_mean']:.3f}±{r['ra_std']:.3f}"
                )},
            ],
        ))

    r0 = results[0]
    fig3.update_layout(
        title_text=(
            f"Layer 0 — cosine sim distribution  |  "
            f"Cohen's d={r0['cohens_d']:.2f}  |  MW p={_fmt_p(r0['mw_pvalue'])}  |  "
            f"rr={r0['rr_mean']:.3f}±{r0['rr_std']:.3f}  "
            f"ra={r0['ra_mean']:.3f}±{r0['ra_std']:.3f}"
        ),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0.0, y=1.18, showactive=True)],
        barmode="overlay", xaxis_title="Cosine similarity", yaxis_title="Count",
        height=420, legend=dict(x=0.01, y=0.99),
    )

    # ── Figure 4: combined (score avg) vs best single layer ──────────────────
    fig4 = go.Figure()
    for sims, label, color, dash in [
        (best["rr_sims"],    f"refuse-refuse (layer {best['layer']} only)", "royalblue", "dot"),
        (best["ra_sims"],    f"refuse-accept (layer {best['layer']} only)", "tomato",    "dot"),
        (combined["rr_sims"], f"refuse-refuse (score avg layers {min(sig_layers)}-{max(sig_layers)})", "royalblue", "solid"),
        (combined["ra_sims"], f"refuse-accept (score avg layers {min(sig_layers)}-{max(sig_layers)})", "tomato",    "solid"),
    ]:
        fig4.add_trace(go.Histogram(
            x=sims, name=label, opacity=0.55, nbinsx=40,
            marker=dict(color=color),
            hovertemplate=f"{label}: %{{x:.3f}}<extra></extra>",
        ))
    fig4.update_layout(
        title_text=(
            f"Score-avg combined vs best single layer ({best['layer']})  |  "
            f"Combined: d={combined['cohens_d']:.2f}, MW p={_fmt_p(combined['mw_pvalue'])}, "
            f"rr={combined['rr_mean']:.3f}, ra={combined['ra_mean']:.3f}  |  "
            f"Best single: d={best['cohens_d']:.2f}"
        ),
        barmode="overlay",
        xaxis_title="Cosine similarity", yaxis_title="Count",
        height=440, legend=dict(x=0.01, y=0.99),
    )

    # ── Figure 5: classification scores — TRAIN split ────────────────────────
    # Threshold is fit on train only; this panel shows the (somewhat optimistic,
    # since ref_means come from this same data) fit, for comparison against OOD.
    train = ood["train"]
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(
        x=train["refuse_scores"], name="refuse_train (should be HIGH)",
        marker_color="tomato", opacity=0.65, nbinsx=30,
        hovertemplate="score=%{x:.3f}<extra>refuse_train</extra>",
    ))
    fig5.add_trace(go.Histogram(
        x=train["accept_scores"], name="accept_train (should be LOW)",
        marker_color="royalblue", opacity=0.65, nbinsx=30,
        hovertemplate="score=%{x:.3f}<extra>accept_train</extra>",
    ))
    fig5.add_vline(
        x=ood["threshold"], line_dash="dash", line_color="black",
        annotation_text=f"threshold={ood['threshold']:.3f} (fit on train)",
        annotation_position="top right",
    )
    fig5.update_layout(
        title_text=(
            f"TRAIN Classification — score-avg over layers {min(sig_layers)}-{max(sig_layers)}  |  "
            f"Accuracy={train['accuracy']:.1%}  Precision={train['precision']:.1%}  "
            f"Recall={train['recall']:.1%}  F1={train['f1']:.3f}  |  "
            f"TP={train['tp']} TN={train['tn']} FP={train['fp']} FN={train['fn']}"
        ),
        barmode="overlay",
        xaxis_title="Score (avg cosine sim to mean refuse signature)",
        yaxis_title="Count",
        height=440, legend=dict(x=0.01, y=0.99),
    )

    # ── Figure 6: classification scores — OOD (test) split ───────────────────
    # Same train-derived threshold applied to held-out data — the real test of
    # whether the signature generalises, and whether it looks coherent with Fig 5.
    ood_split = ood["ood"]
    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(
        x=ood_split["refuse_scores"], name="refuse_ood (should be HIGH)",
        marker_color="tomato", opacity=0.65, nbinsx=30,
        hovertemplate="score=%{x:.3f}<extra>refuse_ood</extra>",
    ))
    fig6.add_trace(go.Histogram(
        x=ood_split["accept_scores"], name="accept_ood (should be LOW)",
        marker_color="royalblue", opacity=0.65, nbinsx=30,
        hovertemplate="score=%{x:.3f}<extra>accept_ood</extra>",
    ))
    fig6.add_vline(
        x=ood["threshold"], line_dash="dash", line_color="black",
        annotation_text=f"threshold={ood['threshold']:.3f} (fit on train)",
        annotation_position="top right",
    )
    fig6.update_layout(
        title_text=(
            f"OOD (test) Classification — score-avg over layers {min(sig_layers)}-{max(sig_layers)}  |  "
            f"Accuracy={ood_split['accuracy']:.1%}  Precision={ood_split['precision']:.1%}  "
            f"Recall={ood_split['recall']:.1%}  F1={ood_split['f1']:.3f}  |  "
            f"TP={ood_split['tp']} TN={ood_split['tn']} FP={ood_split['fp']} FN={ood_split['fn']}"
        ),
        barmode="overlay",
        xaxis_title="Score (avg cosine sim to mean refuse signature)",
        yaxis_title="Count",
        height=440, legend=dict(x=0.01, y=0.99),
    )

    figs = [fig1, fig2, fig3, fig4, fig5, fig6]

    # ── Figure 7: signature (few dims/layers) vs all-dims/all-layers baseline ─
    # Same OOD split, same style as Fig 6, but overlaying the full-hidden-state
    # baseline (every dim, every layer) scored with its own train-fit threshold.
    # If the signature is doing real work, its distributions should separate
    # MORE cleanly than the baseline's, despite using a tiny fraction of the
    # information — averaging cosine similarity over irrelevant/noisy dims
    # dilutes the signal rather than adding to it.
    if baseline_combined is not None and baseline_ood is not None:
        b_ood = baseline_ood["ood"]
        fig7 = go.Figure()
        fig7.add_trace(go.Histogram(
            x=ood_split["refuse_scores"], name="refuse_ood (signature)",
            marker_color="tomato", opacity=0.55, nbinsx=30,
            hovertemplate="score=%{x:.3f}<extra>refuse_ood (signature)</extra>",
        ))
        fig7.add_trace(go.Histogram(
            x=ood_split["accept_scores"], name="accept_ood (signature)",
            marker_color="royalblue", opacity=0.55, nbinsx=30,
            hovertemplate="score=%{x:.3f}<extra>accept_ood (signature)</extra>",
        ))
        fig7.add_trace(go.Histogram(
            x=b_ood["refuse_scores"], name="refuse_ood (all-dims baseline)",
            marker_color="darkred", opacity=0.55, nbinsx=30,
            hovertemplate="score=%{x:.3f}<extra>refuse_ood (baseline)</extra>",
        ))
        fig7.add_trace(go.Histogram(
            x=b_ood["accept_scores"], name="accept_ood (all-dims baseline)",
            marker_color="darkslateblue", opacity=0.55, nbinsx=30,
            hovertemplate="score=%{x:.3f}<extra>accept_ood (baseline)</extra>",
        ))
        fig7.add_vline(
            x=ood["threshold"], line_dash="dash", line_color="black",
            annotation_text=f"signature threshold={ood['threshold']:.3f}",
            annotation_position="top right",
        )
        fig7.add_vline(
            x=baseline_ood["threshold"], line_dash="dot", line_color="gray",
            annotation_text=f"baseline threshold={baseline_ood['threshold']:.3f}",
            annotation_position="top left",
        )
        fig7.update_layout(
            title_text=(
                f"Signature ({len(sig_layers)} layers x {len(sig_dims)} dims) vs "
                f"all-dims/all-layers baseline — OOD split  |  "
                f"Signature: d={combined['cohens_d']:.2f}, Acc={ood_split['accuracy']:.1%}, F1={ood_split['f1']:.3f}  |  "
                f"Baseline: d={baseline_combined['cohens_d']:.2f}, Acc={b_ood['accuracy']:.1%}, F1={b_ood['f1']:.3f}"
            ),
            barmode="overlay",
            xaxis_title="Score (avg cosine sim to mean refuse signature)",
            yaxis_title="Count",
            height=460, legend=dict(x=0.01, y=0.99),
        )
        figs.append(fig7)

    divider = "\n<hr style='margin:40px 0;border:none;border-top:1px solid #ccc'>\n"
    body = divider.join([
        pio.to_html(fig, full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
        for i, fig in enumerate(figs)
    ])
    return (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>Signature Report — dims {sig_dims}</title>"
        "<style>body{font-family:sans-serif;padding:20px}</style>"
        "</head>\n<body>\n"
        f"<h2>Signature Report — dims {sig_dims}  |  combined layers {min(sig_layers)}-{max(sig_layers)}</h2>\n"
        + body
        + "\n</body></html>"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key from models.yml (default: first entry)")
    args = parser.parse_args()

    model_key, model_id = resolve_model(args.model)
    print(f"Using model '{model_key}' -> {model_id}")

    ACTIVATIONS_DIR = os.path.join("activations", model_key)
    OBSERVATIONS_DIR = os.path.join("observations", model_key)
    os.makedirs(OBSERVATIONS_DIR, exist_ok=True)
    SIGNATURE_PATH  = os.path.join(OBSERVATIONS_DIR, "signature.pt")
    STATS_OUT       = os.path.join(OBSERVATIONS_DIR, "signature_stats.json")
    HTML_OUT        = os.path.join(OBSERVATIONS_DIR, "signature_report.html")

    signature = torch.load(SIGNATURE_PATH)
    sig_layers = signature["layers"]
    sig_dims   = signature["dims"]
    print(f"Signature layers: {sig_layers}")
    print(f"Signature dims:   {sig_dims}")

    print("\nLoading cached activations ...")
    refuse_train_acts = load_activation_set("harmfull_train")
    accept_train_acts = load_activation_set("harmless_train")
    refuse_ood_acts   = load_activation_set("harmfull_test")
    accept_ood_acts   = load_activation_set("harmless_test")
    print(f"  harmfull_train: {tuple(refuse_train_acts.shape)}")
    print(f"  harmless_train: {tuple(accept_train_acts.shape)}")
    print(f"  harmfull_test:  {tuple(refuse_ood_acts.shape)}")
    print(f"  harmless_test:  {tuple(accept_ood_acts.shape)}")

    print("\nPer-layer analysis ...")
    results = analyse(refuse_train_acts, accept_train_acts, sig_dims)

    print(f"\n{'layer':>5}  {'rr_mean':>8}  {'ra_mean':>8}  {'cohen_d':>8}  {'mw_p':>10}")
    print("-" * 50)
    for r in results:
        marker = "  <-- combined window" if r["layer"] in sig_layers else ""
        print(f"{r['layer']:>5}  {r['rr_mean']:>8.3f}  {r['ra_mean']:>8.3f}  "
              f"{r['cohens_d']:>8.3f}  {_fmt_p(r['mw_pvalue']):>10}{marker}")

    best = max(results, key=lambda r: r["cohens_d"])
    print(f"\nBest single layer: {best['layer']}  "
          f"(d={best['cohens_d']:.3f}, rr={best['rr_mean']:.3f}, ra={best['ra_mean']:.3f})")

    print(f"\nCombined analysis — score averaging over layers {sig_layers} ...")
    combined = analyse_combined(refuse_train_acts, accept_train_acts, sig_dims, sig_layers)
    gain = combined["cohens_d"] - best["cohens_d"]
    print(f"  Cohen's d : {combined['cohens_d']:.3f}  (best single: {best['cohens_d']:.3f}, gain: {gain:+.3f})")
    print(f"  rr_mean   : {combined['rr_mean']:.3f} ± {combined['rr_std']:.3f}")
    print(f"  ra_mean   : {combined['ra_mean']:.3f} ± {combined['ra_std']:.3f}")
    print(f"  MW p-value: {_fmt_p(combined['mw_pvalue'])}")

    print(f"\nClassification (layers {sig_layers}, threshold fit on train only) ...")
    ood = classify_ood(refuse_train_acts, accept_train_acts, refuse_ood_acts, accept_ood_acts, sig_dims, sig_layers)
    print(f"  Threshold : {ood['threshold']:.3f}  (from train refuse/accept mean scores)")
    for split_name in ("train", "ood"):
        m = ood[split_name]
        print(f"\n  [{split_name}]")
        print(f"    Accuracy  : {m['accuracy']:.1%}  (TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})")
        print(f"    Precision : {m['precision']:.1%}")
        print(f"    Recall    : {m['recall']:.1%}")
        print(f"    F1        : {m['f1']:.3f}")

    # ── All-dims/all-layers baseline (does dimensionality reduction help?) ────
    N_layers = refuse_train_acts.shape[1]
    D        = refuse_train_acts.shape[2]
    all_dims   = list(range(D))
    all_layers = list(range(N_layers))
    print(f"\nBaseline — ALL {N_layers} layers x ALL {D} dims (no reduction) ...")
    baseline_combined = analyse_combined(refuse_train_acts, accept_train_acts, all_dims, all_layers)
    baseline_ood = classify_ood(refuse_train_acts, accept_train_acts, refuse_ood_acts, accept_ood_acts, all_dims, all_layers)
    print(f"  Cohen's d : {baseline_combined['cohens_d']:.3f}  (signature: {combined['cohens_d']:.3f})")
    print(f"  Threshold : {baseline_ood['threshold']:.3f}")
    for split_name in ("train", "ood"):
        m = baseline_ood[split_name]
        print(f"\n  [{split_name}]")
        print(f"    Accuracy  : {m['accuracy']:.1%}  (TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']})")
        print(f"    Precision : {m['precision']:.1%}")
        print(f"    Recall    : {m['recall']:.1%}")
        print(f"    F1        : {m['f1']:.3f}")

    stats_out = [{k: v for k, v in r.items() if k not in ("rr_sims", "ra_sims")} for r in results]
    combined_out = {k: v for k, v in combined.items() if k not in ("rr_sims", "ra_sims")}
    ood_out = {
        "threshold": ood["threshold"],
        "train": {k: v for k, v in ood["train"].items() if k not in ("refuse_scores", "accept_scores")},
        "ood":   {k: v for k, v in ood["ood"].items() if k not in ("refuse_scores", "accept_scores")},
    }
    baseline_combined_out = {k: v for k, v in baseline_combined.items() if k not in ("rr_sims", "ra_sims")}
    baseline_ood_out = {
        "threshold": baseline_ood["threshold"],
        "train": {k: v for k, v in baseline_ood["train"].items() if k not in ("refuse_scores", "accept_scores")},
        "ood":   {k: v for k, v in baseline_ood["ood"].items() if k not in ("refuse_scores", "accept_scores")},
    }
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "signature_dims":   sig_dims,
            "signature_layers": sig_layers,
            "layers":           stats_out,
            "combined":         combined_out,
            "classification":   ood_out,
            "baseline_all_dims_all_layers": {
                "combined":       baseline_combined_out,
                "classification": baseline_ood_out,
            },
        }, f, indent=2)
    print(f"\nSaved stats -> {STATS_OUT}")

    html = build_html(
        results, combined, ood, sig_dims=sig_dims, sig_layers=sig_layers,
        baseline_combined=baseline_combined, baseline_ood=baseline_ood,
    )
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved report -> {HTML_OUT}")
