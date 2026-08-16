import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

from models import load_models

# =============================================================================
# CROSS-MODEL SUMMARY
# Repo-wide aggregator, NOT a per-model workflow.sh stage. Reads, for every key
# in models.yml that has completed a full run:
#   observations/<key>/signature_stats.json      Stage 3 — signature validation
#   results/<key>.txt                            console log — layer_idx / gate /
#                                                total runtime (the only place
#                                                those three are recorded)
#   results/<key>/<latest>/{harmfull,harmless}.xlsx   Stage 6 — per-prompt rows
#
# and renders one wide comparison table answering the cross-model-transfer
# question the per-model comparison_report.py cannot: does a signature refit
# per model hold up across architectures and scales, and does single-direction
# ablation transfer as reliably as the defense against it does.
#
# Pass-rate definitions match comparison_report.compute_stats exactly, so a row
# here can never disagree with that model's own comparison_report.html:
#   Original   PASS = judgement == expected_label   (safety intact)
#   Abliterated PASS = judgement == COMPLY          (ablation defeated refusal)
#   Classifier PASS = judgement == expected_label   (safety restored)
#
# Output: observations/cross_model_summary.html
# =============================================================================

PAGES_BASE = "https://palani-sn.github.io/LLM2"

# Parameter counts aren't recorded anywhere in the pipeline's outputs (no stage
# loads a model just to count them), so they're carried here, keyed the same as
# models.yml. Purely a presentation column — nothing computes from it.
PARAMS = {
    "qwen_3_0p6b": "0.6B", "falcon_3_1b": "1B", "gemma_3_1b": "1B",
    "llama_3p2_1b": "1B", "stablelm_2_1p6b": "1.6B", "qwen_3_1p7b": "1.7B",
    "qwen_1p5_1p8b": "1.8B", "llama_3p2_3b": "3B", "phi_3p5_mini": "3.8B",
    "qwen_3_4b": "4B", "gemma_3_4b": "4B", "mistral_7b": "7B",
    "olmo_2_7b": "7B", "llama_3_8b": "8B", "llama_3p1_8b": "8B",
    "qwen_3_8b": "8B", "granite_3p3_8b": "8B", "recurrentgemma_9b": "9B",
    "yi_1p5_9b": "9B", "glm_4_9b": "9B", "gemma_3_12b": "12B",
}


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


# ── Collection ────────────────────────────────────────────────────────────────

def parse_console_log(path):
    """layer_idx, num_layers, gate_threshold and total runtime are only ever
    printed, never serialized — scrape them back out of the run's console log."""
    out = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")

    m = re.search(r"^layer_idx = int\((\d+) \* ([\d.]+)\) = (\d+)", text, re.M)
    if m:
        out["num_layers"] = int(m.group(1))
        out["mult_factor"] = float(m.group(2))
        out["abl_layer"] = int(m.group(3))
    m = re.search(r"gate_threshold=(-?[\d.]+)\)", text)
    if m:
        out["gate_threshold"] = float(m.group(1))
    m = re.search(r"total time taken:\s*(\S+)", text)
    if m:
        out["runtime"] = m.group(1)
    return out


def latest_run_dir(results_dir):
    runs = [p for p in results_dir.iterdir() if p.is_dir()] if results_dir.is_dir() else []
    return max(runs, key=lambda p: p.name) if runs else None


def category_stats(df):
    return {
        "n": len(df),
        "orig": int((df["original_judgement"] == df["expected_label"]).sum()),
        "abl": int((df["ablated_judgement"] == "COMPLY").sum()),
        "clf": int((df["classifier_judgement"] == df["expected_label"]).sum()),
        "escalated": int((df["fast_gate_output"] == "SUSPICIOUS").sum()),
        "orig_ts": df["original_ts"].mean(),
        "abl_ts": df["ablated_ts"].mean(),
        "clf_ts": df["classification_ts"].mean(),
        "gate_ts": df["fast_gate_ts"].mean(),
    }


def collect(key, model_id, observations_root, results_root):
    row = {"key": key, "model_id": model_id, "params": PARAMS.get(key, "—")}
    row.update(parse_console_log(results_root / f"{key}.txt"))

    stats_path = observations_root / key / "signature_stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        baseline = s["baseline_all_dims_all_layers"]
        layers = s["signature_layers"]
        row.update({
            "d_model": len(baseline["combined"]["dims"]),
            "sig_band": f"{layers[0]}–{layers[-1]}",
            "sig_d": s["combined"]["cohens_d"],
            "sig_ood": s["classification"]["ood"]["accuracy"] * 100,
            "sig_f1": s["classification"]["ood"]["f1"],
            "base_d": baseline["combined"]["cohens_d"],
            "base_ood": baseline["classification"]["ood"]["accuracy"] * 100,
        })

    run = latest_run_dir(results_root / key)
    if run is None:
        return None
    row["run"] = run.name
    for category, prefix in (("harmfull", "hf"), ("harmless", "hl")):
        xlsx = run / f"{category}.xlsx"
        if not xlsx.exists():
            return None
        for stat, value in category_stats(pd.read_excel(xlsx)).items():
            row[f"{prefix}_{stat}"] = value
    return row


# ── Rendering ─────────────────────────────────────────────────────────────────

def heat(value, low, high, invert=False):
    """Green→red background scaled between low and high, so a 21-row table can
    be read by shape rather than by comparing 21 numbers by eye."""
    if value is None:
        return ""
    frac = max(0.0, min(1.0, (value - low) / (high - low) if high != low else 0.0))
    if invert:
        frac = 1.0 - frac
    hue = 8 + frac * 122  # 8 = red, 130 = green
    return f' style="background:hsl({hue:.0f} 72% 92%)"'


def report_links(row):
    key, run = row["key"], row["run"]
    targets = [
        ("act", f"{PAGES_BASE}/observations/{key}/activation_analysis.html", "Stage 2 — activation analysis"),
        ("sig", f"{PAGES_BASE}/observations/{key}/signature_report.html", "Stage 3 — signature validation"),
        ("cmp", f"{PAGES_BASE}/results/{key}/{run}/comparison_report.html", "Stage 7 — comparison report"),
    ]
    return " ".join(
        f'<a href="{esc(url)}" title="{esc(tip)}" target="_blank" rel="noopener">{label}</a>'
        for label, url, tip in targets
    )


def render_row(row):
    ratio = row["d_model"] * (row["num_layers"] + 1)
    coverage = 36 / ratio * 100
    return f"""
    <tr>
      <td class="sticky-col"><code>{esc(row['key'])}</code><br>
          <a class="muted" href="https://huggingface.co/{esc(row['model_id'])}" target="_blank" rel="noopener">{esc(row['model_id'])}</a></td>
      <td class="num">{esc(row['params'])}</td>
      <td class="num">{row['num_layers']}</td>
      <td class="num">{row['d_model']}</td>
      <td class="num">{row['abl_layer']}</td>
      <td class="num">{esc(row['sig_band'])}</td>
      <td class="num">{row['gate_threshold']:.4f}</td>
      <td class="num"{heat(row['sig_d'], 1.4, 8.0)}>{row['sig_d']:.2f}</td>
      <td class="num muted">{row['base_d']:.2f}</td>
      <td class="num"{heat(row['sig_ood'], 91, 100)}>{row['sig_ood']:.1f}</td>
      <td class="num muted">{row['base_ood']:.1f}</td>
      <td class="num muted">{coverage:.3f}%</td>
      <td class="num sep"{heat(row['hf_orig'], 24, 100)}>{row['hf_orig']}</td>
      <td class="num"{heat(row['hf_abl'], 7, 100)}>{row['hf_abl']}</td>
      <td class="num muted">{row['hf_escalated']}</td>
      <td class="num"{heat(row['hf_clf'], 91, 100)}><b>{row['hf_clf']}</b></td>
      <td class="num sep">{row['hl_orig']}</td>
      <td class="num"{heat(row['hl_abl'], 57, 100)}>{row['hl_abl']}</td>
      <td class="num muted">{row['hl_escalated']}</td>
      <td class="num"{heat(row['hl_clf'], 91, 100)}><b>{row['hl_clf']}</b></td>
      <td class="num sep">{row['hf_gate_ts'] * 1000:.0f}</td>
      <td class="num">{row['hf_orig_ts']:.2f}</td>
      <td class="num">{row['hf_abl_ts']:.2f}</td>
      <td class="num">{row['hf_clf_ts']:.2f}</td>
      <td class="num sep">{row['hl_orig_ts']:.2f}</td>
      <td class="num">{row['hl_abl_ts']:.2f}</td>
      <td class="num">{row['hl_clf_ts']:.2f}</td>
      <td class="num sep muted">{esc(row.get('runtime', '—'))}</td>
      <td class="links">{report_links(row)}</td>
    </tr>"""


def render_footer(rows):
    def rng(field, fmt="{:.0f}"):
        values = [r[field] for r in rows]
        return f"{fmt.format(min(values))}–{fmt.format(max(values))}"

    return f"""
    <tr class="agg">
      <td class="sticky-col">range across {len(rows)} models</td>
      <td colspan="6"></td>
      <td class="num">{rng('sig_d', '{:.2f}')}</td><td></td>
      <td class="num">{rng('sig_ood', '{:.1f}')}</td><td colspan="2"></td>
      <td class="num sep">{rng('hf_orig')}</td>
      <td class="num">{rng('hf_abl')}</td>
      <td class="num">{rng('hf_escalated')}</td>
      <td class="num">{rng('hf_clf')}</td>
      <td class="num sep">{rng('hl_orig')}</td>
      <td class="num">{rng('hl_abl')}</td>
      <td class="num">{rng('hl_escalated')}</td>
      <td class="num">{rng('hl_clf')}</td>
      <td class="num sep">{rng('hf_gate_ts', '{:.3f}')}</td>
      <td colspan="7"></td>
      <td></td>
    </tr>"""


CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 22px 40px; background: #fff; color: #16191d;
       font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size: 21px; margin: 0 0 4px; letter-spacing: -.01em; }
.sub { color: #5c6570; font-size: 13px; margin: 0 0 20px; max-width: 82ch; }
.wrap { overflow-x: auto; border: 1px solid #dfe3e8; border-radius: 8px; }
table { border-collapse: separate; border-spacing: 0; font-size: 12.5px; white-space: nowrap; }
th, td { padding: 6px 9px; border-bottom: 1px solid #eceff2; text-align: left; }
thead th { position: sticky; top: 0; background: #f6f8fa; font-weight: 600;
           border-bottom: 1px solid #dfe3e8; z-index: 2; }
thead tr.groups th { text-align: center; font-size: 11px; letter-spacing: .06em;
                     text-transform: uppercase; color: #5c6570; background: #eef1f4; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.sticky-col { position: sticky; left: 0; background: #fff; z-index: 1;
              border-right: 1px solid #dfe3e8; white-space: nowrap; }
thead .sticky-col { background: #f6f8fa; z-index: 3; }
tbody tr:hover td { background: #f4f7fb; }
tbody tr:hover .sticky-col { background: #f4f7fb; }
.sep { border-left: 1px solid #dfe3e8; }
.muted, .muted a { color: #6b7480; font-weight: 400; }
code { font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
a { color: #0b62d0; text-decoration: none; }
a:hover { text-decoration: underline; }
.links a { display: inline-block; padding: 1px 6px; margin-right: 3px; font-size: 11px;
           border: 1px solid #cfd6de; border-radius: 4px; background: #fbfcfd; }
tr.agg td { background: #f6f8fa; font-size: 11.5px; color: #5c6570; border-top: 1px solid #dfe3e8; }
.notes { margin: 16px 0 0; padding: 0 0 0 18px; color: #5c6570; font-size: 12.5px; max-width: 100ch; }
.notes li { margin: 3px 0; }
"""

GROUPS = [
    ("", 1), ("Model", 3), ("Refusal direction &amp; signature (Stage 2–3)", 8),
    ("Harmful OOD — 100 prompts (Stage 6)", 4),
    ("Harmless OOD — 100 prompts (Stage 6)", 4),
    ("Mean latency, harmful (s)", 4), ("Mean latency, harmless (s)", 3),
    ("Run", 2),
]

COLUMNS = [
    ("Model", "sticky-col"), ("Params", "num"), ("Layers", "num"), ("d_model", "num"),
    ("Abl. layer", "num"), ("Sig. band", "num"), ("Gate θ", "num"),
    ("Cohen's d", "num"), ("d (all-dims)", "num muted"), ("OOD acc %", "num"),
    ("OOD acc % (all-dims)", "num muted"), ("Coords used", "num muted"),
    ("Orig. refused", "num sep"), ("Abl. complied", "num"), ("Gate escalated", "num muted"), ("C++ blocked", "num"),
    ("Orig. complied", "num sep"), ("Abl. complied", "num"), ("Gate escalated", "num muted"), ("C++ allowed", "num"),
    ("FastGate (ms)", "num sep"), ("Original", "num"), ("Abliterated", "num"), ("Classifier++", "num"),
    ("Original", "num sep"), ("Abliterated", "num"), ("Classifier++", "num"),
    ("Wall clock", "num sep muted"), ("Reports", "links"),
]


def build_html(rows):
    groups = "".join(
        f'<th class="groups-cell" colspan="{span}">{title}</th>' if title else "<th></th>"
        for title, span in GROUPS
    )
    headers = "".join(f'<th class="{cls}">{title}</th>' for title, cls in COLUMNS)
    body = "".join(render_row(r) for r in rows)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-Model Transfer Summary</title>
<style>{CSS}</style></head>
<body>
<h1>Cross-Model Transfer — {len(rows)} instruction-tuned models</h1>
<p class="sub">Every row is one complete pipeline run: activations collected, refusal direction and
Cohen's-d signature fit <em>on that model</em>, then evaluated over the same held-out 100 harmful +
100 harmless out-of-distribution prompts. Counts are out of 100. Pass definitions match each model's
own <code>comparison_report.html</code> exactly.</p>
<div class="wrap">
<table>
  <thead>
    <tr class="groups">{groups}</tr>
    <tr>{headers}</tr>
  </thead>
  <tbody>{body}{render_footer(rows)}</tbody>
</table>
</div>
<ul class="notes">
  <li><b>Orig. refused / Orig. complied</b> — the un-ablated model's own behaviour. A low
      harmful figure means weak baseline refusal training, not a pipeline failure.</li>
  <li><b>Abl. complied</b> — harmful prompts the ablated model answered: the attack's success
      rate. Higher = ablation worked. On the harmless side it is a damage check — a drop below
      ~99 means ablation degraded ordinary helpfulness.</li>
  <li><b>Gate escalated</b> — prompts FastGate scored at or above <span class="mono">gate θ</span>
      and forwarded to the ExchangeClassifier. On the harmless side this is the false-alarm count.</li>
  <li><b>C++ blocked / allowed</b> — the two-stage Constitutional Classifiers++ verdict, the
      headline safety-restoration number.</li>
  <li><b>Coords used</b> — the 36-coordinate signature as a fraction of the full
      <span class="mono">(layers+1) × d_model</span> activation tensor the all-dims baseline scores.</li>
  <li>Latencies exclude the LLM-as-Judge call in every condition. Classifier++ latency includes a full
      response generation on the cleared path, so it stays comparable to the other two columns.</li>
</ul>
</body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", default="observations")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="observations/cross_model_summary.html")
    args = parser.parse_args()

    observations_root, results_root = Path(args.observations), Path(args.results)

    rows, skipped = [], []
    for key, model_id in load_models().items():
        row = collect(key, model_id, observations_root, results_root)
        (rows if row else skipped).append(row if row else key)

    if not rows:
        raise FileNotFoundError("No completed runs found — run workflow.sh for at least one model first.")
    if skipped:
        print(f"Skipped (no completed run): {', '.join(skipped)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(rows), encoding="utf-8")
    print(f"Saved -> {out_path}  ({len(rows)} models)")
