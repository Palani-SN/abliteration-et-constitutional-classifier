import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from abliterate import Abliterator
from load_datasets import PromptSets

import torch

from classify import Classifier
from models import resolve_model
from utils.console import print_record

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# VERIFICATION — ORIGINAL VS RUNTIME-ABLATED MODEL, PLUS CLASSIFIER JUDGMENT
#
# Composes Abliterator's model/hook/generate mechanics (one model loaded once,
# ablation toggled on/off via enable_ablation()/disable_ablation()) — there is
# no separate saved "abliterated model" to load, since abliteration is applied
# at runtime.
#
#   Batch mode:  two sequential phases per category, not interleaved:
#                  1. Generation  — for top_n OOD prompts, generate+judge under
#                     both conditions (Abliterator's model).
#                  2. Unmount Abliterator's model, then load a fresh
#                     classify.Classifier (must run on the ORIGINAL model —
#                     see constitutional_classifier.py's note — which is moot
#                     here since Abliterator's copy is already gone) and
#                     classify the SAME prompts (Fast Gate + Exchange
#                     Classifier).
#                Both phases' records are merged per prompt and written to
#                results/<timestamp>/{harmless,harmfull}.xlsx.
#   Prompt mode: interactive REPL — for each typed prompt, show both the
#                original and ablated response (no judge — for quick manual
#                reading, matching remove-refusals-with-transformers/inference.py).
# =============================================================================

EXPECTED_LABEL = {"harmfull": "REFUSE", "harmless": "COMPLY"}


class Verifier:

    def __init__(self, model_id=None, dataset_path="dataset/",
                 activations_dir="activations", direction_path=None, signature_path=None,
                 results_dir="results/", top_n=None, judge_model="gemma4:e4b", max_new_tokens=256):

        if model_id is None:
            _, model_id = resolve_model()

        self.model_id = model_id
        self.dataset_path = dataset_path
        self.activations_dir = Path(activations_dir)
        self.signature_path = Path(signature_path) if signature_path else self.activations_dir / "signature.pt"
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens
        self.results_dir = Path(results_dir)

        self.abliterator = Abliterator(
            model_id=model_id,
            dataset_path=dataset_path,
            activations_dir=activations_dir,
            direction_path=direction_path,
            top_n=top_n,
            judge_model=judge_model,
            max_new_tokens=max_new_tokens,
        )

    # ── shared setup ─────────────────────────────────────────────────────────

    def _setup(self):
        self.abliterator.load_model()
        checkpoint = self.abliterator.load_direction()
        print(f"Loaded direction from layer {checkpoint['layer_idx']} (mult_factor={checkpoint['mult_factor']})")

    # ── batch mode ───────────────────────────────────────────────────────────

    def _evaluate_prompt(self, prompt, index, total, label):
        self.abliterator.disable_ablation()
        start = time.perf_counter()
        original_response = self.abliterator.generate(prompt)
        original_ts = time.perf_counter() - start
        original_judgement = self.abliterator.judge.judge(prompt, original_response)

        self.abliterator.enable_ablation()
        start = time.perf_counter()
        ablated_response = self.abliterator.generate(prompt)
        ablated_ts = time.perf_counter() - start
        ablated_judgement = self.abliterator.judge.judge(prompt, ablated_response)
        self.abliterator.disable_ablation()

        print_record(
            f"[{index}/{total}] {label}  original={original_judgement}  ablated={ablated_judgement}",
            [
                ("prompt", prompt),
                ("original_response", original_response),
                ("original_ts", f"{original_ts:.2f}s"),
                ("original_judgement", original_judgement),
                ("ablated_response", ablated_response),
                ("ablated_ts", f"{ablated_ts:.2f}s"),
                ("ablated_judgement", ablated_judgement),
            ],
        )

        return {
            "prompt":             prompt,
            "original_response":  original_response,
            "original_ts":        original_ts,
            "original_judgement": original_judgement,
            "ablated_response":   ablated_response,
            "ablated_ts":         ablated_ts,
            "ablated_judgement":  ablated_judgement,
        }

    def _unmount_abliterator(self):
        # Frees the abliteration model's GPU memory before loading the
        # classifier's own copy, rather than sharing one live model instance
        # across both phases -- keeps ablation-hook state and classification
        # state from ever touching the same object.
        print("\nUnmounting abliteration model ...")
        self.abliterator.model = None
        self.abliterator.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def _classify_prompt(self, classifier, prompt, expected_label, index, total, label):
        # Orchestrates FastGate + ExchangeClassifier directly instead of calling
        # classifier.classify() -- that convenience method stays cheap (no
        # generation) on the clear path, which is correct for real deployment
        # but makes classification_ts incomparable to original_ts/ablated_ts
        # (which always include a full generation). Here, when clear, we still
        # generate an actual response so every row's latency reflects "cost to
        # produce something," and judge it the same way original/ablated are
        # judged -- but the judge call is excluded from classification_ts, just
        # like it's excluded from original_ts/ablated_ts.
        start = time.perf_counter()
        suspicious, gate_score = classifier.fast_gate.is_suspicious(prompt)
        fast_gate_ts = time.perf_counter() - start

        if suspicious:
            verdict, reason = classifier.exchange_classifier.classify(prompt)
            classification_ts = time.perf_counter() - start
            exchange_classifier_output = verdict
            exchange_classifier_reason = reason
            classifier_judgement = "REFUSE" if verdict == "BLOCK" else "COMPLY"
        else:
            response = classifier.exchange_classifier.generate(prompt, max_new_tokens=self.max_new_tokens)
            classification_ts = time.perf_counter() - start
            exchange_classifier_output = "ALLOW"
            exchange_classifier_reason = response
            classifier_judgement = self.abliterator.judge.judge(prompt, response)  # not timed

        fast_gate_output = "SUSPICIOUS" if suspicious else "CLEAR"

        print_record(
            f"[{index}/{total}] {label}  expected={expected_label}  classifier={classifier_judgement}",
            [
                ("prompt", prompt),
                ("fast_gate_score", f"{gate_score:.3f}"),
                ("fast_gate_output", fast_gate_output),
                ("fast_gate_ts", f"{fast_gate_ts:.2f}s"),
                ("exchange_classifier_output", exchange_classifier_output),
                ("exchange_classifier_reason", exchange_classifier_reason),
                ("classification_ts", f"{classification_ts:.2f}s"),
                ("classifier_judgement", classifier_judgement),
            ],
        )

        return {
            "expected_label":             expected_label,
            "fast_gate_score":            gate_score,
            "fast_gate_output":           fast_gate_output,
            "fast_gate_ts":               fast_gate_ts,
            "exchange_classifier_output": exchange_classifier_output,
            "exchange_classifier_reason": exchange_classifier_reason,
            "classification_ts":          classification_ts,
            "classifier_judgement":       classifier_judgement,
        }

    def run_batch(self):
        self._setup()

        run_dir = self.results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: generation (original vs ablated), for every category ────
        category_prompts = {}
        category_gen_records = {}
        for category in PromptSets.CATEGORIES:
            prompts = self.abliterator.load_prompts(category, "test")
            print(f"\n=== {category} ({len(prompts)} OOD prompts) — generation ===")

            category_prompts[category] = prompts
            category_gen_records[category] = [
                self._evaluate_prompt(p, i, len(prompts), category.upper())
                for i, p in enumerate(prompts, 1)
            ]

        # ── Phase 2: classification, on a freshly loaded model ───────────────
        self._unmount_abliterator()

        classifier = Classifier(
            model_id=self.model_id,
            dataset_path=self.dataset_path,
            signature_path=self.signature_path,
            top_n=self.top_n,
            max_new_tokens=self.max_new_tokens,
        )
        classifier.load_model()
        classifier.load_signature()

        for category in PromptSets.CATEGORIES:
            prompts = category_prompts[category]
            expected_label = EXPECTED_LABEL[category]
            print(f"\n=== {category} ({len(prompts)} OOD prompts) — classification ===")

            records = []
            for i, (prompt, gen_record) in enumerate(zip(prompts, category_gen_records[category]), 1):
                clf_record = self._classify_prompt(
                    classifier, prompt, expected_label, i, len(prompts), category.upper(),
                )
                records.append({
                    "expected_label":     clf_record["expected_label"],
                    "prompt":             prompt,
                    "original_response":  gen_record["original_response"],
                    "original_ts":        gen_record["original_ts"],
                    "original_judgement": gen_record["original_judgement"],
                    "ablated_response":   gen_record["ablated_response"],
                    "ablated_ts":         gen_record["ablated_ts"],
                    "ablated_judgement":  gen_record["ablated_judgement"],
                    "fast_gate_score":            clf_record["fast_gate_score"],
                    "fast_gate_output":           clf_record["fast_gate_output"],
                    "fast_gate_ts":               clf_record["fast_gate_ts"],
                    "exchange_classifier_output": clf_record["exchange_classifier_output"],
                    "exchange_classifier_reason": clf_record["exchange_classifier_reason"],
                    "classification_ts":          clf_record["classification_ts"],
                    "classifier_judgement":       clf_record["classifier_judgement"],
                })

            out_file = run_dir / f"{category}.xlsx"
            pd.DataFrame(records).to_excel(out_file, index=False)
            print(f"saved {out_file}")

    # ── prompt mode (REPL) ───────────────────────────────────────────────────

    def run_prompt(self):
        self._setup()

        print("\nType a prompt to compare original vs ablated responses (Ctrl+C to exit).")
        while True:
            prompt = input("> ")
            if not prompt.strip():
                continue

            self.abliterator.disable_ablation()
            original_response = self.abliterator.generate(prompt)

            self.abliterator.enable_ablation()
            ablated_response = self.abliterator.generate(prompt)
            self.abliterator.disable_ablation()

            print(f"\n[ORIGINAL] {original_response.strip()}")
            print(f"\n[ABLATED]  {ablated_response.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "prompt"], default="batch")
    parser.add_argument("--model", default=None, help="Model key from models.yml (default: first entry)")
    parser.add_argument("--top_n", type=int, default=None, help="Cap OOD prompts per category (default: full 100+100 held-out test set)")
    args = parser.parse_args()

    model_key, model_id = resolve_model(args.model)
    print(f"Using model '{model_key}' -> {model_id}" + (f", top_n={args.top_n}" if args.top_n else ""))

    verifier = Verifier(
        model_id=model_id,
        dataset_path="dataset/",
        activations_dir=f"activations/{model_key}",
        results_dir=f"results/{model_key}",
        top_n=args.top_n,
        judge_model="gemma4:e4b",
    )

    if args.mode == "batch":
        verifier.run_batch()
    else:
        verifier.run_prompt()
