import os
import sys
from pathlib import Path

# load_datasets (pandas/pyarrow) must be imported before torch — importing torch
# first causes a deterministic access-violation crash inside pyarrow on Windows.
from load_datasets import PromptSets

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_judge import LLM_as_Judge
from utils.console import print_record

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# RUNTIME DYNAMIC ABLATION
#
# Mirrors remove-refusals-with-transformers/inference.py: the refusal direction
# is projected out of every layer's output at inference time via forward hooks
# — no permanent weight editing, no saved model. Ablation is only active while
# hooks are registered, so the same loaded model can toggle between original
# and ablated behavior (this is what verify.py composes this class for).
#
# Run standalone, this is a quick sanity check: generate on top_n harmful +
# top_n harmless OOD prompts with ablation active, judge each with Qwen3, and
# print the result so you can eyeball whether harmful prompts now comply and
# harmless prompts still comply.
# =============================================================================


class Abliterator:

    DEVICE = "cuda"
    PROMPT_COLUMN = "text"

    def __init__(self, model_id="tiiuae/Falcon3-1B-Instruct", dataset_path="dataset/",
                 activations_dir="activations", direction_path=None, top_n=3,
                 judge_model="gemma4:e4b", max_new_tokens=128):

        self.model_id = model_id
        self.activations_dir = Path(activations_dir)
        self.direction_path = Path(direction_path) if direction_path else self.activations_dir / "direction.pt"
        self.top_n = top_n
        self.max_new_tokens = max_new_tokens

        self.prompt_sets = PromptSets(dataset_path)
        self.judge = LLM_as_Judge(judge_model)

        self.tokenizer = None
        self.model = None
        self.direction = None
        self._hook_handles = []

    # ── prompts ──────────────────────────────────────────────────────────────

    def load_prompts(self, category, split):
        df = self.prompt_sets.get(category, split)
        prompts = df[self.PROMPT_COLUMN] if self.top_n is None else df[self.PROMPT_COLUMN].head(self.top_n)
        return prompts.tolist()

    # ── model / direction loading ────────────────────────────────────────────

    def load_model(self):
        if self.model is not None:
            return
        print(f"Loading {self.model_id} in bfloat16 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            clean_up_tokenization_spaces=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def load_direction(self):
        checkpoint = torch.load(self.direction_path)
        self.direction = checkpoint["direction"].to(self.DEVICE, dtype=torch.bfloat16)
        return checkpoint

    # ── ablation hooks ───────────────────────────────────────────────────────

    @staticmethod
    def _make_ablation_hook(direction):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            proj = (hidden @ direction).unsqueeze(-1) * direction
            hidden = hidden - proj
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook

    def enable_ablation(self):
        if self._hook_handles:
            return  # already enabled
        hook = self._make_ablation_hook(self.direction)
        self._hook_handles = [self.model.model.embed_tokens.register_forward_hook(hook)]
        for layer in self.model.model.layers:
            self._hook_handles.append(layer.register_forward_hook(hook))

    def disable_ablation(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    # ── quick sanity check ───────────────────────────────────────────────────

    def _check_category(self, category, label):
        prompts = self.load_prompts(category, "test")
        print(f"\n{label} — {len(prompts)} OOD prompts (ablation ON) ...")
        for i, p in enumerate(prompts, 1):
            text = self.generate(p)
            verdict = self.judge.judge(p, text)
            print_record(f"[{i}/{len(prompts)}] {label}  verdict={verdict}", [
                ("prompt", p),
                ("response", text.strip()),
            ])

    def run(self):
        self.load_model()
        checkpoint = self.load_direction()
        print(f"Loaded direction from layer {checkpoint['layer_idx']} (mult_factor={checkpoint['mult_factor']})")

        self.enable_ablation()
        self._check_category("harmfull", "HARMFUL")
        self._check_category("harmless", "HARMLESS")
        self.disable_ablation()


if __name__ == "__main__":
    abliterator = Abliterator(
        model_id="tiiuae/Falcon3-1B-Instruct",
        dataset_path="dataset/",
        activations_dir="activations",
        top_n=3,
        judge_model="gemma4:e4b",
    )
    abliterator.run()
