import argparse
import hashlib
import os
import sys
from itertools import product
from pathlib import Path

from models import cli_model_arg, configure_hf_offline_mode, resolve_model

# Resolved here, before transformers/huggingface_hub is imported, since
# HF_HUB_OFFLINE is read once at that import — see models.py.
_early_model_id = resolve_model(cli_model_arg())[1]
configure_hf_offline_mode(_early_model_id)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from load_datasets import PromptSets

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# REFUSAL DIRECTION ACTIVATION COLLECTION
# bfloat16, full precision | last prompt-token residual stream, all layers
# Model is selected dynamically via --model <key> from models.yml.
# =============================================================================


class ActivationCollector:

    DEVICE = "cuda"
    PROMPT_COLUMN = "text"
    SPLITS = ("train", "test")

    def __init__(self, model_id=None, dataset_path="dataset/",
                 save_dir="activations", top_n=None):

        if model_id is None:
            _, model_id = resolve_model()

        self.top_n = top_n
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.prompt_sets = PromptSets(dataset_path)

        print(f"Loading {model_id} in bfloat16 ...")
        self.tokenizer, self.model = self._load_model(model_id)

    def _load_model(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return tokenizer, model

    def run(self):
        for category, split in product(PromptSets.CATEGORIES, self.SPLITS):
            name = f"{category}_{split}"
            print(f"Collecting activations for '{name}' ...")
            self._collect_set(category, split)
        print("Done.")

    def _load_prompts(self, category, split):
        df = self.prompt_sets.get(category, split)
        prompts = df[self.PROMPT_COLUMN] if self.top_n is None else df[self.PROMPT_COLUMN].head(self.top_n)
        return prompts.tolist()

    @staticmethod
    def _prompt_id(prompt):
        # Content-addressed filename: if a prompt's text ever changes (e.g. after
        # re-curating with a different seed/dataset), its hash changes too, so a
        # stale cached activation can never be silently reused for new content.
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def _last_token_hidden_states(self, prompt):
        # Activations are taken at the last prompt token, right before generation
        # starts — this is the position where the model "decides" to refuse or
        # comply, before any response tokens exist.
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False so the prompt-side template stays consistent
        # with abliterate.py/classify.py's generation calls (harmless no-op
        # for models without a "thinking mode" template).
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.DEVICE)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # hidden_states: (num_layers + 1) tensors of [1, seq_len, d_model]
        # (index 0 is the embedding layer, 1..L are transformer layer outputs)
        last_token = torch.stack([h.squeeze(0)[-1] for h in out.hidden_states], dim=0)
        return last_token.cpu()  # [num_layers + 1, d_model]

    def _collect_set(self, category, split):
        name = f"{category}_{split}"
        prompts = self._load_prompts(category, split)
        out_dir = self.save_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        valid_hashes = {self._prompt_id(p) for p in prompts}

        pending = [
            (i, p) for i, p in enumerate(prompts)
            if not (out_dir / f"{self._prompt_id(p)}.pt").exists()
        ]
        if not pending:
            print(f"  [{name}] all {len(prompts)} activations already collected — skipping.")
        else:
            for i, prompt in pending:
                acts = self._last_token_hidden_states(prompt)
                torch.save(acts, out_dir / f"{self._prompt_id(prompt)}.pt")
                print(f"  [{name}] {i + 1}/{len(prompts)} done — shape={tuple(acts.shape)}")

        # remove cached activations for prompts no longer in the current set
        orphans = [
            f for f in os.listdir(out_dir)
            if f.endswith(".pt") and f[:-3] not in valid_hashes
        ]
        for f in orphans:
            (out_dir / f).unlink()
        if orphans:
            print(f"  [{name}] removed {len(orphans)} stale activation(s) no longer matching the current prompt set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model key from models.yml (default: first entry)")
    args = parser.parse_args()

    model_key, model_id = resolve_model(args.model)
    print(f"Using model '{model_key}' -> {model_id}")

    collector = ActivationCollector(
        model_id=model_id,
        dataset_path="dataset/",
        save_dir=f"activations/{model_key}",
        top_n=None,
    )
    collector.run()
