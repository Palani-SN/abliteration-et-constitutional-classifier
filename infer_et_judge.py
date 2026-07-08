import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from infer_model import Inference_Model
from llm_judge import LLM_as_Judge
from load_datasets import PromptSets

sys.stdout.reconfigure(encoding="utf-8")


class InferAndJudge:

    PROMPT_COLUMN = "text"

    def __init__(self, dataset_path="dataset/", split="test", top_n=5, results_dir="results/",
                 model_id="tiiuae/Falcon3-1B-Instruct", judge_model="gemma4:e4b"):

        self.split = split
        self.top_n = top_n

        self.prompt_sets = PromptSets(dataset_path)
        self.inf_llm = Inference_Model(model_id)
        self.judge = LLM_as_Judge(judge_model)

        self.run_dir = Path(results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print("init done")

    def run(self):
        for category in PromptSets.CATEGORIES:
            self._run_category(category)

    def _run_category(self, category):
        df = self.prompt_sets.get(category, self.split)
        prompts = df[self.PROMPT_COLUMN].head(self.top_n)
        print(f"\n=== {category} ({len(prompts)} prompts) ===")

        records = [self._evaluate(prompt) for prompt in prompts]
        self._save_results(category, records)

    def _evaluate(self, prompt):

        start = time.perf_counter()
        response = self.inf_llm.query(user=prompt)
        time_taken = time.perf_counter() - start

        verdict = self.judge.judge(prompt, response)

        print(f"prompt     : {prompt}")
        print(f"response   : {response}")
        print(f"time_taken : {time_taken:.2f}s")
        print(f"judgement  : {verdict}")
        print()

        return {
            "prompt": prompt,
            "response": response,
            "time_taken": time_taken,
            "judgement": verdict,
        }

    def _save_results(self, category, records):
        out_file = self.run_dir / f"{category}.xlsx"
        pd.DataFrame(records).to_excel(out_file, index=False)
        print(f"saved {out_file}")


if __name__ == '__main__':

    print("start")
    runner = InferAndJudge(
        dataset_path="dataset/",
        split="test",
        top_n=3,
        results_dir="results/",
        model_id="tiiuae/Falcon3-1B-Instruct",
        judge_model="gemma4:e4b",
    )
    runner.run()
