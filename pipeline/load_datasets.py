from pathlib import Path

import pandas as pd


class PromptSets:
    """Loads the curated-* prompt sets produced by dataset/store_datasets.py."""

    CATEGORIES = ("harmless", "harmfull")

    def __init__(self, dataset: str):
        self.path = Path(dataset)
        self._store = {category: {"train": None, "test": None} for category in self.CATEGORIES}
        self._load()

    def _load(self):
        for category in self.CATEGORIES:
            category_path = self.path / category
            train_file = category_path / "curated-train_set.xlsx"
            test_file = category_path / "curated-test_set.xlsx"

            if train_file.exists():
                self._store[category]["train"] = pd.read_excel(train_file)
            if test_file.exists():
                self._store[category]["test"] = pd.read_excel(test_file)

    def get(self, category: str, split: str) -> pd.DataFrame:
        return self._store[category][split]


if __name__ == '__main__':

    prompt_sets = PromptSets("dataset/")
    print(prompt_sets.get("harmless", "train").shape)
    print(prompt_sets.get("harmless", "test").shape)
    print(prompt_sets.get("harmfull", "train").shape)
    print(prompt_sets.get("harmfull", "test").shape)