import pandas as pd

CURATED_TRAIN_SIZE = 250
CURATED_TEST_SIZE = 100
PROMPT_COLUMN = "text"


def verify_curated_split(curated_train: pd.DataFrame, curated_test: pd.DataFrame):
    """Ensure the curated split has the right sizes and mutually exclusive prompts."""
    assert len(curated_train) == CURATED_TRAIN_SIZE, \
        f"expected {CURATED_TRAIN_SIZE} curated train prompts, got {len(curated_train)}"
    assert len(curated_test) == CURATED_TEST_SIZE, \
        f"expected {CURATED_TEST_SIZE} curated test prompts, got {len(curated_test)}"
    assert not curated_train[PROMPT_COLUMN].duplicated().any(), "duplicate prompts within curated train set"
    assert not curated_test[PROMPT_COLUMN].duplicated().any(), "duplicate prompts within curated test set"

    overlap = set(curated_train[PROMPT_COLUMN]) & set(curated_test[PROMPT_COLUMN])
    assert not overlap, f"curated train/test prompts are not mutually exclusive: {overlap}"


def curate_split(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Sample a curated train/test split with mutually exclusive prompts."""
    curated_train = train_df.sample(n=CURATED_TRAIN_SIZE)
    test_pool = test_df[~test_df[PROMPT_COLUMN].isin(curated_train[PROMPT_COLUMN])]
    curated_test = test_pool.sample(n=CURATED_TEST_SIZE)
    verify_curated_split(curated_train, curated_test)
    return curated_train, curated_test


# Load Harmless Dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}

# Train Set
harmless_train = pd.read_parquet("hf://datasets/mlabonne/harmless_alpaca/" + splits["train"])
harmless_train.to_excel("harmless/train_set.xlsx")

# Test Set
harmless_test = pd.read_parquet("hf://datasets/mlabonne/harmless_alpaca/" + splits["test"])
harmless_test.to_excel("harmless/test_set.xlsx")

# Curated Harmless Set (250 train / 100 test, mutually exclusive prompts)
curated_harmless_train, curated_harmless_test = curate_split(harmless_train, harmless_test)
curated_harmless_train.to_excel("harmless/curated-train_set.xlsx")
curated_harmless_test.to_excel("harmless/curated-test_set.xlsx")

# Load Harmfull Dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}

# Train Set
harmfull_train = pd.read_parquet("hf://datasets/mlabonne/harmful_behaviors/" + splits["train"])
harmfull_train.to_excel("harmfull/train_set.xlsx")

# Test Set
harmfull_test = pd.read_parquet("hf://datasets/mlabonne/harmful_behaviors/" + splits["test"])
harmfull_test.to_excel("harmfull/test_set.xlsx")

# Curated Harmfull Set (250 train / 100 test, mutually exclusive prompts)
curated_harmfull_train, curated_harmfull_test = curate_split(harmfull_train, harmfull_test)
curated_harmfull_train.to_excel("harmfull/curated-train_set.xlsx")
curated_harmfull_test.to_excel("harmfull/curated-test_set.xlsx")