"""
Data preprocessing script: converts zzzghttt/context2test into a training format
that matches the inference prompt.

Prompt format used by model_server.py during inference:
    mode=COMPLETION
    projectPath=...
    assertionStyle=JUNIT
    staticSnapshot:
    {Java method + context}
    runtimeFacts:
    (empty)

Training data uses the same format to keep train/infer consistent.
"""

import re
from datasets import load_dataset, Dataset
import pandas as pd


PROMPT_TEMPLATE = """\
mode=COMPLETION
projectPath=unknown
assertionStyle=JUNIT
staticSnapshot:
{context}
runtimeFacts:

### JUnit Test:
"""


def build_prompt(context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context.strip())


def build_full_sample(context: str, test: str) -> str:
    """During training, concatenate input + output into one full sequence."""
    return build_prompt(context) + test.strip()


def has_assertion(test: str) -> bool:
    """Filter out invalid tests that have no assertions."""
    assertion_keywords = ["assert", "Assert", "verify", "Verify", "fail(", "Fail("]
    return any(kw in test for kw in assertion_keywords)


def is_valid_java(code: str) -> bool:
    """Simple structural validation: braces are roughly balanced."""
    return code.count("{") > 0 and abs(code.count("{") - code.count("}")) <= 2


def estimate_tokens(text: str) -> int:
    """Rough token estimate using character_count / 4."""
    return len(text) // 4


def preprocess(
    dataset_name: str = "zzzghttt/context2test",
    split: str = "train",
    max_samples: int = 5000,
    max_token_len: int = 2048,
    output_path: str = "data/train_formatted.jsonl",
):
    print(f"Loading dataset: {dataset_name} ...")
    raw = load_dataset(dataset_name, split=split)
    print(f"Raw samples: {len(raw)}")

    # Print column names for easier debugging
    print(f"Columns: {raw.column_names}")

    # context2test column names may be 'input'/'output' or 'context'/'test'
    # Auto-detect
    col_context = None
    col_test = None
    for c in ["context", "input", "source"]:
        if c in raw.column_names:
            col_context = c
            break
    for t in ["test", "output", "target"]:
        if t in raw.column_names:
            col_test = t
            break

    if col_context is None or col_test is None:
        print(f"ERROR: Cannot find context/test columns. Available: {raw.column_names}")
        return

    print(f"Using columns: context='{col_context}', test='{col_test}'")

    df = raw.to_pandas()[[col_context, col_test]].copy()
    df.columns = ["context", "test"]

    before = len(df)

    # 1. Deduplicate
    df = df.drop_duplicates(subset=["context"])
    print(f"After dedup: {len(df)} (removed {before - len(df)})")

    # 2. Filter samples without assertions
    df = df[df["test"].apply(has_assertion)]
    print(f"After assertion filter: {len(df)}")

    # 3. Filter structurally invalid test code
    df = df[df["test"].apply(is_valid_java)]
    print(f"After java structure filter: {len(df)}")

    # 4. Filter samples with empty context
    df = df[df["context"].str.strip().str.len() > 30]
    print(f"After empty context filter: {len(df)}")

    # 5. Build prompt and filter over-length samples
    df["full_text"] = df.apply(
        lambda row: build_full_sample(row["context"], row["test"]), axis=1
    )
    df["token_est"] = df["full_text"].apply(estimate_tokens)
    df = df[df["token_est"] <= max_token_len]
    print(f"After token length filter (<={max_token_len} tokens): {len(df)}")

    # 6. Keep first max_samples after random shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(max_samples)
    print(f"Final samples: {len(df)}")

    # 7. Save
    import os
    os.makedirs("data", exist_ok=True)
    df[["full_text"]].to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved to {output_path}")

    # Print one sample
    print("\n=== Sample ===")
    print(df["full_text"].iloc[0][:800])
    print("...")


if __name__ == "__main__":
    preprocess()
