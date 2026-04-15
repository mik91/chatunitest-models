import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPT_TEMPLATE = """mode=COMPLETION
projectPath=unknown
assertionStyle=JUNIT
staticSnapshot:
{context}
runtimeFacts:

### JUnit Test:
"""


@dataclass
class EvalConfig:
    dataset_name: str
    split: str
    base_model: str
    lora_model: Optional[str]
    max_samples: int
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    output_predictions: Optional[str]


def build_prompt(context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context.strip())


def has_assertion(test: str) -> bool:
    assertion_keywords = ["assert", "Assert", "verify", "Verify", "fail(", "Fail("]
    return any(keyword in test for keyword in assertion_keywords)


def is_valid_java(code: str) -> bool:
    return code.count("{") > 0 and abs(code.count("{") - code.count("}")) <= 2


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_f1(text: str) -> List[str]:
    # Keep Java identifiers and punctuation as separate tokens.
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = tokenize_for_f1(pred)
    ref_tokens = tokenize_for_f1(ref)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def detect_columns(column_names: List[str]) -> Tuple[str, str]:
    col_context = next((c for c in ["context", "input", "source"] if c in column_names), None)
    col_test = next((t for t in ["test", "output", "target"] if t in column_names), None)

    if col_context is None or col_test is None:
        raise ValueError(f"Could not find context/test columns in {column_names}")

    return col_context, col_test


def load_eval_samples(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, str]]:
    print(f"Loading evaluation dataset: {dataset_name} ({split})")
    raw = load_dataset(dataset_name, split=split)
    print(f"Raw samples: {len(raw)}")
    print(f"Columns: {raw.column_names}")

    col_context, col_test = detect_columns(raw.column_names)

    rows: List[Dict[str, str]] = []
    for sample in raw:
        context = str(sample[col_context]).strip()
        test = str(sample[col_test]).strip()

        if len(context) < 30:
            continue

        rows.append({"context": context, "reference": test})

    if not rows:
        raise ValueError("No valid evaluation samples found.")

    rows = rows[:max_samples]
    print(f"Evaluation samples after filtering: {len(rows)}")
    return rows


def load_model_and_tokenizer(base_model: str, lora_model: Optional[str]):
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        quantization_config = None
        torch_dtype = torch.float32
        device_map = None

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if lora_model:
        print(f"Loading LoRA adapter: {lora_model}")
        model = PeftModel.from_pretrained(model, lora_model)

    model.eval()
    return tokenizer, model


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_input_tokens,
        padding=False,
        return_tensors="pt",
    )

    if model.device.type != "cpu":
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

    do_sample = temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate(config: EvalConfig) -> Dict[str, float]:
    samples = load_eval_samples(config.dataset_name, config.split, config.max_samples)
    tokenizer, model = load_model_and_tokenizer(config.base_model, config.lora_model)

    exact_matches = 0
    assertion_hits = 0
    valid_java_hits = 0
    token_f1_sum = 0.0

    predictions: List[Dict[str, str]] = []

    for idx, row in enumerate(samples, start=1):
        prompt = build_prompt(row["context"])
        reference = row["reference"]

        prediction = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_input_tokens=config.max_input_tokens,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        if normalize_text(prediction) == normalize_text(reference):
            exact_matches += 1

        if has_assertion(prediction):
            assertion_hits += 1

        if is_valid_java(prediction):
            valid_java_hits += 1

        token_f1_sum += token_f1(prediction, reference)

        if config.output_predictions:
            predictions.append(
                {
                    "id": idx,
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                }
            )

        if idx % 10 == 0 or idx == len(samples):
            print(f"Evaluated {idx}/{len(samples)} samples")

    total = len(samples)
    metrics = {
        "num_samples": total,
        "exact_match": exact_matches / total,
        "token_f1": token_f1_sum / total,
        "assertion_rate": assertion_hits / total,
        "valid_java_rate": valid_java_hits / total,
    }

    if config.output_predictions:
        with open(config.output_predictions, "w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {config.output_predictions}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Java test generation model (base or base+LoRA)")
    parser.add_argument("--dataset-name", default="zzzghttt/context2test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-model", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--lora-model", default="zzzghttt/TestGen2-lora")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-predictions", default="")
    args = parser.parse_args()

    cfg = EvalConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        base_model=args.base_model,
        lora_model=args.lora_model if args.lora_model else None,
        max_samples=args.max_samples,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_predictions=args.output_predictions if args.output_predictions else None,
    )

    metrics = evaluate(cfg)

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and k != "num_samples":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
