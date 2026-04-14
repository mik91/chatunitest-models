import argparse
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


PROMPT_TEMPLATE = """mode=COMPLETION
projectPath=unknown
assertionStyle=JUNIT
staticSnapshot:
{context}
runtimeFacts:

### JUnit Test:
"""


@dataclass
class TrainConfig:
    dataset_name: str
    split: str
    base_model: str
    hf_username: str
    output_model_name: str
    max_samples: int
    max_token_len: int
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    save_steps: int
    push_to_hub: bool


def build_full_text(context: str, test: str) -> str:
    return PROMPT_TEMPLATE.format(context=context.strip()) + test.strip()


def has_assertion(test: str) -> bool:
    assertion_keywords = ["assert", "Assert", "verify", "Verify", "fail(", "Fail("]
    return any(keyword in test for keyword in assertion_keywords)


def is_valid_java(code: str) -> bool:
    return code.count("{") > 0 and abs(code.count("{") - code.count("}")) <= 2


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def build_training_dataset(dataset_name: str, split: str, max_samples: int, max_token_len: int) -> Dataset:
    print(f"Loading dataset: {dataset_name} ({split})")
    raw = load_dataset(dataset_name, split=split)
    print(f"Raw samples: {len(raw)}")
    print(f"Columns: {raw.column_names}")

    col_context = next((c for c in ["context", "input", "source"] if c in raw.column_names), None)
    col_test = next((t for t in ["test", "output", "target"] if t in raw.column_names), None)

    if col_context is None or col_test is None:
        raise ValueError(f"Could not find context/test columns in {raw.column_names}")

    df = raw.to_pandas()[[col_context, col_test]].copy()
    df.columns = ["context", "test"]

    before = len(df)
    df = df.drop_duplicates(subset=["context"])
    print(f"After dedup: {len(df)} (removed {before - len(df)})")

    df = df[df["test"].apply(has_assertion)]
    print(f"After assertion filter: {len(df)}")

    df = df[df["test"].apply(is_valid_java)]
    print(f"After java structure filter: {len(df)}")

    df = df[df["context"].str.strip().str.len() > 30]
    print(f"After empty context filter: {len(df)}")

    df["text"] = df.apply(lambda row: build_full_text(row["context"], row["test"]), axis=1)
    df["token_est"] = df["text"].apply(estimate_tokens)
    df = df[df["token_est"] <= max_token_len]
    print(f"After token filter (<= {max_token_len}): {len(df)}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(max_samples)
    print(f"Final training samples: {len(df)}")

    if len(df) == 0:
        raise ValueError("No samples left after filtering. Relax filtering constraints.")

    return Dataset.from_pandas(df[["text"]])


def train(config: TrainConfig):
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dataset = build_training_dataset(
        dataset_name=config.dataset_name,
        split=config.split,
        max_samples=config.max_samples,
        max_token_len=config.max_token_len,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = f"./{config.output_model_name}"
    full_hub_name = f"{config.hf_username}/{config.output_model_name}"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        gradient_checkpointing=True,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        report_to="none",
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_token_len,
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)

    if config.push_to_hub:
        print(f"Pushing model to Hugging Face Hub: {full_hub_name}")
        model.push_to_hub(full_hub_name, private=False)
        tokenizer.push_to_hub(full_hub_name, private=False)
        print(f"Done. Model URL: https://huggingface.co/{full_hub_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CodeLlama LoRA for Java JUnit test generation")
    parser.add_argument("--dataset-name", default="zzzghttt/context2test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--base-model", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--hf-username", required=True, help="Your Hugging Face username")
    parser.add_argument("--output-model-name", default="my-testgen-lora")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-token-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--push-to-hub", action="store_true")

    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        base_model=args.base_model,
        hf_username=args.hf_username,
        output_model_name=args.output_model_name,
        max_samples=args.max_samples,
        max_token_len=args.max_token_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        push_to_hub=args.push_to_hub,
    )

    train(cfg)
