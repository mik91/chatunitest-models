import os
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel


app = Flask(__name__)

BASE_MODEL = os.getenv("BASE_MODEL", "codellama/CodeLlama-7b-Instruct-hf")
LORA_MODEL = os.getenv("LORA_MODEL", "zzzghttt/TestGen2-lora")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2048"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "1234"))


def build_prompt(data: dict) -> str:
    if "input" in data:
        return str(data["input"])

    mode = data.get("mode", "COMPLETION")
    project_path = data.get("projectPath", "unknown")
    assertion_style = data.get("assertionStyle", "JUNIT")
    static_snapshot = data.get("staticSnapshot", "")
    runtime_facts = data.get("runtimeFacts", "")

    return (
        f"mode={mode}\n"
        f"projectPath={project_path}\n"
        f"assertionStyle={assertion_style}\n"
        f"staticSnapshot:\n{static_snapshot}\n"
        f"runtimeFacts:\n{runtime_facts}\n\n"
        "### JUnit Test:\n"
    )


def load_model():
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model in 8-bit: {BASE_MODEL}")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter: {LORA_MODEL}")
    model = PeftModel.from_pretrained(model, LORA_MODEL, torch_dtype=torch.float16)
    model.eval()
    return tokenizer, model


def generate_completion(prompt: str, max_new_tokens: int, temperature: float) -> str:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
        return_tensors="pt",
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    generation_config = GenerationConfig(
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )

    prompt_len = encoded["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "baseModel": BASE_MODEL, "loraModel": LORA_MODEL}), 200


@app.route("/generation", methods=["POST"])
def completion():
    data = request.get_json(silent=True) or {}

    try:
        prompt = build_prompt(data)
        max_new_tokens = int(data.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))

        generated_text = generate_completion(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        return jsonify(
            {
                "success": True,
                "message": "ok",
                "result": generated_text,
                "Result": generated_text,
                "files": [
                    {
                        "relativePath": "se/kth/castor/generated/HybridRockyTest.java",
                        "content": generated_text,
                    }
                ],
            }
        ), 200
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc), "files": []}), 500


if __name__ == "__main__":
    tokenizer, model = load_model()
    app.run(debug=False, host=HOST, port=PORT, threaded=True)
