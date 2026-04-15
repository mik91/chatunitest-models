# ChatUniTest Models
Here we provide finetuned models for Java test generation tasks based on Code Llama.

## Models
|        Model        |     Base Model     |   parameters    |                                                  Model Weights                                                  |
|:---------------------------------------------------------------------:|:------------:|:-------------:|:-----------------------------------------------------------------------------------------------------------:|
|        codellama-testgen         |   CodeLlama   |   7B   |       [zzzghttt/CodeLlama-7b-Test-Instruct-lora](https://huggingface.co/zzzghttt/CodeLlama-7b-Test-Instruct-lora)          |
|        codellama-testgen2        |   CodeLlama   |   7B   |       [zzzghttt/TestGen2-lora](https://huggingface.co/zzzghttt/TestGen2-lora)          |

## Usage

1. Start the model server
```python
python model_server.py
```

2. Run a completion or generation route
```python
python chatunitest_completion_route.py
```

3. Then you can make requests by configure the url in ChatUniTest, see example request script in `completion_example.py`.

## Evaluation

Evaluate the fine-tuned model on a held-out split with generation metrics.

```python
python evaluate_model.py --dataset-name zzzghttt/context2test --split test --base-model codellama/CodeLlama-7b-Instruct-hf --lora-model zzzghttt/TestGen2-lora --max-samples 200
```

Metrics reported:

- `exact_match`: normalized string exact match against reference test
- `token_f1`: token-level F1 overlap between prediction and reference
- `assertion_rate`: ratio of generations containing assertion keywords
- `valid_java_rate`: ratio of generations with roughly balanced Java braces

Optional prediction dump:

```python
python evaluate_model.py --output-predictions eval_predictions.jsonl
```

## Details

### Fine-Tuning Method

The fine-tuning process of codellama-testgen2.
![finetune](img/fine-tuning.png)
