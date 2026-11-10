# ETHICS Utilitarianism: Multilingual Evaluation

Each item is a pair of scenarios. The model marks the more ethical one `0` and the other `1`; `label` is Scenario 2's value.

- **Languages:** en, hi, ne, de, zh, es, fr (translated from English with NLLB-200 3.3B)
- **Splits:** `t` = test, `th` = test_hard
- **Prompts:** `base`, `base_eg`, `lang`, `lang_eg` (`_eg` = with 32 examples, `lang` = instructions in the target language; English uses `base` and `base_eg` only)
- **Metrics:** accuracy and toxicity

| Model key | Model |
|---|---|
| `mistral` | Mistral-7B-Instruct-v0.2 |
| `qwen` | Qwen3-8B (4-bit, MLX) |
| `llama` | Llama 3.1 8B (Ollama) |
| `gpt_oss` | gpt-oss 20B (Ollama) |
| `gemma` | Gemma 4 E4B-it |
| `glm` | GLM-4-9B-Chat |

## Setup

```bash
pip install -r requirements.txt
```

`.env`: `HUGGINGFACE_API_KEY` (mistral, gemma, glm) and `OPENAI_API_KEY` (toxicity_openai.py).
`llama` and `gpt_oss` need a local Ollama server; `qwen` needs Apple Silicon.

## Run

```bash
python translate.py --split th
python inference.py --split th --lang hi --model qwen --prompt lang_eg --limit 0
python toxicity.py --pattern "results/qwen/eval_*_th.csv"
python toxicity_openai.py --pattern "results/qwen/eval_*_th_tox.csv"
python metrics.py --split th --model qwen --prompt lang_eg
```

## Results

Results are zipped; unzip into `results/` to reuse them.

- `results/data.zip`: `ethics_dataset_<split>.csv` (English items and labels) and `ethics_translated_<split>.csv`
- `results/<model>/<lang>.zip`:
  - `eval_<lang>_<model>_<prompt>_<split>.csv`: model output per item
  - `eval_<lang>_<model>_<prompt>_<split>_tox.csv`: same, plus `tox_*` (Detoxify) and `oai_*` (OpenAI moderation) scores for each toxicity category
