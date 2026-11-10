import requests

from .models import clean, parse

OLLAMA_URL = "http://localhost:11434/api/generate"

# Llama 3.1 chat template with the assistant turn already started with "response: ".
# Sent with raw=True so Ollama doesn't wrap it again; the model has to continue
# right after "response: " (i.e. with the 0/1) instead of opening a fresh turn
# where it often refuses ("Lo siento, pero no puedo cumplir con esa solicitud.").
PREFILL_TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\nresponse: "
)


def query_model(prompt, model_id="llama3.1:8b", max_new_tokens=220, temperature=0.0):
    prompt_with_prefill = prompt.rstrip() + "\nresponse: "

    def generate(text, raw=False):
        data = {
            "model": model_id,
            "prompt": text,
            "raw": raw,
            "stream": False,
            "keep_alive": "2h",
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }

        response = requests.post(OLLAMA_URL, json=data, timeout=600)

        if response.status_code != 200:
            return f"Error: {response.status_code}"

        return response.json().get("response", "").strip()

    # First attempt (unchanged protocol)
    raw = generate(prompt_with_prefill)
    full_response = clean(raw)

    score, justification = parse(
        "response: " + full_response,
        fallback="leading_digit",
    )

    # No valid 0/1 (usually a refusal): retry once with a real assistant prefill.
    # (Retrying the same request is pointless at temperature 0 -- it returns the same text.)
    if score is None:
        raw = generate(PREFILL_TEMPLATE.format(prompt=prompt.strip()), raw=True)
        full_response = clean(raw)

        score, justification = parse(
            "response: " + full_response,
            fallback="leading_digit",
        )

    return full_response, score, justification
