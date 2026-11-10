import os
import re
import requests

from .models import clean, parse

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


def query_model(prompt, model_id="gpt-oss:20b", max_new_tokens=220, temperature=0.0, think="low"):
    """Same prompt, prefill, token cap and retry as llama.py. gpt-oss always reasons first:
    think="low" keeps that short (the reasoning counts toward max_new_tokens and is not saved)."""
    prompt_with_prefill = prompt.rstrip() + "\nresponse: "

    def generate():
        data = {
            "model": model_id,
            "prompt": prompt_with_prefill,
            "stream": False,
            "think": think,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }

        response = requests.post(OLLAMA_URL, json=data)

        if response.status_code != 200:
            return f"Error: {response.status_code}"

        # gpt-oss repeats the prefilled "response:" label itself; drop it so parse() sees one label.
        text = response.json().get("response", "").strip()
        return re.sub(r"^\s*\**\s*response\s*:\s*", "", text, flags=re.IGNORECASE)

    # First attempt
    raw = generate()
    full_response = clean(raw)

    score, justification = parse(
        "response: " + full_response,
        fallback="leading_digit",
    )

    # Retry once if no valid 0/1 response
    if score is None:
        raw = generate()
        full_response = clean(raw)

        score, justification = parse(
            "response: " + full_response,
            fallback="leading_digit",
        )

    return full_response, score, justification
