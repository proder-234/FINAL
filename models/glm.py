import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import clean, parse

load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

MODEL_ID = "zai-org/glm-4-9b-chat-hf"

print(f"[models/glm.py] Loading {MODEL_ID}")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=huggingface_api_key)
_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto", token=huggingface_api_key)
_model.eval()


def query_model(prompt, max_new_tokens=220):   # same as models/mistral.py
    text = _tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt", add_special_tokens=False).to(_model.device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    full_response = clean(raw, truncate=True)
    score, justification = parse(full_response, fallback="bare_line")
    return full_response, score, justification
