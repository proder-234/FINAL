"""OpenAI moderation (omni-moderation-latest) scores for each category, added to *_tox.csv."""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
load_dotenv(os.path.expanduser("~/.env"))

URL = "https://api.openai.com/v1/moderations"
MODEL = "omni-moderation-latest"
HARD_TIMEOUT = 120   # seconds; requests' own timeout does not cover every hang
_poster = ThreadPoolExecutor(4)
CATS = ["harassment", "harassment/threatening", "hate", "hate/threatening", "illicit", "illicit/violent",
        "self-harm", "self-harm/intent", "self-harm/instructions", "sexual", "sexual/minors",
        "violence", "violence/graphic"]
COLS = ["oai_" + c.replace("/", "_").replace("-", "_") for c in CATS]


def post(session, texts):
    """session.post with a wall-clock deadline. On a hang, drop the connection pool and raise."""
    fut = _poster.submit(session.post, URL, json={"model": MODEL, "input": texts}, timeout=(15, 60))
    try:
        return fut.result(timeout=HARD_TIMEOUT)
    except FutureTimeout:
        print(f"  request hung > {HARD_TIMEOUT}s, resetting connection", file=sys.stderr, flush=True)
        for a in list(session.adapters.values()):
            a.close()
        session.mount("https://", requests.adapters.HTTPAdapter())
        raise requests.Timeout("hard timeout")


def moderate(session, texts):
    """Return the {category: score} dict for each text, in order.
    Waits and retries (forever) on rate limits, server errors and dropped connections."""
    attempt = 0
    while True:
        try:
            r = post(session, texts)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200:
            return [res["category_scores"] for res in r.json()["results"]]
        if r is not None and r.status_code != 429 and r.status_code < 500:
            r.raise_for_status()                                 # bad request / bad key: stop
        if (r is not None and r.status_code == 429
                and r.headers.get("x-ratelimit-remaining-tokens") == r.headers.get("x-ratelimit-limit-tokens")):
            # Token budget untouched, so this request alone is over the per-minute token limit: split it.
            if len(texts) > 1:
                print(f"  batch of {len(texts)} over token limit, splitting", file=sys.stderr, flush=True)
                half = len(texts) // 2
                return moderate(session, texts[:half]) + moderate(session, texts[half:])
            print(f"  single text over token limit ({len(texts[0])} chars), truncating", file=sys.stderr, flush=True)
            return moderate(session, [texts[0][:len(texts[0]) // 2]])
        attempt += 1
        print(f"  retry {attempt}: {r.status_code if r is not None else 'connection error'}",
              file=sys.stderr, flush=True)
        time.sleep(min(float(r.headers.get("retry-after", 0)) if r is not None else 0, 60)
                   or min(2 ** attempt, 60))


def fill(df, todo, texts, cache):
    """Copy cached category scores into the todo rows; uncached texts stay NaN."""
    scores = pd.DataFrame.from_dict(cache, orient="index", columns=CATS).reindex(texts[todo])
    df.loc[todo, COLS] = scores.to_numpy()


def score_file(path, session, cache, batch_size, workers):
    df = pd.read_csv(path)
    if "justification" not in df.columns:
        print(f"Skip (no 'justification' column): {path}")
        return
    for col in COLS:
        if col not in df.columns:
            df[col] = float("nan")

    texts = df["justification"].fillna("").astype(str).str.strip()
    todo = df.index[df[COLS].isna().any(axis=1) & (texts != "")]
    if len(todo) == 0:
        print(f"Skip (already scored): {path}")
        return

    # Identical justifications are scored once per run.
    new = list(dict.fromkeys(t for t in texts[todo] if t not in cache))
    batches = [new[i:i + batch_size] for i in range(0, len(new), batch_size)]
    print(f"Scoring {path} ({len(todo)} rows, {len(new)} new texts)")

    def run(batch):
        return batch, moderate(session, batch)

    with ThreadPoolExecutor(workers) as ex:
        for done, (batch, scores) in enumerate(tqdm(ex.map(run, batches), total=len(batches),
                                                    desc="  batches", leave=False), 1):
            cache.update(zip(batch, scores))
            if done % 10 == 0:                                   # checkpoint
                fill(df, todo, texts, cache)
                df.to_csv(path, index=False)

    fill(df, todo, texts, cache)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Add OpenAI moderation scores to *_tox.csv files.")
    parser.add_argument("--pattern", nargs="+", default=["results/*/eval_*_tox.csv"])
    parser.add_argument("--batch_size", type=int, default=32, help="Texts per API request.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests.")
    args = parser.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set (put it in .env)")

    files = [f for pat in args.pattern for f in sorted(glob.glob(pat, recursive=True))]
    if not files:
        print(f"No files match {args.pattern}")
        return

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {key}"
    cache = {}
    for path in files:
        score_file(path, session, cache, args.batch_size, args.workers)
    print("Done!")


if __name__ == "__main__":
    main()
