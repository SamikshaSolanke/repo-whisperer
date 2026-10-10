import json
import re
from pathlib import Path
from groq import Groq
import config

CACHE = Path(".cache/rewrites.json")

PROMPT = """You help search a Python codebase. Rewrite the user's question into {n} short search queries.
Use general technical vocabulary and class/function names you are confident exist. Do NOT invent identifiers.
Each query should take a different angle (e.g. the concept, the data structure, the mechanism). Return ONLY a JSON array of strings."""


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def rewrite_query(question: str, n: int = config.REWRITE_N) -> list[str]:
    cache = _load()
    key = f"{config.GROQ_MODEL}|{n}|{question}"
    if key in cache:
        return cache[key]
    try:
        r = Groq(api_key=config.GROQ_API_KEY).chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": PROMPT.format(n=n)},
                      {"role": "user", "content": question}],
            temperature=0,
            max_tokens=1024,          # reasoning models spend part of this thinking
            reasoning_effort="low",
        )
        text = r.choices[0].message.content or ""
        arr = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        queries = [q for q in arr if isinstance(q, str) and q.strip()][:n]
    except Exception as e:                 # never let a rewrite failure break search
        print(f"[rewrite failed: {e}]")
        return []
    CACHE.parent.mkdir(exist_ok=True)
    cache[key] = queries
    CACHE.write_text(json.dumps(cache, indent=2))

    return queries