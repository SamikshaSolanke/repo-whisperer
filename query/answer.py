import re
from typing import Iterator
from groq import Groq
import config

SYSTEM = """You are a code assistant that answers questions about a software repository.
Rules:
1. Answer ONLY from the numbered code excerpts provided. Do not use outside knowledge about the project. If a detail is not in the excerpts, leave it out.
2. Cite with exactly this format: [1] or [2][3], placed inline right after each claim. Never use 【 】, †, or line ranges inside the brackets.
3. Do not write a "Sources" section; the application adds one.
4. If the excerpts do not contain the answer, say exactly: "I couldn't find this in the indexed code." Then briefly say what the excerpts do cover.
5. Mention function, class, and file names precisely as they appear.
6. Be concise. When you quote code, copy it exactly as it appears without eliding lines, and keep it to a few lines. Quote documentation only if the sentence appears verbatim in an excerpt.
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['path']}:{c['start_line']}-{c['end_line']}  ({c['kind']}: {c['symbol']})"
        parts.append(f"{header}\n```\n{c['text']}\n```")
    return "\n\n".join(parts)


def answer_stream(question: str, chunks: list[dict]) -> Iterator[str]:
    """Yield answer tokens as they arrive from Groq."""
    client = Groq(api_key=config.GROQ_API_KEY)
    user = f"Code excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    stream = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        temperature=config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS,
        stream=True,
    )
    for event in stream:
        token = event.choices[0].delta.content
        if token:
            yield token


def cited_indices(answer: str, n_chunks: int) -> list[int]:
    """Return the sorted chunk numbers the model actually cited (1-based)."""
    nums = {int(m) for m in re.findall(r"[\[【](\d+)(?:†[^\]】]*)?[\]】]", answer)}
    return sorted(n for n in nums if 1 <= n <= n_chunks)