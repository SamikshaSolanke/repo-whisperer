# import argparse
# import json
# import time
# from collections import defaultdict
# from query.retrieve import retrieve

# CONFIGS = {
#     "1. Dense only":                  dict(hybrid=False, rewrite=False, rerank=False),
#     "2. + BM25 hybrid":               dict(hybrid=True,  rewrite=False, rerank=False),
#     "3. hybrid + query rewriting":    dict(hybrid=True,  rewrite=True,  rerank=False),
#     "4. hybrid + reranker":           dict(hybrid=True,  rewrite=False, rerank=True),
#     "5. hybrid + rewrite + reranker": dict(hybrid=True,  rewrite=True,  rerank=True),
#     "6. + reranker + code preference":    dict(hybrid=True, rewrite=False, rerank=True, prefer_code=True),
#     "7. full + code preference":          dict(hybrid=True, rewrite=True,  rerank=True, prefer_code=True),
# }


# def evaluate(questions, repo, k, flags, verbose=False):
#     hits, rr, n = 0, 0.0, 0
#     by_diff = defaultdict(lambda: [0, 0])
#     t0 = time.time()
#     for q in questions:
#         if not q.get("gold_files"):        # negative questions are scored separately
#             continue
#         chunks = retrieve(q["question"], k=k, repo=repo, **flags)
#         paths = [c["path"] for c in chunks]
#         rank = next((i for i, p in enumerate(paths, 1) if p in q["gold_files"]), None)
#         n += 1
#         hits += rank is not None
#         rr += 1 / rank if rank else 0.0
#         by_diff[q["difficulty"]][0] += rank is not None
#         by_diff[q["difficulty"]][1] += 1
#         if verbose and rank is None:
#             print(f"   MISS q{q['id']}: {q['question']}\n        got: {paths}")
#     return {
#         "hit": hits / n, "mrr": rr / n, "n": n,
#         "latency_s": (time.time() - t0) / n,
#         "by_difficulty": {d: f"{h}/{t}" for d, (h, t) in by_diff.items()},
#     }


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data", default="eval/dataset.json")
#     ap.add_argument("-k", type=int, default=5)
#     ap.add_argument("--only", help="run a single config by its number, e.g. 2")
#     ap.add_argument("-v", "--verbose", action="store_true", help="print misses")
#     args = ap.parse_args()

#     data = json.load(open(args.data))
#     repo = data["repo"]
#     rows = {}
#     for name, flags in CONFIGS.items():
#         if args.only and not name.startswith(args.only + "."):
#             continue
#         print(f"\n== {name}")
#         rows[name] = evaluate(data["questions"], repo, args.k, flags, args.verbose)
#         print("  ", rows[name])

#     print(f"\n| Config | Hit@{args.k} | MRR@{args.k} | easy | medium | hard | s/query |")
#     print("|---|---|---|---|---|---|---|")
#     for name, r in rows.items():
#         d = r["by_difficulty"]
#         print(f"| {name} | {r['hit']:.0%} | {r['mrr']:.2f} | {d.get('easy','-')} | "
#               f"{d.get('medium','-')} | {d.get('hard','-')} | {r['latency_s']:.2f} |")

#     json.dump(rows, open("eval/results.json", "w"), indent=2)


# if __name__ == "__main__":
#     main()

import argparse
import json
import re
import time
from pathlib import Path
from groq import Groq
import config
from query.answer import answer_stream, cited_indices
from query.retrieve import retrieve

# Must match the default pipeline chosen on the dev split (row 3)
BEST = dict(hybrid=True, rewrite=True, rerank=False, prefer_code=False)
REFUSAL = "I couldn't find this in the indexed code"
JUDGE_MODEL = getattr(config, "JUDGE_MODEL", config.GROQ_MODEL)

JUDGE = """You check whether an answer about a codebase is supported by the code excerpts it was given.
Split the answer into its factual claims. For each claim, decide whether the excerpts support it.
Return ONLY JSON: {"total": <int>, "supported": <int>, "unsupported": [<short strings>]}"""


def with_retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "tokens per day" in msg:
                raise SystemExit("Daily Groq token limit reached. Progress is saved; "
                                 "re-run later to resume.")
            print(f"   retry {i + 1}: {msg[:120]}")
            time.sleep(15 * (i + 1))
    return None


def judge(question, chunks, answer):
    ctx = "\n\n".join(f"[{i}] {c['path']}\n{c['text']}" for i, c in enumerate(chunks, 1))
    r = Groq(api_key=config.GROQ_API_KEY).chat.completions.create(
        model=JUDGE_MODEL, temperature=0, max_tokens=1500, reasoning_effort="low",
        messages=[{"role": "system", "content": JUDGE},
                  {"role": "user",
                   "content": f"Excerpts:\n{ctx}\n\nQuestion: {question}\n\nAnswer:\n{answer}"}])
    m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
    return json.loads(m.group(0)) if m else None


def summarize(results):
    neg = [r for r in results if r["type"] == "negative"]
    pos = [r for r in results if r["type"] != "negative"]
    answered = [r for r in pos if not r["refused"]]
    faith = [r["faithfulness"] for r in pos if "faithfulness" in r]
    print("\n--- Generation results ---")
    print(f"Questions scored: {len(results)}")
    print(f"Correct refusals on negatives: {sum(r['refused'] for r in neg)}/{len(neg)}")
    print(f"False refusals on answerable:  {sum(r['refused'] for r in pos)}/{len(pos)}")
    print(f"Answers with a valid citation: {sum(bool(r['cited']) for r in answered)}/{len(answered)}")
    if faith:
        print(f"Mean faithfulness (LLM judge): {sum(faith) / len(faith):.2f} over {len(faith)} answers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--n", type=int, help="only the first N questions (saves API quota)")
    args = ap.parse_args()

    data = json.load(open("eval/dataset.json"))
    qs = [q for q in data["questions"] if args.split in ("all", q.get("split", "dev"))]
    if args.n:
        qs = qs[: args.n]

    # Resume support: keep earlier successful answers, retry the ones that failed
    out_path = Path(f"eval/answers_{args.split}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    results = [r for r in results if not r.get("error")]
    done = {r["id"] for r in results}
    if done:
        print(f"Resuming: {len(done)} question(s) already answered in {out_path}")

    for q in qs:
        if q["id"] in done:
            continue
        print(f"q{q['id']} [{q['type']}] {q['question']}")
        chunks = retrieve(q["question"], k=config.TOP_K, repo=data["repo"], **BEST)
        answer = with_retry(lambda: "".join(answer_stream(q["question"], chunks)))
        if answer is None or not answer.strip():
            print("   FAILED after retries; skipped (re-run to retry it)")
            continue

        refused = REFUSAL.lower() in answer.lower()
        rec = {"id": q["id"], "type": q["type"], "question": q["question"],
               "refused": refused, "cited": cited_indices(answer, len(chunks)),
               "answer": answer}
        if q["type"] != "negative" and not refused:
            j = with_retry(lambda: judge(q["question"], chunks, answer))
            if j and j.get("total"):
                rec["faithfulness"] = min(1.0, j["supported"] / j["total"])
                rec["unsupported"] = j.get("unsupported", [])
        results.append(rec)
        out_path.write_text(json.dumps(results, indent=2))   # save after every question
        time.sleep(20)                                        # stay under the per-minute limit

    summarize(results)


if __name__ == "__main__":
    main()