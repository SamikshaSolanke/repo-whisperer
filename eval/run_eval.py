import argparse
import json
import time
from collections import defaultdict
from query.retrieve import retrieve

CONFIGS = {
    "1. Dense only":                  dict(hybrid=False, rewrite=False, rerank=False),
    "2. + BM25 hybrid":               dict(hybrid=True,  rewrite=False, rerank=False),
    "3. hybrid + query rewriting":    dict(hybrid=True,  rewrite=True,  rerank=False),
    "4. hybrid + reranker":           dict(hybrid=True,  rewrite=False, rerank=True),
    "5. hybrid + rewrite + reranker": dict(hybrid=True,  rewrite=True,  rerank=True),
    "6. + reranker + code preference":    dict(hybrid=True, rewrite=False, rerank=True, prefer_code=True),
    "7. full + code preference":          dict(hybrid=True, rewrite=True,  rerank=True, prefer_code=True),
}


def evaluate(questions, repo, k, flags, verbose=False):
    hits, rr, n = 0, 0.0, 0
    by_diff = defaultdict(lambda: [0, 0])
    t0 = time.time()
    for q in questions:
        if not q.get("gold_files"):        # negative questions are scored separately
            continue
        chunks = retrieve(q["question"], k=k, repo=repo, **flags)
        paths = [c["path"] for c in chunks]
        rank = next((i for i, p in enumerate(paths, 1) if p in q["gold_files"]), None)
        n += 1
        hits += rank is not None
        rr += 1 / rank if rank else 0.0
        by_diff[q["difficulty"]][0] += rank is not None
        by_diff[q["difficulty"]][1] += 1
        if verbose and rank is None:
            print(f"   MISS q{q['id']}: {q['question']}\n        got: {paths}")
    return {
        "hit": hits / n, "mrr": rr / n, "n": n,
        "latency_s": (time.time() - t0) / n,
        "by_difficulty": {d: f"{h}/{t}" for d, (h, t) in by_diff.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/dataset.json")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--only", help="run a single config by its number, e.g. 2")
    ap.add_argument("-v", "--verbose", action="store_true", help="print misses")
    args = ap.parse_args()

    data = json.load(open(args.data))
    repo = data["repo"]
    rows = {}
    for name, flags in CONFIGS.items():
        if args.only and not name.startswith(args.only + "."):
            continue
        print(f"\n== {name}")
        rows[name] = evaluate(data["questions"], repo, args.k, flags, args.verbose)
        print("  ", rows[name])

    print(f"\n| Config | Hit@{args.k} | MRR@{args.k} | easy | medium | hard | s/query |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        d = r["by_difficulty"]
        print(f"| {name} | {r['hit']:.0%} | {r['mrr']:.2f} | {d.get('easy','-')} | "
              f"{d.get('medium','-')} | {d.get('hard','-')} | {r['latency_s']:.2f} |")

    json.dump(rows, open("eval/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()