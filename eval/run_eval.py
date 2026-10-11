import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import config
from query.retrieve import retrieve

DEFAULT_COLLECTION = config.COLLECTION

CONFIGS = {
    "0. Fixed chunks, dense only":       dict(collection="code_fixed", hybrid=False, rewrite=False, rerank=False),
    "1. AST chunks, dense only":         dict(hybrid=False, rewrite=False, rerank=False),
    "2. + BM25 hybrid":                  dict(hybrid=True,  rewrite=False, rerank=False),
    "3. hybrid + query rewriting":       dict(hybrid=True,  rewrite=True,  rerank=False),
    "4. hybrid + reranker":              dict(hybrid=True,  rewrite=False, rerank=True),
    "5. reranker + code preference":     dict(hybrid=True,  rewrite=False, rerank=True, prefer_code=True),
    "6. full (rewrite + rerank + code)": dict(hybrid=True,  rewrite=True,  rerank=True, prefer_code=True),
}


def evaluate(questions, repo, k, flags, verbose=False):
    flags = dict(flags)
    config.COLLECTION = flags.pop("collection", DEFAULT_COLLECTION)
    hits, rr = 0, 0.0
    by_type = defaultdict(lambda: [0, 0])
    t0 = time.time()
    try:
        for q in questions:
            qtype = q.get("type", "direct")
            chunks = retrieve(q["question"], k=k, repo=repo, **flags)
            paths = [c["path"] for c in chunks]
            rank = next((i for i, p in enumerate(paths, 1) if p in q["gold_files"]), None)
            hits += rank is not None
            rr += 1 / rank if rank else 0.0
            by_type[qtype][0] += rank is not None
            by_type[qtype][1] += 1
            if verbose and rank is None:
                print(f"   MISS q{q['id']} [{qtype}]: {q['question']}\n        got: {paths}")
    finally:
        config.COLLECTION = DEFAULT_COLLECTION
    n = max(len(questions), 1)
    return {"hit": hits / n, "mrr": rr / n, "n": len(questions),
            "latency_s": (time.time() - t0) / n, "by_type": dict(by_type)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/dataset.json")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--only", help="config number(s), comma-separated, e.g. 3 or 1,2,3")
    ap.add_argument("--penalty", type=float, help="override NON_CODE_PENALTY (rows 5-6)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print misses")
    args = ap.parse_args()

    if args.penalty is not None:
        config.NON_CODE_PENALTY = args.penalty

    data = json.load(open(args.data))
    repo = data["repo"]
    qs = [q for q in data["questions"]
          if q.get("gold_files") and args.split in ("all", q.get("split", "dev"))]
    types = sorted({q.get("type", "direct") for q in qs})
    print(f"Split: {args.split}   questions with gold files: {len(qs)}")

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    rows = {}
    for name, flags in CONFIGS.items():
        if only and name.split(".")[0] not in only:
            continue
        print(f"\n== {name}")
        rows[name] = evaluate(qs, repo, args.k, flags, args.verbose)
        r = rows[name]
        print(f"   Hit@{args.k}={r['hit']:.0%}  MRR={r['mrr']:.2f}  {r['latency_s']:.2f}s/query")

    print(f"\n| Config | Hit@{args.k} | MRR@{args.k} | " + " | ".join(types) + " | s/query |")
    print("|---|---|---|" + "---|" * len(types) + "---|")
    for name, r in rows.items():
        cols = " | ".join(f"{r['by_type'][t][0]}/{r['by_type'][t][1]}" if t in r["by_type"] else "-"
                          for t in types)
        print(f"| {name} | {r['hit']:.0%} | {r['mrr']:.2f} | {cols} | {r['latency_s']:.2f} |")

    # Merge into the existing results file so separate runs don't overwrite each other
    out = Path(f"eval/results_{args.split}.json")
    existing = json.loads(out.read_text()) if out.exists() else {}
    if args.penalty is None:                      # penalty experiments aren't the canonical rows
        existing.update(rows)
        out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()