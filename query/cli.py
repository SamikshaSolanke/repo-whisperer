import argparse
import sys
from query.answer import answer_stream, cited_indices
from query.retrieve import retrieve
from query.rewrite import rewrite_query


def ask(question, repo, k, show_chunks, **flags):
    if show_chunks and flags.get("rewrite"):
        print("Rewrites:", rewrite_query(question))
    chunks = retrieve(question, k=k, repo=repo, **flags)
    if show_chunks:
        print("\n--- Retrieved ---")
        for i, c in enumerate(chunks, 1):
            print(f"[{i}] {c['score']:.3f}  {c['path']}:{c['start_line']}-{c['end_line']}  {c['symbol']}")
        print("-----------------\n")

    full = ""
    for token in answer_stream(question, chunks):
        sys.stdout.write(token)
        sys.stdout.flush()
        full += token
    print("\n")

    used = cited_indices(full, len(chunks))
    print("Sources:")
    for i in used or range(1, len(chunks) + 1):
        c = chunks[i - 1]
        print(f"  [{i}] {c['path']}:{c['start_line']}-{c['end_line']}  {c['symbol']}\n      {c['url']}")
    if not used:
        print("  (the model cited nothing, so all retrieved chunks are listed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="omit for interactive mode")
    ap.add_argument("--repo", default="pallets/flask")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved chunks and scores")
    ap.add_argument("--dense-only", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    flags = dict(hybrid=not args.dense_only,
                 rewrite=not args.no_rewrite,
                 rerank=not args.no_rerank)

    if args.question:
        ask(args.question, args.repo, args.k, args.show_chunks, **flags)
        return

    print("Ask about the repo (Ctrl+C to quit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            ask(q, args.repo, args.k, args.show_chunks, **flags)


if __name__ == "__main__":
    main()