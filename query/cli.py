import argparse
import sys
from query.answer import answer_stream, cited_indices
from query.retrieve import retrieve


def ask(question: str, repo: str | None, k: int, show_chunks: bool):
    chunks = retrieve(question, k=k, repo=repo)
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
    args = ap.parse_args()

    if args.question:
        ask(args.question, args.repo, args.k, args.show_chunks)
        return
    print("Ask about the repo (Ctrl+C to quit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            ask(q, args.repo, args.k, args.show_chunks)


if __name__ == "__main__":
    main()