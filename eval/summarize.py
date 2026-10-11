import json
import sys

for split in ("dev", "test"):
    try:
        res = json.load(open(f"eval/answers_{split}.json"))
    except FileNotFoundError:
        continue
    empty = [r["id"] for r in res if not r.get("error") and not r.get("answer", "").strip()]
    res = [r for r in res if not r.get("error") and r.get("answer", "").strip()]
    neg = [r for r in res if r["type"] == "negative"]
    pos = [r for r in res if r["type"] != "negative"]
    answered = [r for r in pos if not r["refused"]]
    faith = [r["faithfulness"] for r in pos if "faithfulness" in r]
    print(f"\n=== {split} ({len(res)} questions scored) ===")
    print(f"Empty/failed answers excluded: {empty}")
    print(f"Correct refusals on negatives: {sum(r['refused'] for r in neg)}/{len(neg)}")
    print(f"False refusals on answerable:  {sum(r['refused'] for r in pos)}/{len(pos)}")
    print(f"Answers with a valid citation: {sum(bool(r['cited']) for r in answered)}/{len(answered)}")
    if faith:
        print(f"Mean faithfulness: {sum(faith) / len(faith):.2f} over {len(faith)} answers")
    low = sorted((r for r in pos if r.get("faithfulness", 1) < 0.8), key=lambda r: r["faithfulness"])
    for r in low[:5]:
        print(f"  low faithfulness q{r['id']}: {r['faithfulness']:.2f}  unsupported: {r.get('unsupported')}")