import json
import config
from eval.answers_eval import judge
from query.retrieve import retrieve

BEST = dict(hybrid=True, rewrite=True, rerank=False, prefer_code=False)

# Two appended claims that the excerpts cannot support:
FALSE_CLAIM = " Flask also automatically caches every rendered template in Redis for 24 hours."
TRUE_BUT_ABSENT = " Python was created by Guido van Rossum."

data = json.load(open("eval/dataset.json"))
answers = [a for a in json.load(open("eval/answers_dev.json"))
           if not a["refused"] and a["type"] != "negative"][:5]
qs = {q["id"]: q for q in data["questions"]}

print("id | original | + false claim | + true-but-absent claim")
for a in answers:
    chunks = retrieve(a["question"], k=config.TOP_K, repo=data["repo"], **BEST)
    row = []
    for text in (a["answer"], a["answer"] + FALSE_CLAIM, a["answer"] + TRUE_BUT_ABSENT):
        j = judge(a["question"], chunks, text)
        row.append(f"{j['supported']}/{j['total']}" if j else "err")
    print(f"q{a['id']} | " + " | ".join(row))