import json

d = json.load(open("eval/dataset.json"))
for q in d["questions"]:
    q.setdefault("type", "direct")
    q.setdefault("split", "dev")
json.dump(d, open("eval/dataset.json", "w"), indent=2)
print("tagged", len(d["questions"]), "questions")