import json
import subprocess
from pathlib import Path

root = Path("data/repos/flask")
data = json.load(open("eval/dataset.json"))

for q in data["questions"]:
    for f in q.get("gold_files", []):
        if not (root / f).exists():
            print(f"q{q['id']}: gold file does not exist: {f}")
    for term in q.get("absent_terms", []):
        out = subprocess.run(
            ["grep", "-rliF", term, str(root / "src"), str(root / "docs")],
            capture_output=True, text=True).stdout.split()
        if out:
            print(f"q{q['id']}: '{term}' appears in {out[:3]} -> not a clean negative, replace it")
print("checked", len(data["questions"]), "questions")