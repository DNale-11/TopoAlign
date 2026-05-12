import json
import pathlib
import statistics

root = pathlib.Path(r"F:\programme\nxy\Cell registration\benchmark\results\unregistration_full")
f1s = []
for p in sorted(root.rglob("diagnostics.json")):
    with open(p) as f:
        d = json.load(f)
    f1 = d.get("quality", {}).get("match_f1", None)
    if f1 is not None:
        f1s.append(f1)
        print(f"  {p.parent.name}: F1={f1:.4f}")

print(f"\nPairs: {len(f1s)}")
print(f"Mean F1: {statistics.mean(f1s):.4f}")
print(f"Median F1: {statistics.median(f1s):.4f}")
print(f"Min F1: {min(f1s):.4f}")
print(f"Max F1: {max(f1s):.4f}")
