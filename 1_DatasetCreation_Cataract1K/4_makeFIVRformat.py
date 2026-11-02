import json
from pathlib import Path


# ---- paths (edit if needed) ----
ANNOTATION_FORMAT_FILE = Path("./FIVR_annotation.json")          # only used to learn which keys exist


# ========= Main Mode =========
DS_FILE = Path("./annotations_CBVR_CAT.json") 
OUT_FILE = Path("./annotations_CBVR_CAT.json")
# ==================


INPUT = DS_FILE
OUTPUT = OUT_FILE

with INPUT.open() as f:
    annotations = json.load(f)

merged = {}

for item in annotations:
    q = item["query"]
    positives = item.get("positive", [])

    merged.setdefault(q, {"SPVR": []})
    merged[q]["SPVR"].extend(
        p for p in positives if p != q
    )

for q in merged:
    merged[q]["SPVR"] = list(dict.fromkeys(merged[q]["SPVR"]))

with OUTPUT.open("w") as f:
    json.dump(merged, f, indent=2)

print(f"Total queries: {len(merged)}")
print(f"Saved to: {OUTPUT}")