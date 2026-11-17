"""Clean results.jsonl: keep the LAST valid record per dedup key."""
import json
import os
from collections import OrderedDict

F = os.environ.get("AS_RESULTS", os.path.join(os.getcwd(), "results")) + "/results.jsonl"
recs = OrderedDict()
bad = 0
for line in open(F):
    try:
        r = json.loads(line)
        key = (r["phase"], r.get("dataset"), r.get("variant"), r.get("dk"),
               r.get("seed"))
        recs[key] = r          # later records overwrite earlier ones
    except Exception:
        bad += 1
with open(F, "w") as f:
    for r in recs.values():
        f.write(json.dumps(r) + "\n")
print(f"kept {len(recs)} records, dropped {bad} corrupt lines")
