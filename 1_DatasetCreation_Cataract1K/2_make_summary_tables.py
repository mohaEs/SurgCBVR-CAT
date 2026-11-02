#!/usr/bin/env python3
"""Print the LaTeX rows of Table 1 (clips per phase and mode) from extracted_clips_nd_dsvr.pkl.

  python make_tables.py /path/to/extracted_clips_nd_dsvr.pkl
"""
import pickle, sys
from collections import defaultdict

MODES = ["fixed5", "fixed10", "proportional", "speed2", "speed3", "speed5"]
clips = pickle.load(open(sys.argv[1], "rb"))
assert len({c["clip_id"] for c in clips}) == len(clips), "duplicate clip IDs: re-extract with the fixed script"

count, cases = defaultdict(int), defaultdict(set)
for c in clips:
    count[(c["phase"], c["mode"])] += 1
    cases[c["phase"]].add(c["case_id"])

phases = sorted(cases)
for p in phases + ["Total"]:
    sel = phases if p == "Total" else [p]
    n_cases = len(set().union(*(cases[q] for q in sel)))
    row = [sum(count[(q, m)] for q in sel) for m in MODES]
    print(f"{p:24s} & {n_cases} & " + " & ".join(map(str, row)) + f" & {sum(row)} \\\\")
print(f"% total clips: {len(clips)}")
