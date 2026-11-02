#!/usr/bin/env python3
"""
Check that every ND-DSVR clip contains only its own phase.

  python check_clip_purity.py --pkl extracted_clips_nd_dsvr.pkl \
      --annotations /path/Phase_recognition_dataset_224/annotations [--margin 1.0] [--probe]

For each clip it checks, against ALL annotated segments of its case (Idle included):
  outside_phase   the clip's source window leaves its own phase segment
  overlaps_other  the window overlaps another annotated segment (other phase or Idle),
                  which happens only if the annotation CSV has overlapping segments
  near_boundary   the window starts/ends within --margin s of a phase boundary
                  (annotation boundaries are approximate; such clips may show the
                  neighbouring phase for a moment)
  short_file      (--probe) the encoded file is >0.2 s shorter than intended,
                  e.g. because the phase annotation runs past the end of the video

Also reports overlapping segments found directly in the annotation files.
Works with the old pkl (no 'source_duration' field) and the fixed one.
"""
import argparse
import csv
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def load_all_segments(ann_dir):
    segs = defaultdict(list)
    for case_dir in Path(ann_dir).iterdir():
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        cid = case_dir.name.replace("case_", "")
        f = case_dir / f"case_{cid}_annotations_phases.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                segs[cid].append((r["comment"].strip(), float(r["sec"]), float(r["endSec"])))
    return segs


def source_window(c):
    if "source_duration" in c:
        return c["start_time"], c["start_time"] + c["source_duration"]
    m = re.search(r"_speed(\d+)_", c["clip_id"])
    sp = int(m.group(1)) if m else 1
    return c["start_time"], c["start_time"] + sp * c["duration"]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(out) if out else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--probe", action="store_true", help="also check encoded file durations")
    ap.add_argument("--out", default="clip_purity_report.csv")
    a = ap.parse_args()

    segs = load_all_segments(a.annotations)
    eps = 1e-3

    print("Overlapping segments in the annotation files:")
    n_ov = 0
    for cid, ss in sorted(segs.items()):
        ss = sorted(ss, key=lambda x: x[1])
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                if ss[j][1] < ss[i][2] - eps:
                    n_ov += 1
                    print(f"  case {cid}: {ss[i]} overlaps {ss[j]}")
    print(f"  -> {n_ov} overlapping pairs\n")

    clips = pickle.load(open(a.pkl, "rb"))
    rows, flags = [], Counter()
    for c in clips:
        s, e = source_window(c)
        own = [x for x in segs[c["case_id"]] if x[0] == c["phase"] and x[1] - eps <= s and e <= x[2] + eps]
        f = []
        if not own:
            f.append("outside_phase")
            seg = None
        else:
            seg = own[0]
            if s - seg[1] < a.margin or seg[2] - e < a.margin:
                f.append("near_boundary")
        others = [x for x in segs[c["case_id"]] if x is not seg and x[1] < e - eps and x[2] > s + eps]
        if others:
            f.append("overlaps_other")
        if a.probe:
            d = probe(c["clip_path"])
            if d < c["duration"] - 0.2:
                f.append("short_file")
        flags.update(f or ["ok"])
        rows.append({"clip_id": c["clip_id"], "phase": c["phase"], "src_start": round(s, 3),
                     "src_end": round(e, 3), "flags": ";".join(f),
                     "other_labels": ";".join(sorted({x[0] for x in others}))})

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Clips: {len(clips)}")
    for k, v in flags.most_common():
        print(f"  {k:16s} {v}")
    print(f"Details per clip: {a.out}")


if __name__ == "__main__":
    main()
