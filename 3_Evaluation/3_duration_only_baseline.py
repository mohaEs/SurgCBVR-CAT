"""
Duration-only baseline for SurgCBVR-CAT (Data in Brief table, last row).

Ranks each query's gallery by clip length alone: sim(q, g) = -|L(q) - L(g)|, with L read
from the clip name (fixed5s -> 5, fixed10s -> 10, prop24s -> 24, speedN -> 5 s output).
It measures how much mAP clip length alone gives. Only the proportional mode has
varying lengths; in every other mode all clips have the same length, so this baseline
equals a random ranking there.

Protocol: identical to 2_evaluate_map.py (imported from the same folder): same-mode
gallery, same-surgery clips excluded, same AP. Lengths are whole seconds, so many clips
tie; ties are broken at random and the result is averaged over N_SEEDS tie-breaks
(mean +- SD), so the score does not depend on the order of the clip list.

Outputs (in OUT_DIR): duration_only_summary.csv, and duration_only_seed0.npy (a similarity
matrix in the same layout as 1_calculate_similarities.py, if you want to run it through
2_evaluate_map.py yourself).
"""
import csv
import importlib.util
import json
import os
import re
from collections import defaultdict

import numpy as np

# ========= Main Mode =========
annotations_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.json"
dataset_ids_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.txt"
OUT_DIR = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Results/Results/"
EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2_evaluate_map.py")
N_SEEDS = 20
# ==============================

spec = importlib.util.spec_from_file_location("evalmap", EVAL_SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)          # reuse parse(), gallery_mode(), average_precision(), settings


def clip_length(name):
    m = re.search(r"_(fixed|prop)(\d+)s_", name)
    return float(m.group(2)) if m else 5.0          # speed clips are rendered to 5 s


def main():
    with open(annotations_file) as f:
        annotations = json.load(f)
    with open(dataset_ids_file) as f:
        dataset = [l.strip() for l in f if l.strip()]
    col = {v: i for i, v in enumerate(dataset)}
    meta = [ev.parse(v) for v in dataset]
    cases = np.array([m[0] for m in meta])
    modes = np.array([m[2] for m in meta])
    L = np.array([clip_length(v) for v in dataset])
    queries = [q for q, g in annotations.items() if ev.TASK_KEY in g]
    print(f"Queries: {len(queries)} | gallery rule: {ev.GALLERY}, same-case excluded: "
          f"{ev.EXCLUDE_SAME_CASE}, speed gallery: {ev.SPEED_GALLERY}")

    # Galleries and positives do not depend on the seed: build them once.
    plan = []
    for q in queries:
        qc, qp, qm = ev.parse(q)
        keep = np.ones(len(dataset), dtype=bool)
        keep[col[q]] = False
        if ev.EXCLUDE_SAME_CASE:
            keep &= cases != qc
        if ev.GALLERY == "same_mode":
            keep &= modes == ev.gallery_mode(qm)
        idx = np.flatnonzero(keep)
        pos = {col[v] for v in annotations[q][ev.TASK_KEY] if v in col}
        is_pos = np.array([i in pos for i in idx])
        if is_pos.sum() == 0:
            continue
        plan.append((q, qp, qm, idx, is_pos, -np.abs(L[idx] - L[col[q]])))

    groups = ev.MODES + ["All"]
    per_seed = defaultdict(list)       # (mode) -> list of mAP over seeds
    per_seed_cls = defaultdict(list)
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        aps = defaultdict(list)
        by_phase = defaultdict(lambda: defaultdict(list))
        for q, qp, qm, idx, is_pos, base in plan:
            scores = base + rng.uniform(0, 1e-3, size=len(idx))   # random tie-break only
            ap = ev.average_precision(scores, is_pos, int(is_pos.sum()))
            for g in (qm, "All"):
                aps[g].append(ap)
                by_phase[g][qp].append(ap)
        for g in groups:
            per_seed[g].append(np.mean(aps[g]) if aps[g] else np.nan)
            per_seed_cls[g].append(np.mean([np.mean(v) for v in by_phase[g].values()])
                                   if by_phase[g] else np.nan)

    rows = []
    print(f"\nDuration-only baseline (mean +- SD over {N_SEEDS} random tie-breaks)")
    print(f"{'mode':10s} {'mAP':>16s} {'class-avg mAP':>18s}")
    for g in groups:
        m, s = np.mean(per_seed[g]), np.std(per_seed[g])
        cm, cs = np.mean(per_seed_cls[g]), np.std(per_seed_cls[g])
        print(f"{g:10s} {m:9.4f} +- {s:.4f} {cm:11.4f} +- {cs:.4f}")
        rows.append({"mode": g, "mAP_mean": m, "mAP_sd": s, "class_avg_mAP_mean": cm,
                     "class_avg_mAP_sd": cs, "n_seeds": N_SEEDS})

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "duration_only_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Seed-0 similarity matrix in the 1_calculate_similarities.py layout (optional cross-check).
    all_q = list(annotations.keys())
    rng = np.random.default_rng(0)
    S = -np.abs(L[[col[q] for q in all_q]][:, None] - L[None, :]).astype(np.float32)
    S += rng.uniform(0, 1e-3, size=S.shape).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "duration_only_seed0.npy"), S)
    print(f"\nSaved duration_only_summary.csv and duration_only_seed0.npy in {OUT_DIR}")


if __name__ == "__main__":
    main()
