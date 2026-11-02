"""
mAP evaluation for SurgCBVR-CAT (Data in Brief: SPVR task).

Protocol (all gallery rules live here; the similarity file can contain every pair):
  For each query, the gallery is every clip in the dataset list EXCEPT
    * the query itself,
    * clips from the query's own surgery            (EXCLUDE_SAME_CASE),
    * clips of another mode                         (GALLERY = "same_mode"),
  then the gallery is ranked by similarity and
    AP = (1/|P|) * sum_k P(k) * rel(k),   P = positives that are IN that gallery.
  Queries whose filtered gallery contains no positive are skipped and counted.
  Gallery clips missing from the similarity file are treated as ranked last.

Modes are read from clip names:
  <case>_<phase>_fixed5s_..., _fixed10s_, _prop<L>s_, _speed<s>_   ->  fixed5s, fixed10s, prop, speed2...
"All" = mean of all per-query APs, each computed in its own mode's gallery.
Also reports class-averaged mAP (mean over phases of the per-phase mAP).

Inputs
  annotations_file : FIVR-style JSON {query: {TASK_KEY: [positive clips], ...}}
  dataset_ids_file : clip list (same order as the feature matrix)
  result_file      : either the similarity JSON {query: {clip: sim}} or the .npy matrix
                     saved by 1_calculate_similarities.py (rows = annotation queries in
                     file order, columns = dataset_ids_file order)
Outputs (next to result_file): <name>_per_query.csv and <name>_summary.csv


    # Retrieval Evaluation Settings (`2_evaluate_map.py`)
    These three settings at the top of `2_evaluate_map.py` decide which clips each query is ranked against (the gallery) and which rows the report shows. Clips from the query's own surgery are always removed first, because `EXCLUDE_SAME_CASE = True`.

    ## `GALLERY`
    Whether a query is searched among its own mode or among all modes.

    | Value | A 5 s query (`fixed5s`) is ranked against… | Use |
    |---|---|---|
    | `"same_mode"` | Only 5 s clips from other surgeries | Your Data in Brief table, as in the caption |
    | `"all"` | Every clip from other surgeries: 5 s, 10 s, proportional, all speeds | One mixed gallery, as in your old evaluation |

    ## `SPEED_GALLERY`
    A special rule for speed queries. It applies only with `GALLERY = "same_mode"`.

    | Value | A 3x query (`speed3`) is ranked against… | What it measures |
    |---|---|---|
    | `"same"` | Only 3x clips from other surgeries | Retrieval among speed-up clips; matches your caption |
    | `"fixed5s"` | 5 s normal-speed clips from other surgeries | Robustness: can a speed-up query find normal-speed clips of the same phase? This is the FIVR-style use of speed variants |

    ## `MODES`
    The list of modes, in the order they're reported (`fixed5s`, `fixed10s`, `prop`, `speed2`, `speed3`, `speed5`). It only controls which rows appear in the summary table and in what order.

    Every proportional length (`prop12s`, `prop24s`, …) is grouped under `prop`. Change it only if you add or drop a mode, for example by removing `speed5`.

    ## Recommended defaults for Data in Brief

    ```python
    GALLERY = "same_mode"
    SPEED_GALLERY = "same"
    ```

    These give one mAP per mode, exactly as your caption describes.

"""
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

# ========= Main Mode =========
annotations_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.json"
dataset_ids_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.txt"

result_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Results/SimilarityScores/video_features_vit_base_patch14_dinov2_mean_every_k_5.json"
saveDir = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Results/Results/"
print(result_file)
os.makedirs(saveDir, exist_ok=True)

TASK_KEY = "SPVR"            # key of the positive list in the annotation JSON (e.g. "ND" or "SPVR")
TASK_NAME = TASK_KEY        # name used in the printed report
EXCLUDE_SAME_CASE = True
GALLERY = "same_mode"      # "same_mode" (paper table) or "all" (one mixed gallery)
SPEED_GALLERY = "same"     # "same": speed-s queries vs speed-s clips (DiB caption);
                           # "fixed5s": speed queries vs the 5 s fixed clips (robustness protocol)
MODES = ["fixed5s", "fixed10s", "prop", "speed2", "speed3", "speed5"]
# ==============================

# phase may contain underscores (e.g. Viscoelastic_Suction): it is everything between the
# case ID and the mode token
NAME_RE = re.compile(r"^(?P<case>[^_]+)_(?P<phase>.+?)_(?P<mode>fixed\d+s|prop\d+s|speed\d+)_offset-?\d+_\d+\.mp4$")


def parse(name):
    m = NAME_RE.match(os.path.basename(name))
    if not m:
        raise ValueError(f"Cannot parse clip name (expected the fixed/prop/speed naming): {name}")
    mode = "prop" if m["mode"].startswith("prop") else m["mode"]
    return m["case"], m["phase"], mode


def gallery_mode(query_mode):
    if query_mode.startswith("speed") and SPEED_GALLERY == "fixed5s":
        return "fixed5s"
    return query_mode


def average_precision(scores, is_pos, n_pos):
    """scores/is_pos over the ranked gallery; n_pos = positives in the gallery."""
    order = np.argsort(-scores, kind="stable")
    hits = is_pos[order]
    if n_pos == 0:
        return np.nan
    ranks = np.flatnonzero(hits) + 1                       # 1-based ranks of positives
    return float(np.sum(np.arange(1, len(ranks) + 1) / ranks) / n_pos)


def main():
    with open(annotations_file) as f:
        annotations = json.load(f)
    with open(dataset_ids_file) as f:
        dataset = [l.strip() for l in f if l.strip()]
    assert len(set(dataset)) == len(dataset), "duplicate clip names in the dataset list"
    col = {v: i for i, v in enumerate(dataset)}
    meta = [parse(v) for v in dataset]
    cases = np.array([m[0] for m in meta])
    modes = np.array([m[2] for m in meta])

    queries = [q for q, g in annotations.items() if TASK_KEY in g]
    print(f"Queries with '{TASK_KEY}' positives: {len(queries)}")

    if result_file.endswith(".npy"):
        S = np.load(result_file)
        all_q = list(annotations.keys())
        assert S.shape == (len(all_q), len(dataset)), \
            f"matrix {S.shape} != (annotation queries {len(all_q)}, clips {len(dataset)})"
        row_of = {q: r for r, q in enumerate(all_q)}
        get_scores = lambda q: S[row_of[q]].astype(np.float64)
    else:
        with open(result_file) as f:
            results = json.load(f)

        def get_scores(q):
            s = np.full(len(dataset), -np.inf)             # missing clips -> ranked last
            for v, sim in results.get(q, {}).items():
                if v in col:
                    s[col[v]] = sim
            return s

    rows, skipped, missing_scores = [], defaultdict(int), 0
    for q in queries:
        if q not in col:
            raise KeyError(f"query {q} not in {dataset_ids_file}")
        qc, qp, qm = parse(q)
        keep = np.ones(len(dataset), dtype=bool)
        keep[col[q]] = False
        if EXCLUDE_SAME_CASE:
            keep &= cases != qc
        if GALLERY == "same_mode":
            keep &= modes == gallery_mode(qm)
        idx = np.flatnonzero(keep)

        pos = {col[v] for v in annotations[q][TASK_KEY] if v in col}
        is_pos = np.array([i in pos for i in idx])
        n_pos = int(is_pos.sum())
        if n_pos == 0:
            skipped[qm] += 1
            continue
        scores = get_scores(q)[idx]
        missing_scores += int(np.isneginf(scores).sum())
        rows.append({"query": q, "case": qc, "phase": qp, "mode": qm,
                     "gallery_size": len(idx), "n_positives": n_pos,
                     "AP": average_precision(scores, is_pos, n_pos)})

    if missing_scores:
        print(f"! {missing_scores} gallery entries had no similarity in the results file "
              f"(ranked last). Recompute similarities without the 'sims > 0' filter.")

    def mAP(sel):
        return float(np.mean([r["AP"] for r in sel])) if sel else float("nan")

    def class_avg(sel):
        by_phase = defaultdict(list)
        for r in sel:
            by_phase[r["phase"]].append(r["AP"])
        return float(np.mean([np.mean(v) for v in by_phase.values()])) if by_phase else float("nan")

    summary = []
    print(f"\n{TASK_NAME} | gallery: {GALLERY}, same-case excluded: {EXCLUDE_SAME_CASE}, "
          f"speed gallery: {SPEED_GALLERY}")
    print(f"{'mode':10s} {'queries':>8s} {'skipped':>8s} {'mAP':>7s} {'class-avg':>10s}")
    for m in MODES + ["All"]:
        sel = rows if m == "All" else [r for r in rows if r["mode"] == m]
        sk = sum(skipped.values()) if m == "All" else skipped[m]
        print(f"{m:10s} {len(sel):8d} {sk:8d} {mAP(sel):7.4f} {class_avg(sel):10.4f}")
        summary.append({"group": "mode", "name": m, "queries": len(sel), "skipped": sk,
                        "mAP": mAP(sel), "class_avg_mAP": class_avg(sel)})

    print(f"\n{'phase':22s} " + " ".join(f"{m:>9s}" for m in MODES + ["All"]))
    for p in sorted({r["phase"] for r in rows}):
        vals = [mAP([r for r in rows if r["phase"] == p and (m == "All" or r["mode"] == m)])
                for m in MODES + ["All"]]
        print(f"{p:22s} " + " ".join(f"{v:9.4f}" for v in vals))
        for m, v in zip(MODES + ["All"], vals):
            summary.append({"group": f"phase:{p}", "name": m,
                            "queries": len([r for r in rows if r["phase"] == p and (m == "All" or r["mode"] == m)]),
                            "skipped": "", "mAP": v, "class_avg_mAP": ""})

    base = os.path.join(saveDir, os.path.splitext(os.path.basename(result_file))[0])
    with open(base + "_per_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(base + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved {base}_per_query.csv and {base}_summary.csv")


if __name__ == "__main__":
    main()
