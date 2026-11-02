"""
Query-gallery similarities for SurgCBVR-CAT (FIVR-style results JSON).

Output: {query_clip: {gallery_clip: similarity, ...}, ...}, the same format as
FIVR's calculate_similarities.py, so the existing evaluation script reads it.

Changes from the FIVR version:
  * No `sims > 0` filter. FIVR drops gallery clips with similarity <= 0; for
    features with negative values (DINOv2, VideoMAE) that silently removes some
    positives from the ranking, which lowers AP. Every gallery clip is kept.
  * Same-case exclusion. Clips from the query's own surgery are removed from its
    ranking (they overlap in time with the query and would rank first as false
    negatives). The case ID is the first token of the clip name (e.g. "0012").
  * Optional same-mode gallery (SAME_MODE_GALLERY), for one result per clip mode.
  * Checks: unique clip names, no NaN rows, all queries present.

Important: One decision for you: SAME_MODE_GALLERY (default False, as in your original). keep it false in this script.
With False, a 5 s query searches all clips: 5 s, 10 s, proportional and speed variants from other surgeries.
For the per-mode table in the Data in Brief manuscript (one column each for 5 s / 10 s / Prop. / 2x / 3x / 5x), set it to True, so each query searches only clips of its own mode.

"""
import json
import os
import re

import numpy as np

# ========= Main Mode =========

EXCLUDE_SAME_CASE = True     # remove the query's own surgery from its ranking
SAME_MODE_GALLERY = False   # keep it False for generalization. in the next script, we can also make the filter # True: gallery = clips of the query's mode only (fixed5, fixed10,
                             # prop, speedN); False: all clips (FIVR-style single gallery)
SAVE_MATRIX = False          # also save the full query x gallery matrix as .npy

feature_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Features/videoMAE2/video_features_videomaev2_vitb_k710_mean.npy"
annotations_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.json"
dataset_ids_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.txt"

result_file = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Results/SimilarityScores/video_features_videomaev2_vitb_k710_mean.json"

# ==============================


def case_of(name):
    return os.path.basename(name).split("_")[0]


def mode_of(name):
    m = re.search(r"_(fixed\d+s|prop\d+s|speed\d+)_", name)
    if not m:
        return "unknown"
    return "prop" if m.group(1).startswith("prop") else m.group(1)


def main():
    print("Loading features from:", feature_file)
    features = np.load(feature_file).astype(np.float32)
    if np.isnan(features).any():
        raise ValueError("Feature matrix contains NaN rows: extraction not finished?")

    with open(dataset_ids_file) as f:
        dataset = [line.strip() for line in f if line.strip()]
    with open(annotations_file) as f:
        query_ids = list(json.load(f).keys())

    assert features.shape[0] == len(dataset), \
        f"{features.shape[0]} feature rows but {len(dataset)} clips in {dataset_ids_file}"
    assert len(set(dataset)) == len(dataset), "Duplicate clip names in the dataset list"
    id_to_index = {vid: i for i, vid in enumerate(dataset)}
    missing = [q for q in query_ids if q not in id_to_index]
    assert not missing, f"{len(missing)} queries not in the dataset list, e.g. {missing[:3]}"
    print("Feature shape:", features.shape, "| queries:", len(query_ids))

    # Cosine similarity (rows are re-normalised so this holds for any feature file).
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    q_idx = np.array([id_to_index[q] for q in query_ids])
    similarities = features[q_idx] @ features.T                      # (Q, N)

    cases = np.array([case_of(v) for v in dataset])
    modes = np.array([mode_of(v) for v in dataset])
    results, n_gallery = {}, []
    for j, query in enumerate(query_ids):
        keep = np.ones(len(dataset), dtype=bool)
        keep[q_idx[j]] = False                                       # the query itself
        if EXCLUDE_SAME_CASE:
            keep &= cases != case_of(query)
        if SAME_MODE_GALLERY:
            keep &= modes == mode_of(query)
        idx = np.flatnonzero(keep)
        n_gallery.append(len(idx))
        results[query] = {dataset[i]: float(similarities[j, i]) for i in idx}

    print(f"Gallery size per query: min {min(n_gallery)}, max {max(n_gallery)}")
    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    print("Saving results to:", result_file)
    with open(result_file, "w") as f:
        json.dump(results, f)
    if SAVE_MATRIX:
        np.save(result_file.replace(".json", "_matrix.npy"), similarities)
    print("Done.")


if __name__ == "__main__":
    main()
