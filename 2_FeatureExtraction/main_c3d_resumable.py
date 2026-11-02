from feature_checkpoint import FeatureCheckpoint
"""
C3D clip features for SurgCBVR-CAT (fc7, mean and max pooling over windows in one run).

Model
  C3D (Tran et al., ICCV 2015) pretrained on Sports-1M, PyTorch port by
  D. Abati (github.com/DavideA/c3d-pytorch). Frozen; fc8 is not used.
  Weights: download c3d.pickle (link in that repository's README) and set CKPT_PATH.
  The layer names below match that port, so the file loads directly.

Input (as in the port's demo, predict.py)
  16 frames, 112x112, RGB, raw values 0-255 (no mean subtraction), layout (B,3,16,112,112).
  Our clips are square (224x224): resized to 128x128 and centre-cropped to 112x112
  (C3D's original 128-px resize + 112 crop geometry, without changing aspect ratio).
  C3D was trained on consecutive frames at ~25-30 fps; our clips are 59.94 fps, so
  FRAME_STRIDE = 2 keeps the same motion speed (16 frames = ~0.5 s).

Clip coverage
  Non-overlapping windows of 16 frames x stride 2 (32 source frames) tile the clip:
  N = floor(n_frames / 32) windows, evenly spread; capped at MAX_WINDOWS, still spread
  over the whole clip. 5 s (300 frames) -> 9 windows, 10 s -> 18, longer -> 32.

Feature
  fc7 after ReLU (4096-d) per window (dropout off in eval mode), pooled over windows:
  mean (the standard C3D video descriptor) and max. L2-normalised.
  Set LAYER = "fc6" to use fc6 instead (used in the C3D paper for retrieval-type tasks).
  One .npy per pooling, rows in the order of VIDEO_LIST_TXT (identical to the DINOv2
  script), plus a _config.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

# =======================
# CONFIG (edit these)
# =======================
VIDEO_LIST_TXT = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.txt"
VideoDIR = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1"
CKPT_PATH = "./c3d-pytorch-master/c3d.pickle"

OUTPUT_DIR = Path("/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Features/C3D")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = "fc7"            # "fc7" (4096-d) or "fc6"
NUM_FRAMES = 16
FRAME_STRIDE = 2         # every 2nd frame at 59.94 fps ~ consecutive frames at ~30 fps
MAX_WINDOWS = 32
BATCH_WINDOWS = 32
RESIZE = 128
CROP = 112

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PREFIX = os.path.join(OUTPUT_DIR, f"video_features_c3d_{LAYER}")      # -> <prefix>_mean.npy / _max.npy
# =======================


class C3D(nn.Module):
    """C3D architecture with the parameter names of the Sports-1M PyTorch port."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d(2, 2)
        self.conv5a = nn.Conv3d(512, 512, 3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool3d(2, 2, padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)
        self.relu = nn.ReLU()

    def features(self, x, layer="fc7"):
        r = self.relu
        h = self.pool1(r(self.conv1(x)))
        h = self.pool2(r(self.conv2(h)))
        h = self.pool3(r(self.conv3b(r(self.conv3a(h)))))
        h = self.pool4(r(self.conv4b(r(self.conv4a(h)))))
        h = self.pool5(r(self.conv5b(r(self.conv5a(h)))))
        h = r(self.fc6(h.reshape(h.size(0), 8192)))
        if layer == "fc6":
            return h
        return r(self.fc7(h))      # dropout is a no-op at inference


def load_video_list(txt_path: str, search_directory: str):
    """Same strict loader as the DINOv2 script: same rows, same order, fail if missing."""
    file_map = {p.name: str(p.absolute()) for p in Path(search_directory).rglob('*') if p.is_file()}
    paths = []
    with open(txt_path) as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if name not in file_map:
                raise FileNotFoundError(f"'{name}' listed in {txt_path} not found under {search_directory}")
            paths.append(file_map[name])
    return paths


def count_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_frames(path, indices):
    wanted = sorted(set(indices))
    got, cap, i, k = {}, cv2.VideoCapture(path), 0, 0
    while k < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i == wanted[k]:
            got[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            k += 1
        i += 1
    cap.release()
    last = got[max(got)]
    return [got.get(j, last) for j in indices]


def window_indices(n_frames):
    span = NUM_FRAMES * FRAME_STRIDE
    n_win = min(MAX_WINDOWS, max(1, n_frames // span))
    last_start = max(0, n_frames - span)
    starts = np.linspace(0, last_start, n_win).round().astype(int).tolist()
    return [[min(s + t * FRAME_STRIDE, n_frames - 1) for t in range(NUM_FRAMES)] for s in starts]


def to_tensor(frames):
    """16 RGB frames -> (3, 16, 112, 112), values 0-255."""
    off = (RESIZE - CROP) // 2
    clip = [cv2.resize(f, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)[off:off + CROP, off:off + CROP]
            for f in frames]
    x = torch.from_numpy(np.stack(clip)).float()          # T,H,W,C
    return x.permute(3, 0, 1, 2)                          # C,T,H,W


def main():
    print("Device:", DEVICE, flush=True)
    model = C3D()
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model = model.to(DEVICE).eval()

    paths = load_video_list(VIDEO_LIST_TXT, VideoDIR)
    D = 4096
    ck = FeatureCheckpoint({pool: f"{OUT_PREFIX}_{pool}.npy" for pool in ("mean", "max")}, paths, D,
        {"model": "c3d", "layer": LAYER, "stride": FRAME_STRIDE, "max_windows": MAX_WINDOWS,
         "resize": RESIZE, "crop": CROP, "checkpoint": str(Path(CKPT_PATH).resolve())})
    n_windows = ck.extras()

    with torch.no_grad():
        for i, vp in enumerate(tqdm(paths)):
            if i in ck.done:
                continue
            wins = window_indices(count_frames(vp))

            frames = read_frames(vp, [j for w in wins for j in w])
            clips = [to_tensor(frames[k * NUM_FRAMES:(k + 1) * NUM_FRAMES]) for k in range(len(wins))]
            feats = torch.cat([model.features(torch.stack(clips[b:b + BATCH_WINDOWS]).to(DEVICE), LAYER)
                               for b in range(0, len(clips), BATCH_WINDOWS)])          # (n_win, 4096)
            ck.save(i, {"mean": F.normalize(feats.mean(0), dim=0).cpu().numpy(),
                        "max": F.normalize(feats.amax(0), dim=0).cpu().numpy()}, extra=len(wins))
            n_windows.append(len(wins))

    if len(ck.done) != len(paths):
        raise RuntimeError("Incomplete extraction")
    n_windows = ck.extras()
    ck.close()
    with open(f"{OUT_PREFIX}_config.json", "w") as fh:
        json.dump({"model": "C3D, Sports-1M (PyTorch port, c3d.pickle)", "frozen": True,
                   "layer": f"{LAYER} after ReLU", "num_frames": NUM_FRAMES,
                   "frame_stride": FRAME_STRIDE, "windows": "non-overlapping, floor(n_frames/32), "
                   "evenly spread over clip", "max_windows": MAX_WINDOWS,
                   "preprocessing": f"resize {RESIZE}x{RESIZE}, centre crop {CROP}, RGB 0-255, "
                                    "no mean subtraction (as in the port's demo)",
                   "pooling": {"mean": "mean over windows", "max": "max over windows"},
                   "l2_normalized": True, "dim": D, "n_videos": len(paths),
                   "windows_per_clip": {"min": min(n_windows), "max": max(n_windows),
                                        "mean": float(np.mean(n_windows))}}, fh, indent=2)


if __name__ == "__main__":
    main()
