from feature_checkpoint import FeatureCheckpoint
"""
VideoMAE V2 clip features for SurgCBVR-CAT (mean and max pooling in one run).

Model
  ViT-B/16 VideoMAE V2, distilled from ViT-g and trained on Kinetics-710
  (official checkpoint vit_b_k710_dl_from_giant.pth), built with the official
  VideoMAEv2 code. Frozen; the classification head is not used.

  Setup (once):
    git clone https://github.com/OpenGVLab/VideoMAEv2.git
    download https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth
    set VIDEOMAEV2_REPO and CKPT_PATH below.

Input (matches the official fine-tuning/test pipeline)
  16 frames, 224x224, RGB in [0,1], ImageNet mean/std.
  Official sampling: every 4th frame at ~30 fps (16 frames = ~2.1 s). Our clips
  are 59.94 fps, so FRAME_STRIDE = 8 keeps the same temporal span and motion speed.

Clip coverage
  A window = 16 frames taken every FRAME_STRIDE frames (128 source frames, ~2.1 s).
  N = ceil(n_frames / 128) windows are spread evenly from the first to the last frame
  of the clip (capped at MAX_WINDOWS, still spread over the whole clip), so the whole
  clip is always covered: 5 s -> 3 windows, 10 s -> 5, 24 s -> 12.

Pooling (over all 8x14x14 = 1568 tokens of every window, last transformer block)
  mean: fc_norm(mean of tokens)  -- exactly the model's own pooled feature,
        extended to several windows
  max : fc_norm(max of tokens)
  Both 768-d, then L2-normalised. One .npy per pooling, rows in the order of
  VIDEO_LIST_TXT (identical to the DINOv2 script), plus a _config.json.
"""
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# =======================
# CONFIG (edit these)
# =======================
VIDEO_LIST_TXT = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.txt"
VideoDIR = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1"

VIDEOMAEV2_REPO = "./VideoMAEv2-master"                       # clone of the official repo
CKPT_PATH = "./VideoMAEv2-master/vit_b_k710_dl_from_giant.pth"
MODEL_NAME = "vit_base_patch16_224"

OUTPUT_DIR = Path("/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Features/videoMAE2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


NUM_FRAMES = 16          # frames per window (model input)
FRAME_STRIDE = 8         # take every 8th frame (59.94 fps) == official every 4th at ~30 fps
MAX_WINDOWS = 16         # cap per clip; windows stay spread over the whole clip
BATCH_WINDOWS = 8        # windows per forward pass (speed only)
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PREFIX = os.path.join(OUTPUT_DIR, "video_features_videomaev2_vitb_k710")   # -> <prefix>_mean.npy / _max.npy
# =======================


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
    """Return RGB frames (H,W,3 uint8) for the given indices (sorted or not, repeats allowed)."""
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
    """List of windows, each a list of NUM_FRAMES frame indices, spread over the whole clip."""
    span = (NUM_FRAMES - 1) * FRAME_STRIDE + 1
    n_win = min(MAX_WINDOWS, max(1, math.ceil(n_frames / (NUM_FRAMES * FRAME_STRIDE))))
    if n_frames <= span:
        starts = [0]
        stride = max(1, (n_frames - 1) // (NUM_FRAMES - 1)) if n_frames > 1 else 1
    else:
        starts = np.linspace(0, n_frames - span, n_win).round().astype(int).tolist()
        stride = FRAME_STRIDE
    return [[min(s + t * stride, n_frames - 1) for t in range(NUM_FRAMES)] for s in starts]


def to_tensor(frames):
    """16 RGB frames -> (3, 16, 224, 224) normalised tensor."""
    x = torch.from_numpy(np.stack(frames)).float().div_(255.0)          # T,H,W,C
    x = x.permute(0, 3, 1, 2)                                          # T,C,H,W
    if x.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.permute(1, 0, 2, 3)                                       # C,T,H,W


def build_model():
    sys.path.insert(0, os.path.abspath(VIDEOMAEV2_REPO))
    # Call the official constructor directly (timm >= 0.9's create_model passes extra
    # arguments that the official VideoMAE V2 code, written for timm 0.4, does not accept).
    from models import modeling_finetune
    model = getattr(modeling_finetune, MODEL_NAME)(
        pretrained=False, img_size=IMG_SIZE, num_classes=710,
        all_frames=NUM_FRAMES, tubelet_size=2, use_mean_pooling=True)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    for key in ("model", "module"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    missing = [k for k in missing if not k.startswith("head")]
    if missing:
        raise RuntimeError(f"Checkpoint is missing backbone weights: {missing[:10]}")
    print(f"Loaded {CKPT_PATH} (unexpected keys ignored: {len(unexpected)})")
    return model.to(DEVICE).eval()


@torch.no_grad()
def tokens(model, x):
    """Same steps as VisionTransformer.forward_features, but returns all tokens (B, N, 768)."""
    B = x.size(0)
    x = model.patch_embed(x)
    if model.pos_embed is not None:
        x = x + model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
    x = model.pos_drop(x)
    for blk in model.blocks:
        x = blk(x)
    return model.norm(x)          # Identity when use_mean_pooling=True


def main():
    print("Device:", DEVICE, flush=True)
    model = build_model()
    paths = load_video_list(VIDEO_LIST_TXT, VideoDIR)
    D = model.embed_dim
    ck = FeatureCheckpoint({pool: f"{OUT_PREFIX}_{pool}.npy" for pool in ("mean", "max")}, paths, D,
        {"model": "videomaev2", "model_name": MODEL_NAME, "stride": FRAME_STRIDE,
         "max_windows": MAX_WINDOWS, "img_size": IMG_SIZE, "checkpoint": str(Path(CKPT_PATH).resolve())})
    n_windows = ck.extras()

    for i, vp in enumerate(tqdm(paths)):
        if i in ck.done:
            continue
        n = count_frames(vp)
        wins = window_indices(n)

        frames = read_frames(vp, [j for w in wins for j in w])
        clips = [to_tensor(frames[k * NUM_FRAMES:(k + 1) * NUM_FRAMES]) for k in range(len(wins))]

        sum_tok, max_tok, count = None, None, 0
        for b in range(0, len(clips), BATCH_WINDOWS):
            t = tokens(model, torch.stack(clips[b:b + BATCH_WINDOWS]).to(DEVICE))   # (b, 1568, D)
            s = t.sum(dim=(0, 1))
            m = t.amax(dim=(0, 1))
            sum_tok = s if sum_tok is None else sum_tok + s
            max_tok = m if max_tok is None else torch.maximum(max_tok, m)
            count += t.shape[0] * t.shape[1]

        with torch.no_grad():
            v_mean = model.fc_norm(sum_tok / count)
            v_max = model.fc_norm(max_tok)
        ck.save(i, {"mean": F.normalize(v_mean, dim=0).cpu().numpy(),
                    "max": F.normalize(v_max, dim=0).cpu().numpy()}, extra=len(wins))
        n_windows.append(len(wins))

    if len(ck.done) != len(paths):
        raise RuntimeError("Incomplete extraction")
    n_windows = ck.extras()
    ck.close()
    with open(f"{OUT_PREFIX}_config.json", "w") as fh:
        json.dump({"model": "VideoMAE V2 ViT-B/16 distilled from ViT-g, Kinetics-710 "
                            "(vit_b_k710_dl_from_giant.pth)", "frozen": True,
                   "num_frames": NUM_FRAMES, "frame_stride": FRAME_STRIDE,
                   "window_span_frames": (NUM_FRAMES - 1) * FRAME_STRIDE + 1,
                   "windows": "ceil(n_frames/128), evenly spread over clip",
                   "max_windows": MAX_WINDOWS, "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
                   "tokens": "last block, 8x14x14 per window",
                   "pooling": {"mean": "fc_norm(mean over tokens of all windows)",
                               "max": "fc_norm(max over tokens of all windows)"},
                   "l2_normalized": True, "dim": D, "n_videos": len(paths),
                   "windows_per_clip": {"min": min(n_windows), "max": max(n_windows),
                                        "mean": float(np.mean(n_windows))}}, fh, indent=2)


if __name__ == "__main__":
    main()
