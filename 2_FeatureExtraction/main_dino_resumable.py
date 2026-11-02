from feature_checkpoint import FeatureCheckpoint
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.data import resolve_model_data_config, create_transform
from PIL import Image




# =======================
# CONFIG (edit these)
# =======================

# ========= Dev Mode =========
# VIDEO_LIST_TXT = "../../1_DatasetCreation_Cataract1K/annotations_CBVR_CATv1.txt"     # each line: path/to/video.mp4
# VideoDIR = "../../1_DatasetCreation_Cataract1K/output"                      # directory to search for videos (used for faster loading)
# ==================

# ========= Main Mode Mode =========
VIDEO_LIST_TXT = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/annotations_CBVR_CAT.txt"     # each line: path/to/video.mp4
VideoDIR = "/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1"                      # directory to search for videos (used for faster loading)
# ==================

OUTPUT_DIR = Path("/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1/Features/DinoV2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("===========> Using device:", DEVICE, flush=True)

# Frame sampling options (choose one mode)
SKIP_MODE = "every_k"                 # "none" | "every_k" | "every_seconds"
EVERY_K = 5                           # if SKIP_MODE="every_k"
EVERY_SECONDS = 0.5                   # if SKIP_MODE="every_seconds"

MAX_FRAMES = 150                      # max sampled frames per video (after skipping).
                                      # v3: if a clip has more candidates, MAX_FRAMES are picked
                                      # UNIFORMLY over the whole clip (v2 kept only the first
                                      # 150, i.e. the first 750 frames = 12.5 s at 60 fps).
BATCH_SIZE = 64                       # frames per forward pass (speed only; same features)
RESIZE = (224, 224)

# Aggregation options:
# 1) "mean"  (mean pooling)
# 2) "gap"   (same as mean for (T,D), kept for naming consistency)
# 3) "max"   (max pooling over time)
AGGREGATION = "mean"
print(f"=====> Aggregation method: '{AGGREGATION}'", flush=True)


# DINOv2 model name
DINO_NAME = 'vit_base_patch14_dinov2.lvd142m'   # ViT-B/14 -> D=768 (CLS token after final LayerNorm)

# vit_small_patch14_dinov2
# vit_base_patch14_dinov2
# vit_large_patch14_dinov2
# vit_giant_patch14_dinov2

OUT_NPY = os.path.join(OUTPUT_DIR, f"video_features_{DINO_NAME.split('.')[0]}_{AGGREGATION}_{SKIP_MODE}_{EVERY_K}.npy")  # same name as v2  # output filename reflects settings
# =======================

# -----------------------
# Helpers
# -----------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

from pathlib import Path

def setup_model(model_name='vit_base_patch14_dinov2', device='cuda'):
    # Create model without the classification head (num_classes=0)
    # This automatically returns the pooled feature vector
    img_size = 224
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
    model.to(device)
    model.eval()
    
    # Get the specific configuration for this model (resolution, mean, std)
    config = resolve_model_data_config(model)
    config['input_size'] = (3, img_size, img_size)
    transform = create_transform(**config, is_training=False)
    
    print(f"Model input size default: {model.default_cfg['input_size']}")
    print("patch_embed img_size:", model.patch_embed.img_size)
    print("num_patches:", model.patch_embed.num_patches)
    print(f"Transform resize size: {config['input_size']}")

    return model, transform


# def load_video_list(txt_path: str, search_directory: str):
#     dir_path = Path(search_directory)
    
#     # This specifically looks for files only, across all subfolders
#     file_map = {p.name: str(p.absolute()) for p in dir_path.rglob('*') if p.is_file()}
    
#     paths = []
#     with open(txt_path, "r") as f:
#         for line in f:
#             filename = line.strip()
#             if filename in file_map:
#                 paths.append(file_map[filename])
#             elif filename:
#                 print(f"Skipping: {filename} (Not found in any subfolder)")
                
#     return paths

def load_video_list(txt_path: str, search_directory: str):
    dir_path = Path(search_directory)
    
    # 1. Map the 'stem' (filename without extension) to the full path
    # If your text file HAS extensions, use p.name instead of p.stem
    file_map = {p.name: str(p.absolute()) for p in dir_path.rglob('*') if p.is_file()}
    
    video_paths = []
    with open(txt_path, "r") as f:
        for line in f:
            clean_name = line.strip()
            if not clean_name:
                continue
            
            # 2. Strict Check
            if clean_name in file_map:
                video_paths.append(file_map[clean_name])
            else:
                # This raises an exception and STOPS the script immediately
                raise FileNotFoundError(
                    f"\n[CRITICAL ERROR] Video ID '{clean_name}' listed in {txt_path} "
                    f"was not found in directory: {search_directory}"
                )
                
    return video_paths

def preprocess_frame_rgb(frame_rgb, device):
    frame = cv2.resize(frame_rgb, RESIZE, interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0  # H,W,C in [0,1]
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)  # 1,3,224,224
    frame = frame.to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    frame = (frame - mean) / std
    return frame

def candidate_indices(n_frames: int, fps: float):
    """Frame indices selected by SKIP_MODE (before the MAX_FRAMES cap)."""
    if SKIP_MODE == "none":
        return list(range(n_frames))
    if SKIP_MODE == "every_k":
        return list(range(0, n_frames, max(1, int(EVERY_K))))
    if SKIP_MODE == "every_seconds":
        step = max(1e-6, float(EVERY_SECONDS)) * fps
        return sorted({int(round(i * step)) for i in range(int(n_frames / step) + 1)
                       if int(round(i * step)) < n_frames})
    raise ValueError(f"Unknown SKIP_MODE: {SKIP_MODE}")


def iter_sampled_frames(video_path: str):
    """Yield RGB frames chosen by SKIP_MODE; if more than MAX_FRAMES, pick MAX_FRAMES
    uniformly over the whole clip so the full clip is always represented."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    # Count frames by decoding (CAP_PROP_FRAME_COUNT can be inaccurate after re-encoding).
    n_frames = 0
    while cap.grab():
        n_frames += 1
    cap.release()
    if n_frames == 0:
        raise RuntimeError(f"No frames in video: {video_path}")

    cand = candidate_indices(n_frames, fps)
    if len(cand) > MAX_FRAMES:
        pick = np.linspace(0, len(cand) - 1, MAX_FRAMES).round().astype(int)
        cand = [cand[j] for j in pick]
    wanted = set(cand)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while frame_idx <= cand[-1]:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in wanted:
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_idx += 1
    cap.release()


# -----------------------
# Aggregators
# -----------------------

def aggregate_frames(frame_feats: torch.Tensor, AGGREGATION=None) -> torch.Tensor:
    """
    frame_feats: (T, D)
    returns: (D,) or (out_dim,) depending on aggregator
    """
    if frame_feats.numel() == 0:
        raise RuntimeError("No frame features to aggregate.")

    if AGGREGATION in ("mean", "gap"):
        v = frame_feats.mean(dim=0)
    elif AGGREGATION == "max":
        v, _ = frame_feats.max(dim=0)
    else:
        raise ValueError(f"Unknown AGGREGATION: {AGGREGATION}")

    # L2 normalize final vector (good for cosine similarity)
    v = F.normalize(v, p=2, dim=0)
    return v

from tqdm import tqdm
# -----------------------
# Main
# -----------------------
def main():


    dino, timm_transform = setup_model(model_name=DINO_NAME, device=DEVICE)

    video_paths = load_video_list(VIDEO_LIST_TXT, VideoDIR)
    if not video_paths:
        raise RuntimeError(f"No videos found in {VIDEO_LIST_TXT}")

    print(f"=====> Found {len(video_paths)} videos. \n Starting feature extraction with DINOv2 ({DINO_NAME}) and '{AGGREGATION}' aggregation, SKIP_MODE='{SKIP_MODE}'...", flush=True)
    
    # Load DINOv2
    # dino = torch.hub.load("facebookresearch/dinov2", DINO_NAME).to(DEVICE).eval()

    # Determine feature dim D using dummy input
    with torch.no_grad():
        dummy = torch.zeros(1, 3, RESIZE[1], RESIZE[0], device=DEVICE)
        out = dino(dummy)
        if isinstance(out, dict) and "x_norm_clstoken" in out:
            D = out["x_norm_clstoken"].shape[-1]
        else:
            D = out.shape[-1]

    # Setup aggregator if needed
    aggregator = AGGREGATION
    out_dim = D

    ck = FeatureCheckpoint({"features": OUT_NPY}, video_paths, out_dim,
        {"model": DINO_NAME, "skip_mode": SKIP_MODE, "every_k": EVERY_K,
         "every_seconds": EVERY_SECONDS, "max_frames": MAX_FRAMES,
         "aggregation": AGGREGATION, "resize": RESIZE})

    for i, vp in enumerate(video_paths):
        if i in ck.done:
            continue
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Missing video: {vp}")

        frame_vecs, batch = [], []
        with torch.no_grad():
            def flush():
                if batch:
                    x = torch.stack(batch, dim=0).to(DEVICE)   # (B, 3, 224, 224)
                    frame_vecs.extend(dino(x).cpu())          # B x (768,)
                    batch.clear()
            for frame_rgb in iter_sampled_frames(vp):
                # official DINOv2 preprocessing from timm (bicubic resize 224, center crop,
                # ImageNet mean/std) on a PIL image
                batch.append(timm_transform(Image.fromarray(frame_rgb)))
                if len(batch) == BATCH_SIZE:
                    flush()
            flush()

        if len(frame_vecs) == 0:
            raise RuntimeError(f"No frames sampled from: {vp}")

        frame_feats = torch.stack(frame_vecs, dim=0)  # (T, D)
        video_feat = aggregate_frames(frame_feats, AGGREGATION=aggregator)  # (out_dim,)
        ck.save(i, {"features": video_feat.detach().float().cpu().numpy()})

        if (i + 1) % 50 == 0 or (i + 1) == len(video_paths):
            print(f"===> [{i+1}/{len(video_paths)}] extracted: {vp}")

    if len(ck.done) != len(video_paths):
        raise RuntimeError("Incomplete extraction")
    ck.close()
    import json
    with open(OUT_NPY.replace(".npy", "_config.json"), "w") as fh:
        json.dump({"model": DINO_NAME, "feature": "CLS token after final LayerNorm",
                   "input_size": 224, "preprocessing": "timm eval transform: bicubic resize "
                   "(shorter side 224), center crop 224, ImageNet mean/std",
                   "skip_mode": SKIP_MODE, "every_k": EVERY_K, "every_seconds": EVERY_SECONDS,
                   "max_frames": MAX_FRAMES, "max_frames_selection": "uniform over clip",
                   "aggregation": AGGREGATION, "l2_normalized": True,
                   "n_videos": len(video_paths), "dim": int(out_dim)}, fh, indent=2)
    print("Saved:", OUT_NPY)
    print("Shape:", (len(video_paths), out_dim))  # (N, out_dim)
    print("Aggregation:", AGGREGATION, "| Skip mode:", SKIP_MODE)

if __name__ == "__main__":
    main()
