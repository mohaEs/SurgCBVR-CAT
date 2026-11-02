#!/usr/bin/env python3
"""
SPVR Dataset Creation Script
Extracts clips from same surgical phase but different videos/patients.
Creates annotation file in FIVR format with positive pairs only.
"""

import os
import csv
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import argparse
from tqdm import tqdm
import pickle


def get_video_fps(video_path):
    """Get FPS of a video file."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except:
        return 59.94  # Default FPS


def extract_save_clip(video_path, start_time, duration, output_path, speed=1):
    """Extract a frame-accurate clip (re-encodes, so length is exact)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",        # input seek: fast, and frame-accurate when re-encoding
        "-t", f"{speed * duration:.3f}",   # amount of source video to read
        "-i", video_path,
    ]
    if speed != 1:
        cmd += ["-vf", f"setpts=PTS/{speed}"]
    cmd += [
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip {output_path}: {e.stderr.decode(errors='ignore')[-500:]}")
        return False


def load_annotations(ANNOTATIONS_DIR, EXCLUDED_PHASES):
    """Load all phase annotations from CSV files."""
    annotations = []
    case_ids = []
    
    for case_dir in Path(ANNOTATIONS_DIR).iterdir():
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        
        case_id = case_dir.name.replace("case_", "")
        case_ids.append(case_id)
        
        phases_file = case_dir / f"case_{case_id}_annotations_phases.csv"
        if not phases_file.exists():
            continue
        
        with open(phases_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase_name = row['comment'].strip()
                if phase_name in EXCLUDED_PHASES:
                    continue
                
                annotations.append({
                    'case_id': case_id,
                    'phase': phase_name,
                    'start_sec': float(row['sec']),
                    'end_sec': float(row['endSec']),
                    'start_frame': int(row['frame']),
                    'end_frame': int(row['endFrame'])
                })
    
    return annotations, case_ids


def extract_phase_center_clips_withSpeed(annotations, CLIP_DURATION_target, OUTPUT_CLIPS_DIR, VIDEOS_DIR, speed = 3):
    # we can make larger set, but let's just have 0 offset for speed
    clips = []
    speed=speed
    
    
    for ann in tqdm(annotations):
        video_path = os.path.join(VIDEOS_DIR, f"case_{ann['case_id']}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        phase_duration = ann['end_sec'] - ann['start_sec']
        # print(phase_duration)                
        
       
        if phase_duration < speed*CLIP_DURATION_target:  # Need some buffer
            continue  # Skip phases shorter than clip duration
        
        # Calculate center time
        center_time = ann['start_sec'] + (phase_duration / 2.0)

        clip_start = center_time - ((speed/2)*CLIP_DURATION_target)
        
        # Ensure clip stays within phase boundaries with margin
        clip_start = max(ann['start_sec'], clip_start)
        clip_end = clip_start + speed*CLIP_DURATION_target
               
        phase_string = ann['phase']
        
        phase_string=phase_string.replace("/", "")
        phase_string=phase_string.replace(" ", "")
    
        
        clip_id = f"{ann['case_id']}_{phase_string}_speed{speed}_offset0_{int(clip_start)}"
        clip_path = os.path.join(OUTPUT_CLIPS_DIR, f"{clip_id}.mp4")
        flag_VideoExtractionResult = extract_save_clip(video_path, clip_start, CLIP_DURATION_target, clip_path, speed)
        if flag_VideoExtractionResult:
            clips.append({
                'clip_id': clip_id,
                'case_id': ann['case_id'],
                'phase': ann['phase'],
                'mode': f'speed{speed}',
                'speed': speed,
                'start_time': clip_start,
                'source_duration': speed * CLIP_DURATION_target,  # seconds of surgery covered
                'duration': CLIP_DURATION_target,                 # output length
                'phase_duration': phase_duration,
                'clip_path': clip_path
            })
            
    return clips
    


def extract_phase_center_clips(annotations, CLIP_DURATION_target, OUTPUT_CLIPS_DIR, VIDEOS_DIR):
    """
    Extract 5 clips near the center of each phase (CLIP_DURATION_target: e.g. 3-second clips).
    5 is based on offset
    Following SPVR specification: extract clips from the middle region
    to land in the "core execution" region of each phase.
    """
    clips = []
    CLIP_DURATION_target_input = CLIP_DURATION_target
    
    for ann in tqdm(annotations):
        video_path = os.path.join(VIDEOS_DIR, f"case_{ann['case_id']}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        phase_duration = ann['end_sec'] - ann['start_sec']
        # print(phase_duration)                
        
        if CLIP_DURATION_target_input=='Proportional':        
            CLIP_DURATION_target = int(phase_duration / 5)
            if CLIP_DURATION_target < 5:
                continue
        
        if phase_duration < CLIP_DURATION_target * 1.5:  # Need some buffer
            continue  # Skip phases shorter than clip duration
        
        # Calculate center time
        center_time = ann['start_sec'] + (phase_duration / 2.0)
        
        # Extract 5 clips near the middle of the phase, distributed symmetrically
        # This ensures we capture the "core execution" region
        #offsets = [-2*CLIP_DURATION_target, -CLIP_DURATION_target, 0, CLIP_DURATION_target, 2*CLIP_DURATION_target]
        offsets = [-2*phase_duration/10, -phase_duration/10, 0, phase_duration/10, 2*phase_duration/10]
        #print(offsets)
        
        for clip_idx, offset in enumerate(offsets):
            clip_start = center_time + offset - (CLIP_DURATION_target / 2.0)
            
            # Ensure clip stays within phase boundaries with margin
            clip_start = max(ann['start_sec'], clip_start)
            clip_end = clip_start + CLIP_DURATION_target
            
            if clip_end > ann['end_sec']:
                # Try to fit at the end of the phase
                clip_start = ann['end_sec'] - CLIP_DURATION_target
                if clip_start < ann['start_sec']:
                    continue  # Skip if phase is too short for this clip
            
            phase_string = ann['phase']
            
            phase_string=phase_string.replace("/", "")
            phase_string=phase_string.replace(" ", "")
            
            #####################
            # The mode tag ("fixed"/"prop") keeps IDs unique: without it, a proportional
            # clip with L = 5 or 10 s (phases of 25-30 s or 50-55 s) got the same ID and
            # file as the fixed-length clip, overwrote it, and was listed twice.
            mode_tag = "prop" if CLIP_DURATION_target_input == 'Proportional' else "fixed"
            clip_id = f"{ann['case_id']}_{phase_string}_{mode_tag}{int(CLIP_DURATION_target)}s_offset{round(offset)}_{int(clip_start)}"
            clip_path = os.path.join(OUTPUT_CLIPS_DIR, f"{clip_id}.mp4")
            # Extract clip
            flag_VideoExtractionResult = extract_save_clip(video_path, clip_start, CLIP_DURATION_target, clip_path)
            if flag_VideoExtractionResult:
                clips.append({
                    'clip_id': clip_id,
                    'case_id': ann['case_id'],
                    'phase': ann['phase'],
                    'mode': 'proportional' if mode_tag == 'prop' else f'fixed{int(CLIP_DURATION_target)}',
                    'speed': 1,
                    'offset': offset,
                    'start_time': clip_start,
                    'source_duration': CLIP_DURATION_target,
                    'duration': CLIP_DURATION_target,
                    'phase_duration': phase_duration,
                    'clip_path': clip_path
                })


    
    return clips




if __name__ == "__main__":
       
    # Configuration
    CLIP_DURATION_targets_seconds = [5, 10, 'Proportional']  # seconds for phase center clips
    CLIP_speed_targets = [2, 3, 5] 
    
    # ========= Dev Mode =========
    # Output_dir = 'output'
    # DATASET_ROOT = "./Phase_recognition_dataset_224_short"
    # ==================

    # ========= Main Mode =========
    DATASET_ROOT = "/shared/hdd_36T/meecs/datasets_2/surgery Cataract-1K/Phase_recognition_dataset_224"
    Output_dir = '/shared/hdd_36T/meecs/datasets_2/surgery CBVR_CAT/CBVR_CATv1'    
    # ==================
    
    # path = os.path.expanduser("~/my_folder")
    if os.path.exists(DATASET_ROOT):
        print("Path exists")
    ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "annotations")
    VIDEOS_DIR = os.path.join(DATASET_ROOT, "videos_224")
    
    OUTPUT_CLIPS_DIR = os.path.join(Output_dir, "clips_extracted_nd_dsvr")
    os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
    
    # Phases to exclude (Idle is not explicitly labeled, but we skip if found)
    EXCLUDED_PHASES = {"Idle", "idle"}
    
    print("Loading annotations from Cataract-1K dataset...")
    annotations, case_ids = load_annotations(ANNOTATIONS_DIR, EXCLUDED_PHASES)
    print(f"Loaded {len(annotations)} phase annotations from {len(case_ids)} cases")
    print(f"Excluded phases: {EXCLUDED_PHASES}")
    
    all_clips=[]
    for CLIP_DURATION_target in CLIP_DURATION_targets_seconds:
        print("\n" + "="*60)
        print(f"EXTRACTING CLIPS FROM PHASE CENTERS: length: {CLIP_DURATION_target} seconds")
        print("="*60)    
        clips = extract_phase_center_clips(annotations, CLIP_DURATION_target, 
                                              OUTPUT_CLIPS_DIR, VIDEOS_DIR)
        print(f"✓ Extracted {len(clips)} clips of duration {CLIP_DURATION_target} seconds")
        all_clips = all_clips + clips
        
    for speed in CLIP_speed_targets:
        print("\n" + "="*60)
        print(f"EXTRACTING CLIPS FROM PHASE CENTERS: output length: 5 seconds, but speeded {speed}x")
        print("="*60)    
        CLIP_DURATION_target = 5  # assuming 5 seconds fix
        clips = extract_phase_center_clips_withSpeed(annotations, CLIP_DURATION_target, 
                                              OUTPUT_CLIPS_DIR, VIDEOS_DIR, speed=speed)
        print(f"✓ Extracted {len(clips)} clips of duration {CLIP_DURATION_target} seconds and speed {speed}")
        all_clips = all_clips + clips
    

    print(f"\n✓ Total clips extracted: {len(all_clips)}")
    n_unique = len({c['clip_id'] for c in all_clips})
    assert n_unique == len(all_clips), f"{len(all_clips) - n_unique} duplicate clip IDs"
        
    with open(os.path.join(Output_dir, "extracted_clips_spvr.pkl"), "wb") as f:
        pickle.dump(all_clips, f)

    # 