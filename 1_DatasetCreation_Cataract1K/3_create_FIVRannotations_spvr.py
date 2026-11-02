#!/usr/bin/env python3
"""
SPVR Dataset Creation Script
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

def create_FIVR_annotations_file(clips, OUTPUT_ANNOTATIONS_FILE):
    """
    Create annotation file in FIVR format with positive pairs only.
    
    FIVR Format: Each line contains a query clip followed by its positive matches.
    Format: query_clip_id positive_match_1 positive_match_2 ...
    
    Positive pairs: Same phase, different case_ids (patients/surgeries)
    """
    # Group clips by phase for easier matching
    phase_clips = defaultdict(list)
    for clip in clips:
        phase_clips[clip['phase']].append(clip)
    
    # Create positive pairs: same phase, different videos
    annotations = []
    processed_queries = set()
    
    for phase, phase_clip_list in phase_clips.items():
        # For each clip, find positive matches (same phase, different case_id)
        for query_clip in phase_clip_list:
            query_id = query_clip['clip_id']
            # Append .mp4 to the query_id if it's not already there
            
            query_id_ = query_id if query_id.endswith('.mp4') else f"{query_id}.mp4"

            # Find all clips from same phase but different case
            positive_matches = []
            for candidate_clip in phase_clip_list:
                if candidate_clip['case_id'] != query_clip['case_id'] and candidate_clip['clip_id'] != query_id:
                    # Append .mp4 to each match
                    match_id = candidate_clip['clip_id']
                    if not match_id.endswith('.mp4'):
                        match_id = f"{match_id}.mp4"
                    positive_matches.append(match_id)
            
            # Only add query if it has positive matches
            if positive_matches and query_id not in processed_queries:
                annotations.append({
                    'query': query_id_,
                    'positive': positive_matches
                })
                processed_queries.add(query_id)
    
    # Write annotations in FIVR format
    # Format: query_clip_id positive_match_1 positive_match_2 
    with open(OUTPUT_ANNOTATIONS_FILE, 'w') as f:
        for ann in annotations:
            # Write query followed by all positive matches
            line = f"{ann['query']}\n" #{' '.join(ann['positive'])}\n"
            #print(line)
            f.write(line)




    print(f"Created annotation file: {OUTPUT_ANNOTATIONS_FILE}")
    print(f"Total queries: {len(annotations)}")
    print(f"Total unique clips: {len(clips)}")
    
    # Also create JSON format for easier parsing
    json_file = OUTPUT_ANNOTATIONS_FILE.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Also created JSON annotations: {json_file}")
    
    return annotations


if __name__ == "__main__":   
    
   
    # Configuration

    # ========= Dev Mode =========
    Output_dir = './' 
    pklfile = "./extracted_clips_spvr.pkl"
    # ==================

    
    OUTPUT_ANNOTATIONS_FILE = os.path.join(Output_dir, "annotations_CBVR_CAT.txt")
    
    with open(pklfile, "rb") as f:
        all_clips = pickle.load(f)
    print(f"\n✓ Total clips extracted: {len(all_clips)}")

    
    print("\n" + "="*60)
    print("CREATING ANNOTATION FILE IN FIVR FORMAT")
    print("="*60)
    print("Creating annotations with positive pairs (same phase, different videos)...")
    annotations_data = create_FIVR_annotations_file(all_clips, OUTPUT_ANNOTATIONS_FILE)
    
    print("\n" + "="*60)
    print("✓ ND-DSVR DATASET CREATION COMPLETE!")
    print("="*60)
    #print(f"Output directory: {OUTPUT_CLIPS_DIR}")
    print(f"Annotation file: {OUTPUT_ANNOTATIONS_FILE}")
    print(f"JSON annotation file: {OUTPUT_ANNOTATIONS_FILE.replace('.txt', '.json')}")
    print(f"Total queries with positive matches: {len(annotations_data)}")
    if annotations_data:
        avg_positives = sum(len(a['positive']) for a in annotations_data) / len(annotations_data)
        print(f"Average positive matches per query: {avg_positives:.1f}")
    print("="*60)
