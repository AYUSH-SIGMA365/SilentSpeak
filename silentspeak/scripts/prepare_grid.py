import sys
import os
import glob
from pathlib import Path
import torchvision
import torch
import av
import numpy as np
import cv2
from tqdm import tqdm
import csv

# Add SilentSpeak root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from silentspeak.config import settings

# Dynamically add auto_avsr to sys.path
if settings.AUTO_AVSR_PATH not in sys.path:
    sys.path.insert(0, settings.AUTO_AVSR_PATH)

from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess
from datamodule.transforms import TextTransform

def read_grid_alignment(path: Path) -> str:
    """Reads a GRID .align file and returns uppercase words excluding silences."""
    words = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word = parts[2]
                if word.lower() not in ["sil", "sp"]:
                    words.append(word)
    return " ".join(words).upper()

def load_video_compat(data_filename):
    container = av.open(data_filename)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

def prepare_grid(subset_limit=None):
    base_data_dir = Path(r"C:\Users\Admin\Desktop\dl innov\data\GRID")
    out_data_dir = settings.BASE_DIR / "data" / "GRID_cropped"
    out_csv_dir = settings.BASE_DIR / "data"
    
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Detectors and Tokenizer
    print("Initializing MediaPipe and SentencePiece...")
    landmarks_detector = LandmarksDetector()
    video_process = VideoProcess(convert_gray=False)
    text_transform = TextTransform()
    
    # 1. Collect all valid video/alignment pairs
    aligns = glob.glob(str(base_data_dir / "alignments" / "**" / "*.align"), recursive=True)
    
    valid_pairs = []
    for align_path_str in aligns:
        align_path = Path(align_path_str)
        speaker = align_path.parent.name.upper() # e.g. S1 or S10
        filename = align_path.stem
        
        # Try both direct and nested formats
        vid_path_1 = base_data_dir / speaker / f"{filename}.mpg"
        vid_path_2 = base_data_dir / speaker / speaker.lower() / f"{filename}.mpg"
        
        vid_path = vid_path_1 if vid_path_1.exists() else vid_path_2
        
        if vid_path.exists():
            valid_pairs.append((vid_path, align_path, speaker))

    print(f"Total valid pairs found: {len(valid_pairs)}")
    
    # Sort pairs to ensure deterministic splitting
    valid_pairs.sort(key=lambda x: str(x[0]))
    
    if subset_limit:
        valid_pairs = valid_pairs[:subset_limit]
        print(f"Limiting processing to {subset_limit} videos for testing.")

    # 2. Split logic (Train: S1, Val/Test: S10)
    splits = {"train": [], "val": [], "test": []}
    
    # Separate by speaker
    s1_pairs = [p for p in valid_pairs if p[2] == "S1"]
    s10_pairs = [p for p in valid_pairs if p[2] == "S10"]
    
    splits["train"] = s1_pairs
    if subset_limit:
        # If it's a subset test, just throw everything into train for simplicity
        splits["train"] = valid_pairs
        splits["val"] = []
        splits["test"] = []
    else:
        # 140 val, remaining test
        splits["val"] = s10_pairs[:140]
        splits["test"] = s10_pairs[140:]

    # 3. Process each split
    for split_name, pairs in splits.items():
        if not pairs:
            continue
            
        csv_path = out_csv_dir / f"{split_name}.csv"
        print(f"\nProcessing {split_name} split ({len(pairs)} items)...")
        
        manifest = []
        
        for vid_path, align_path, speaker in tqdm(pairs, desc=split_name):
            try:
                # Target path
                rel_path = f"{speaker}/{vid_path.stem}.mp4"
                target_vid_path = out_data_dir / rel_path
                target_vid_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Transcript -> Tokens
                transcript = read_grid_alignment(align_path)
                if not transcript:
                    continue # Skip empty
                    
                token_ids_tensor = text_transform.tokenize(transcript)
                token_ids_str = " ".join(map(str, token_ids_tensor.tolist()))
                
                # Crop Video
                if not target_vid_path.exists():
                    video = load_video_compat(str(vid_path))
                    landmarks = landmarks_detector(video)
                    video = video_process(video, landmarks)
                    # Output shape from video_process is (T, H, W, C) numpy array uint8
                    
                    # Write to mp4 using cv2
                    T, H, W, C = video.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(target_vid_path), fourcc, 25.0, (W, H))
                    for i in range(T):
                        frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
                        out.write(frame)
                    out.release()
                else:
                    # Already processed, just read length
                    video, _, _ = torchvision.io.read_video(str(target_vid_path), pts_unit="sec", output_format="THWC")
                    video = video.numpy()
                
                num_frames = video.shape[0]
                
                # Format: dataset_name, rel_path, input_length, token_id
                manifest.append([
                    "GRID",
                    rel_path,
                    str(num_frames),
                    token_ids_str
                ])
                
            except Exception as e:
                print(f"Error processing {vid_path.name}: {e}")
                
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(manifest)
            
        print(f"Saved manifest to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Process a small subset of N videos")
    args = parser.parse_args()
    
    prepare_grid(args.subset)
