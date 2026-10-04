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

def get_grid_transcript(vid_path: Path, base_data_dir: Path) -> str:
    """Gets transcript from .align file or generates it from filename."""
    speaker = vid_path.parent.name
    if speaker.islower():
        speaker = vid_path.parent.parent.name
        
    align_path = base_data_dir / "alignments" / speaker.lower() / f"{vid_path.stem}.align"
    if align_path.exists():
        words = []
        with open(align_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    word = parts[2]
                    if word.lower() not in ["sil", "sp"]:
                        words.append(word)
        if words:
            return " ".join(words).upper()
            
    # Fallback to filename parsing
    cmd = {'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'}
    color = {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'}
    prep = {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'}
    digit = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', 'z': 'zero'}
    adv = {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}
    
    filename = vid_path.stem
    if len(filename) == 6:
        try:
            return f"{cmd[filename[0]]} {color[filename[1]]} {prep[filename[2]]} {filename[3]} {digit[filename[4]]} {adv[filename[5]]}".upper()
        except KeyError:
            pass
    return None

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
    
    # 1. Collect all valid video/transcript pairs
    vids = glob.glob(str(base_data_dir / "**" / "*.mpg"), recursive=True)
    
    valid_pairs = []
    for vid_path_str in vids:
        vid_path = Path(vid_path_str)
        speaker = vid_path.parent.name.upper()
        if not (speaker.startswith("S") and speaker[1:].isdigit()):
            speaker = vid_path.parent.parent.name.upper()
            
        # We only want S1 and S10 for our split
        if speaker not in ["S1", "S10"]:
            continue
            
        transcript = get_grid_transcript(vid_path, base_data_dir)
        if transcript:
            valid_pairs.append((vid_path, transcript, speaker))

    # Sort pairs to ensure deterministic splitting
    valid_pairs.sort(key=lambda x: str(x[0]))
    
    # Separate by speaker
    s1_pairs = [p for p in valid_pairs if p[2] == "S1"]
    s10_pairs = [p for p in valid_pairs if p[2] == "S10"]
    
    print(f"Discovered {len(s1_pairs)} S1 videos and {len(s10_pairs)} S10 videos with transcripts.")
    
    # 2. Split logic (Train: S1, Val/Test: S10)
    splits = {"train": [], "val": [], "test": []}
    
    if subset_limit:
        print(f"Limiting processing to {subset_limit} videos per speaker for testing.")
        splits["train"] = s1_pairs[:subset_limit]
        splits["val"] = s10_pairs[:subset_limit]
        splits["test"] = []
    else:
        splits["train"] = s1_pairs
        # 140 val, remaining test (up to exactly 281 for S10 as originally specified)
        s10_pairs = s10_pairs[:281]
        splits["val"] = s10_pairs[:140]
        splits["test"] = s10_pairs[140:]

    total_to_process = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    print(f"Total videos selected for split: {total_to_process} (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})")

    # 3. Process each split
    for split_name, pairs in splits.items():
        if not pairs:
            continue
            
        csv_path = out_csv_dir / f"{split_name}.csv"
        print(f"\nProcessing {split_name} split ({len(pairs)} items)...")
        
        manifest = []
        
        for vid_path, transcript, speaker in tqdm(pairs, desc=split_name):
            try:
                # Target path
                rel_path = f"{speaker}/{vid_path.stem}.mp4"
                target_vid_path = out_data_dir / rel_path
                target_vid_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Transcript -> Tokens
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
                    container = av.open(str(target_vid_path))
                    num_frames = container.streams.video[0].frames
                    if num_frames == 0:
                        num_frames = sum(1 for _ in container.decode(video=0))
                
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
