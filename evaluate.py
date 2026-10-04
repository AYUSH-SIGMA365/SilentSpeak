import os
from pathlib import Path
from jiwer import wer, cer
from silentspeak.core.inference import LipReader

def read_grid_alignment(path: Path) -> str:
    """Reads a GRID .align file and returns the ground truth words (excluding silences)."""
    words = []
    if not path.exists():
        return ""
        
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word = parts[2]
                if word.lower() != "sil" and word.lower() != "sp":
                    words.append(word)
    
    return " ".join(words).upper()

def evaluate_dataset():
    # Base dataset directories
    base_data_dir = Path(r"C:\Users\Admin\Desktop\dl innov\data\GRID")
    align_dir = base_data_dir / "alignments"
    
    # Initialize the lip reader
    print("Initializing LipReader for evaluation...")
    lip_reader = LipReader()
    
    results = []
    total_wer = 0.0
    total_cer = 0.0
    count = 0
    
    # Evaluate over S1 and S10 as an example
    for speaker in ["S1", "S10"]:
        video_dir = base_data_dir / speaker
        speaker_align_dir = align_dir / speaker.lower()
        
        if not video_dir.exists():
            print(f"Skipping {speaker}, directory not found.")
            continue
            
        # Get up to 10 videos per speaker for a quick eval
        videos = list(video_dir.glob("*.mpg"))[:10]
        
        for video_path in videos:
            align_path = speaker_align_dir / f"{video_path.stem}.align"
            
            ground_truth = read_grid_alignment(align_path)
            if not ground_truth:
                print(f"Skipping {video_path.name}, no alignment found.")
                continue
                
            print(f"\nProcessing: {video_path.name}")
            try:
                prediction = lip_reader.transcribe(str(video_path))
                
                sample_wer = wer(ground_truth, prediction)
                sample_cer = cer(ground_truth, prediction)
                
                print(f"Ground Truth: {ground_truth}")
                print(f"Prediction  : {prediction}")
                print(f"WER         : {sample_wer:.4f}")
                print(f"CER         : {sample_cer:.4f}")
                
                results.append({
                    "video": video_path.name,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "WER": sample_wer,
                    "CER": sample_cer
                })
                
                total_wer += sample_wer
                total_cer += sample_cer
                count += 1
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                
    if count > 0:
        avg_wer = total_wer / count
        avg_cer = total_cer / count
        print("\n" + "="*40)
        print("EVALUATION COMPLETE")
        print(f"Total Videos Evaluated: {count}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")
        print("="*40)

if __name__ == "__main__":
    evaluate_dataset()
