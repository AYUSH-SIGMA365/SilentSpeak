import sys
import os
import argparse
import torch
import torchvision
import av
import numpy as np
from silentspeak.config import settings

# Dynamically add auto_avsr to sys.path so we can import its modules
if settings.AUTO_AVSR_PATH not in sys.path:
    sys.path.insert(0, settings.AUTO_AVSR_PATH)

# Now we can import Auto-AVSR modules
from lightning import ModelModule
from datamodule.transforms import VideoTransform

# We need to implement a fallback/compat load_video because sometimes read_video fails on some codecs
def load_video_compat(data_filename):
    container = av.open(data_filename)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

class LipReader:
    """
    Wrapper class to interface with Auto-AVSR inference pipeline.
    Implements a singleton pattern internally to avoid reloading models on every request.
    """
    _instance = None
    _model = None
    _video_transform = None
    _video_process = None
    _landmarks_detector = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LipReader, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = settings.VSR_MODEL_PATH, detector: str = "mediapipe"):
        # Prevent re-initialization if already loaded
        if self._model is not None:
            return
            
        print(f"Initializing LipReader with model: {model_path}")
        
        # Setup arguments required by Auto-AVSR
        parser = argparse.ArgumentParser()
        args, _ = parser.parse_known_args(args=[])
        setattr(args, "modality", "video")
        
        # Load transforms and detectors
        self._video_transform = VideoTransform(subset="test")
        
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess
            self._landmarks_detector = LandmarksDetector()
            self._video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess
            self._landmarks_detector = LandmarksDetector(device="cuda:0")
            self._video_process = VideoProcess(convert_gray=False)
        else:
            raise ValueError(f"Unknown detector {detector}")

        # Load model weights
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        self._model = ModelModule(args)
        self._model.model.load_state_dict(ckpt)
        self._model.eval()
        
        # Determine if we should use torchvision or av for loading
        self.load_video_func = load_video_compat

    def transcribe(self, video_path: str) -> str:
        """
        Runs lip-reading inference on a video file.
        Returns the transcribed text.
        """
        video_path = os.path.abspath(video_path)
        assert os.path.isfile(video_path), f"Video file not found: {video_path}"

        # 1. Load Video
        video = self.load_video_func(video_path)
        
        # 2. Extract Landmarks
        landmarks = self._landmarks_detector(video)
        
        # 3. Crop Mouth and Process
        video = self._video_process(video, landmarks)
        video = torch.tensor(video)
        
        # Auto-AVSR expects (T, C, H, W) format before transform
        video = video.permute((0, 3, 1, 2))
        
        # 4. Transform (Normalize/Augment)
        video = self._video_transform(video)
        
        # 5. Model Inference
        with torch.no_grad():
            transcript = self._model(video)
            
        if isinstance(transcript, (list, tuple)):
            transcript = " ".join(map(str, transcript))
            
        return str(transcript).strip().upper()
