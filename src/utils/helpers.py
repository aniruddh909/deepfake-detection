"""
Utility functions for the Deepfake Detection System.
"""

import os
import logging
import numpy as np
import cv2  # type: ignore
from pathlib import Path
from typing import List, Tuple, Optional, Union
import json

from .config import LOGGING_CONFIG, VIDEO_CONFIG


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["log_file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_video_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if the file is a valid video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        bool: True if valid video file, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        cap = cv2.VideoCapture(str(file_path))
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False


def get_video_info(file_path: Union[str, Path]) -> dict:
    """
    Get video information including fps, frame count, duration.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(str(file_path))
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]
    else:
        info["duration"] = 0
    
    cap.release()
    return info


def resize_frame(frame: np.ndarray, target_width: Optional[int] = None, target_height: Optional[int] = None) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width (optional)
        target_height: Target height (optional)
        
    Returns:
        np.ndarray: Resized frame
    """
    if target_width is None:
        target_width = VIDEO_CONFIG["resize_width"]
    if target_height is None:
        target_height = VIDEO_CONFIG["resize_height"]
    
    height, width = frame.shape[:2]
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    return cv2.resize(frame, (new_width, new_height))


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize face landmarks to [-1, 1] range.
    
    Args:
        landmarks: Raw landmarks array of shape (N, 468, 3)
        
    Returns:
        np.ndarray: Normalized landmarks
    """
    # Center the landmarks
    centered = landmarks - np.mean(landmarks, axis=1, keepdims=True)
    
    # Scale to [-1, 1] range
    max_range = np.max(np.abs(centered), axis=(1, 2), keepdims=True)
    normalized = centered / (max_range + 1e-8)
    
    return normalized


def smooth_predictions(predictions: List[float], window_size: int = 5) -> List[float]:
    """
    Apply moving average smoothing to predictions.
    
    Args:
        predictions: List of prediction values
        window_size: Size of smoothing window
        
    Returns:
        List[float]: Smoothed predictions
    """
    if len(predictions) < window_size:
        return predictions
    
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(predictions), i + window_size // 2 + 1)
        smoothed.append(np.mean(predictions[start_idx:end_idx]))
    
    return smoothed


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Union[str, Path]) -> dict:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        dict: Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def create_directories(directories: List[Union[str, Path]]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def format_confidence(confidence: float) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        str: Formatted confidence string
    """
    return f"{confidence * 100:.1f}%"


def get_prediction_label(confidence: float, threshold: float = 0.5) -> str:
    """
    Get prediction label based on confidence and threshold.
    
    Args:
        confidence: Model confidence score
        threshold: Decision threshold
        
    Returns:
        str: Prediction label
    """
    return "FAKE" if confidence > threshold else "REAL"
