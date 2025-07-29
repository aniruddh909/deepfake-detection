"""
Configuration settings for the Deepfake Detection System.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Model configuration
MODEL_CONFIG = {
    "input_shape": (478, 3),  # MediaPipe Face Mesh landmarks: 478 points with x,y,z coordinates (updated model)
    "sequence_length": 30,    # Number of frames to consider for temporal analysis
    "num_classes": 2,         # Binary classification: Real vs Fake
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2,
    "early_stopping_patience": 10
}

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

# Video processing configuration
VIDEO_CONFIG = {
    "target_fps": 30,
    "max_frames": 300,  # Maximum frames to process per video
    "resize_width": 640,
    "resize_height": 480
}

# Web application configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": True,
    "upload_folder": str(DATA_DIR / "uploads"),
    "max_content_length": 16 * 1024 * 1024,  # 16MB max file size
    "allowed_extensions": {".mp4", ".avi", ".mov", ".mkv", ".webm"}
}

# Detection thresholds
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "smoothing_window": 5,  # Number of frames for prediction smoothing
    "alert_threshold": 0.7   # Threshold for high-confidence fake detection
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": str(PROJECT_ROOT / "logs" / "deepfake_detection.log")
}

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  Path(WEB_CONFIG["upload_folder"]), Path(LOGGING_CONFIG["log_file"]).parent]:
    directory.mkdir(parents=True, exist_ok=True)
