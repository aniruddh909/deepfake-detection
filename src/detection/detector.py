"""
Real-time deepfake detection engine.
This module provides the main detection functionality for real-time analysis.
"""

import cv2  # type: ignore
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import logging
from pathlib import Path
import time
from collections import deque
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, src_dir)

try:
    from src.preprocessing.face_extractor import FaceLandmarkExtractor
    from src.preprocessing.data_loader import LandmarkDataLoader
    from src.utils.config import MODEL_CONFIG, DETECTION_CONFIG, VIDEO_CONFIG
    from src.utils.helpers import setup_logging, smooth_predictions, format_confidence, get_prediction_label
except ImportError:
    # Fallback for when running directly
    from preprocessing.face_extractor import FaceLandmarkExtractor
    from preprocessing.data_loader import LandmarkDataLoader
    from utils.config import MODEL_CONFIG, DETECTION_CONFIG, VIDEO_CONFIG
    from utils.helpers import setup_logging, smooth_predictions, format_confidence, get_prediction_label

import tensorflow as tf


class DeepfakeDetector:
    """
    Main deepfake detection engine for real-time analysis.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the deepfake detector.
        
        Args:
            model_path: Path to trained model file (optional)
        """
        self.logger = setup_logging()
        
        # Initialize components
        self.face_extractor = FaceLandmarkExtractor()
        self.data_loader = LandmarkDataLoader(use_enhanced_features=True, normalization_method='minmax')
        self.model = None  # Will be loaded from file
        
        # Detection state
        self.is_model_loaded = False
        self.landmark_buffer = deque(maxlen=10)  # Keep recent landmarks for stability
        self.prediction_history = deque(maxlen=DETECTION_CONFIG["smoothing_window"])
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        
        self.logger.info("DeepfakeDetector initialized")
    
    def load_model(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler (optional)
        """
        try:
            # Load the Keras model directly
            self.model = tf.keras.models.load_model(model_path)
            self.is_model_loaded = True
            self.logger.info(f"Model loaded from {model_path}")
            
            # Load scaler if provided, otherwise try to find it automatically
            if scaler_path is None:
                # Try to find the corresponding scaler file
                model_dir = Path(model_path).parent
                scaler_path = str(model_dir / "latest_scaler.pkl")
            
            if scaler_path and Path(scaler_path).exists():
                self.data_loader.load_scaler(scaler_path)
                self.logger.info(f"Scaler loaded from {scaler_path}")
            else:
                self.logger.warning("No scaler found - will use default normalization")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            raise
    
    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect deepfake in a single frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Dict containing detection results
        """
        result = {
            "has_face": False,
            "landmarks": None,
            "prediction": None,
            "confidence": None,
            "label": "NO_FACE",
            "smoothed_prediction": None,
            "alert": False
        }
        
        # Extract landmarks
        landmarks = self.face_extractor.extract_landmarks_from_frame(frame)
        
        if landmarks is not None:
            result["has_face"] = True
            result["landmarks"] = landmarks
            
            # Add to buffer
            self.landmark_buffer.append(landmarks)
            
            # For enhanced model, use single frame prediction with geometric features
            if self.is_model_loaded:
                # Use the most recent frame for prediction
                current_landmarks = landmarks
                
                # Process single frame with geometric features (matching training)
                processed_frame = self.data_loader._preprocess_single_landmark(current_landmarks)
                
                # Scale if scaler is available
                if self.data_loader.is_fitted:
                    processed_frame = processed_frame.reshape(1, -1)
                    processed_frame = self.data_loader.transform_data(processed_frame)
                    processed_frame = processed_frame[0]
                
                # Make prediction
                prediction_input = np.expand_dims(processed_frame, axis=0)
                prediction = self.model.predict(prediction_input, verbose=0)[0][0]
                
                result["prediction"] = prediction
                result["confidence"] = prediction if prediction > 0.5 else 1.0 - prediction
                result["label"] = get_prediction_label(prediction, DETECTION_CONFIG["confidence_threshold"])
                
                # Add to prediction history for smoothing
                self.prediction_history.append(prediction)
                
                # Calculate smoothed prediction
                if len(self.prediction_history) >= 3:
                    smoothed_predictions = smooth_predictions(
                        list(self.prediction_history), 
                        window_size=min(len(self.prediction_history), DETECTION_CONFIG["smoothing_window"])
                    )
                    result["smoothed_prediction"] = smoothed_predictions[-1]
                    
                    # Check for alert condition
                    if result["smoothed_prediction"] > DETECTION_CONFIG["alert_threshold"]:
                        result["alert"] = True
        
        return result
    
    def detect_video_file(self, video_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect deepfakes in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
        
        Returns:
            List of detection results for each frame
        """
        self.logger.info(f"Processing video file: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        # Reset detection state
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect deepfake in frame
                detection_result = self.detect_frame(frame)
                detection_result["frame_number"] = frame_count
                detection_result["timestamp"] = frame_count / fps
                results.append(detection_result)
                
                # Annotate frame if writer is available
                if writer:
                    annotated_frame = self.annotate_frame(frame, detection_result)
                    writer.write(annotated_frame)
                
                # Log progress
                if frame_count % 100 == 0:
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        self.logger.info(f"Processed {frame_count} frames from video")
        return results
    
    def detect_webcam(self, duration_seconds: int = 60) -> None:
        """
        Real-time detection from webcam.
        
        Args:
            duration_seconds: Duration to run detection
        """
        self.logger.info(f"Starting webcam detection for {duration_seconds} seconds")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG["resize_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG["resize_height"])
        cap.set(cv2.CAP_PROP_FPS, VIDEO_CONFIG["target_fps"])
        
        start_time = time.time()
        
        # Reset detection state
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration
                if time.time() - start_time > duration_seconds:
                    break
                
                # Detect deepfake
                detection_result = self.detect_frame(frame)
                
                # Update FPS counter
                self._update_fps()
                
                # Annotate and display frame
                annotated_frame = self.annotate_frame(frame, detection_result)
                cv2.imshow('Deepfake Detection', annotated_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        self.logger.info("Webcam detection stopped")
    
    def annotate_frame(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Input frame
            detection_result: Detection results
        
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw landmarks if available
        if detection_result["landmarks"] is not None:
            landmarks = detection_result["landmarks"]
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
        
        # Prepare text information
        info_lines = []
        
        # FPS
        info_lines.append(f"FPS: {self.current_fps:.1f}")
        
        # Face detection status
        if detection_result["has_face"]:
            info_lines.append("Face: DETECTED")
        else:
            info_lines.append("Face: NOT DETECTED")
        
        # Prediction results
        if detection_result["prediction"] is not None:
            confidence = format_confidence(detection_result["confidence"])
            info_lines.append(f"Prediction: {detection_result['label']}")
            info_lines.append(f"Confidence: {confidence}")
            
            if detection_result["smoothed_prediction"] is not None:
                smoothed_confidence = format_confidence(
                    detection_result["smoothed_prediction"] if detection_result["smoothed_prediction"] > 0.5 
                    else 1.0 - detection_result["smoothed_prediction"]
                )
                smoothed_label = get_prediction_label(
                    detection_result["smoothed_prediction"], 
                    DETECTION_CONFIG["confidence_threshold"]
                )
                info_lines.append(f"Smoothed: {smoothed_label} ({smoothed_confidence})")
        
        # Alert status
        if detection_result["alert"]:
            info_lines.append("ALERT: HIGH CONFIDENCE FAKE!")
        
        # Draw text on frame
        y_offset = 30
        for line in info_lines:
            # Choose color based on content
            color = (0, 255, 0)  # Green default
            if "FAKE" in line or "ALERT" in line:
                color = (0, 0, 255)  # Red for fake/alert
            elif "NOT DETECTED" in line:
                color = (0, 255, 255)  # Yellow for no face
            
            cv2.putText(annotated_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Draw alert border if needed
        if detection_result["alert"]:
            cv2.rectangle(annotated_frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
        
        return annotated_frame
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.fps_counter += 1
        
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            
            if elapsed_time > 0:
                self.current_fps = 30 / elapsed_time
            
            self.fps_start_time = current_time
    
    def get_detection_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from detection results.
        
        Args:
            results: List of detection results
        
        Returns:
            Dict containing summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}
        
        total_frames = len(results)
        frames_with_face = sum(1 for r in results if r["has_face"])
        frames_with_prediction = sum(1 for r in results if r["prediction"] is not None)
        
        if frames_with_prediction == 0:
            return {
                "total_frames": total_frames,
                "frames_with_face": frames_with_face,
                "face_detection_rate": frames_with_face / total_frames,
                "frames_with_prediction": 0,
                "average_prediction": None,
                "fake_probability": None
            }
        
        predictions = [r["prediction"] for r in results if r["prediction"] is not None]
        fake_predictions = [p for p in predictions if p > DETECTION_CONFIG["confidence_threshold"]]
        
        summary = {
            "total_frames": total_frames,
            "frames_with_face": frames_with_face,
            "face_detection_rate": frames_with_face / total_frames,
            "frames_with_prediction": frames_with_prediction,
            "average_prediction": np.mean(predictions),
            "fake_probability": len(fake_predictions) / frames_with_prediction,
            "confidence_scores": {
                "min": np.min(predictions),
                "max": np.max(predictions),
                "mean": np.mean(predictions),
                "std": np.std(predictions)
            }
        }
        
        return summary
    
    def reset_detection_state(self) -> None:
        """Reset the detection state (buffers and history)."""
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.logger.info("Detection state reset")


def main():
    """
    Example usage of the DeepfakeDetector.
    """
    # Initialize detector
    detector = DeepfakeDetector()
    
    print("DeepfakeDetector initialized")
    print("Note: Model not loaded - predictions will not be available")
    
    # Test with webcam (5 seconds)
    try:
        print("Starting webcam detection for 5 seconds...")
        print("Press 'q' to quit early")
        detector.detect_webcam(duration_seconds=5)
    except Exception as e:
        print(f"Webcam detection failed: {e}")
    
    print("Detection test completed")


if __name__ == "__main__":
    main()
