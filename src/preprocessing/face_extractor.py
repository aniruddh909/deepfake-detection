"""
Face landmark extraction using MediaPipe Face Mesh.
This module handles the extraction of 3D face landmarks from video frames.
"""

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np
from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path

from ..utils.config import MEDIAPIPE_CONFIG, VIDEO_CONFIG
from ..utils.helpers import setup_logging, resize_frame, get_video_info


class FaceLandmarkExtractor:
    """
    Extracts 3D face landmarks using MediaPipe Face Mesh.
    Supports context manager protocol for automatic resource cleanup.
    """
    
    def __init__(self):
        """Initialize the MediaPipe Face Mesh model."""
        self.logger = setup_logging()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore
        self.mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
        
        # Initialize Face Mesh with error handling
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=MEDIAPIPE_CONFIG["max_num_faces"],
                refine_landmarks=MEDIAPIPE_CONFIG["refine_landmarks"],
                min_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
                min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"]
            )
            self.logger.info("FaceLandmarkExtractor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
            raise RuntimeError(f"MediaPipe initialization failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face landmarks from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            np.ndarray: Landmarks array of shape (468, 3) or None if no face detected
        """
        if self.face_mesh is None:
            self.logger.error("Face mesh not initialized")
            return None
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get the first face (we configured max_num_faces=1)
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            height, width = frame.shape[:2]
            
            for landmark in face_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z * width  # Z is also normalized relative to width
                landmarks.append([x, y, z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def extract_landmarks_from_video(self, 
                                   video_path: Union[str, Path], 
                                   max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], dict]:
        """
        Extract face landmarks from a video file.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process (None for all frames)
            
        Returns:
            Tuple containing:
                - List of landmark arrays (each of shape (468, 3))
                - Video metadata dictionary
        """
        video_path = str(video_path)
        self.logger.info(f"Processing video: {video_path}")
        
        # Get video information
        video_info = get_video_info(video_path)
        self.logger.info(f"Video info: {video_info}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        landmarks_sequence = []
        frame_count = 0
        processed_frames = 0
        
        if max_frames is None:
            max_frames = min(video_info["frame_count"], VIDEO_CONFIG["max_frames"])
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if we've reached the maximum
                if processed_frames >= max_frames:
                    break
                
                # Resize frame for processing
                resized_frame = resize_frame(frame)
                
                # Extract landmarks
                landmarks = self.extract_landmarks_from_frame(resized_frame)
                
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
                    processed_frames += 1
                    
                    if processed_frames % 30 == 0:  # Log progress every 30 frames
                        self.logger.info(f"Processed {processed_frames} frames with face detection")
        
        finally:
            cap.release()
        
        self.logger.info(f"Extracted landmarks from {len(landmarks_sequence)} frames out of {frame_count} total frames")
        
        # Update video info with processing results
        video_info.update({
            "processed_frames": len(landmarks_sequence),
            "total_frames_read": frame_count,
            "detection_rate": len(landmarks_sequence) / frame_count if frame_count > 0 else 0
        })
        
        return landmarks_sequence, video_info
    
    def extract_landmarks_from_webcam(self, duration_seconds: int = 10) -> List[np.ndarray]:
        """
        Extract face landmarks from webcam stream.
        
        Args:
            duration_seconds: Duration to capture in seconds
            
        Returns:
            List of landmark arrays
        """
        self.logger.info(f"Starting webcam capture for {duration_seconds} seconds")
        
        cap = cv2.VideoCapture(0)  # Default webcam
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG["resize_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG["resize_height"])
        cap.set(cv2.CAP_PROP_FPS, VIDEO_CONFIG["target_fps"])
        
        landmarks_sequence = []
        start_time = cv2.getTickCount()
        target_duration = duration_seconds * cv2.getTickFrequency()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if duration exceeded
                current_time = cv2.getTickCount()
                if current_time - start_time > target_duration:
                    break
                
                # Extract landmarks
                landmarks = self.extract_landmarks_from_frame(frame)
                
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
                
                # Display frame (optional, for debugging)
                cv2.imshow('Webcam Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        self.logger.info(f"Captured {len(landmarks_sequence)} frames with face detection from webcam")
        return landmarks_sequence
    
    def visualize_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Visualize face landmarks on a frame.
        
        Args:
            frame: Input frame
            landmarks: Landmarks array of shape (468, 3)
            
        Returns:
            np.ndarray: Frame with landmarks drawn
        """
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark[0]), int(landmark[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
        
        return annotated_frame
    
    def save_landmarks(self, landmarks_sequence: List[np.ndarray], output_path: Union[str, Path]) -> None:
        """
        Save landmarks sequence to a numpy file.
        
        Args:
            landmarks_sequence: List of landmark arrays
            output_path: Output file path
        """
        # Convert to numpy array
        landmarks_array = np.array(landmarks_sequence)
        
        # Save to file
        np.save(output_path, landmarks_array)
        self.logger.info(f"Saved landmarks to {output_path} with shape {landmarks_array.shape}")
    
    def load_landmarks(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load landmarks from a numpy file.
        
        Args:
            file_path: Input file path
            
        Returns:
            np.ndarray: Landmarks array
        """
        landmarks = np.load(file_path)
        self.logger.info(f"Loaded landmarks from {file_path} with shape {landmarks.shape}")
        return landmarks
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
        except Exception as e:
            # Silently handle cleanup errors to avoid issues during garbage collection
            pass
    
    def close(self):
        """Explicitly close MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
            self.logger.info("MediaPipe resources closed")


def main():
    """
    Example usage of the FaceLandmarkExtractor.
    """
    # Example: Extract landmarks from webcam using context manager
    try:
        with FaceLandmarkExtractor() as extractor:
            landmarks = extractor.extract_landmarks_from_webcam(duration_seconds=5)
            print(f"Extracted {len(landmarks)} frames from webcam")
            
            if landmarks:
                # Save landmarks
                extractor.save_landmarks(landmarks, "webcam_landmarks.npy")
                print("Landmarks saved successfully")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
