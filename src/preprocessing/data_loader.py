"""
Data loading and preprocessing utilities for deepfake detection.
This module handles the preparation of landmark data for training and inference.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

from ..utils.config import MODEL_CONFIG, PROCESSED_DATA_DIR
from ..utils.helpers import setup_logging, normalize_landmarks


class LandmarkDataLoader:
    """
    Data loader for face landmark sequences used in deepfake detection.
    """
    
    def __init__(self, use_enhanced_features=True, normalization_method='standard'):
        """
        Initialize the data loader.
        
        Args:
            use_enhanced_features: Whether to use enhanced geometric features
            normalization_method: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), or 'manual' (manual normalization)
        """
        self.logger = setup_logging()
        self.normalization_method = normalization_method
        
        if normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None  # Use manual normalization
            
        self.is_fitted = False
        self.use_enhanced_features = use_enhanced_features
        
        self.logger.info(f"LandmarkDataLoader initialized with {normalization_method} normalization")
    
    def _preprocess_single_landmark(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for individual landmark data to improve model performance.
        
        Args:
            landmarks: Raw landmarks of shape (478, 3)
            
        Returns:
            Enhanced feature vector with improved discriminative power (if enabled)
            or standard flattened landmarks
        """
        # 1. Basic normalization - center landmarks around mean
        centered_landmarks = landmarks - np.mean(landmarks, axis=0)
        
        # 2. Scale normalization - normalize by standard deviation
        std = np.std(centered_landmarks, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normalized_landmarks = centered_landmarks / std
        
        # 4. Combine original normalized landmarks with geometric features (if enabled)
        flattened_landmarks = normalized_landmarks.flatten()  # (1434,)
        
        if self.use_enhanced_features:
            # 3. Extract geometric features
            geometric_features = self._extract_geometric_features(normalized_landmarks)
            
            # Combine all features
            enhanced_features = np.concatenate([flattened_landmarks, geometric_features])
            return enhanced_features
        else:
            return flattened_landmarks
    
    def _extract_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract geometric features that are more discriminative for deepfake detection.
        
        Args:
            landmarks: Normalized landmarks of shape (478, 3)
            
        Returns:
            Geometric features array
        """
        features = []
        
        # 1. Face outline consistency (first 17 points are face outline)
        face_outline = landmarks[:17]
        outline_variance = np.var(face_outline, axis=0)
        features.extend(outline_variance)  # 3 values
        
        # 2. Eye region features (points 36-47 for eyes)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Eye symmetry
        eye_symmetry = np.abs(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        features.extend(eye_symmetry)  # 3 values
        
        # Eye opening (distance between top and bottom of eye)
        left_eye_opening = np.linalg.norm(left_eye[1] - left_eye[5])
        right_eye_opening = np.linalg.norm(right_eye[1] - right_eye[5])
        features.extend([left_eye_opening, right_eye_opening])  # 2 values
        
        # 3. Mouth region features (points 48-67)
        mouth = landmarks[48:68] if landmarks.shape[0] > 68 else landmarks[48:60]
        mouth_center = np.mean(mouth, axis=0)
        mouth_variance = np.var(mouth, axis=0)
        features.extend(mouth_variance)  # 3 values
        
        # Mouth width and height
        if landmarks.shape[0] > 60:
            mouth_width = np.linalg.norm(mouth[6] - mouth[0])  # Corner to corner
            mouth_height = np.linalg.norm(mouth[3] - mouth[9])  # Top to bottom
            features.extend([mouth_width, mouth_height])  # 2 values
        else:
            features.extend([0.0, 0.0])  # Default values
        
        # 4. Nose features (if available)
        if landmarks.shape[0] > 35:
            nose_tip = landmarks[33]
            nose_base = np.mean(landmarks[31:36], axis=0)
            nose_length = np.linalg.norm(nose_tip - nose_base)
            features.append(nose_length)  # 1 value
        else:
            features.append(0.0)
        
        # 5. Overall face proportions
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        face_ratio = face_width / (face_height + 1e-8)
        features.append(face_ratio)  # 1 value
        
        # 6. Landmark density and distribution
        landmarks_std = np.std(landmarks, axis=0)
        features.extend(landmarks_std)  # 3 values
        
        return np.array(features, dtype=np.float32)
    
    def preprocess_landmarks_sequence(self, landmarks_list: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a sequence of landmarks for model input.
        
        Args:
            landmarks_list: List of landmark arrays, each of shape (468, 3)
        
        Returns:
            np.ndarray: Preprocessed sequence of shape (sequence_length, 1404)
        """
        sequence_length = MODEL_CONFIG["sequence_length"]
        
        if len(landmarks_list) == 0:
            raise ValueError("Empty landmarks sequence")
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks_list)  # Shape: (frames, 468, 3)
        
        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(landmarks_array)
        
        # Handle sequence length
        if len(normalized_landmarks) >= sequence_length:
            # Take the first sequence_length frames
            processed_sequence = normalized_landmarks[:sequence_length]
        else:
            # Pad with last frame if sequence is too short
            last_frame = normalized_landmarks[-1]
            padding_needed = sequence_length - len(normalized_landmarks)
            padding = np.tile(last_frame[np.newaxis, :, :], (padding_needed, 1, 1))
            processed_sequence = np.concatenate([normalized_landmarks, padding], axis=0)
        
        # Flatten landmarks for each frame: (sequence_length, 468*3)
        flattened_sequence = processed_sequence.reshape(sequence_length, -1)
        
        return flattened_sequence
    
    def create_sequences_from_video_landmarks(self, 
                                            landmarks_list: List[np.ndarray],
                                            stride: int = 15) -> List[np.ndarray]:
        """
        Create multiple overlapping sequences from a long video.
        
        Args:
            landmarks_list: List of landmark arrays from a video
            stride: Stride between sequences (frames to skip)
        
        Returns:
            List of preprocessed sequences
        """
        sequence_length = MODEL_CONFIG["sequence_length"]
        sequences = []
        
        if len(landmarks_list) < sequence_length:
            # If video is too short, return single padded sequence
            return [self.preprocess_landmarks_sequence(landmarks_list)]
        
        # Create overlapping sequences
        for start_idx in range(0, len(landmarks_list) - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            sequence = landmarks_list[start_idx:end_idx]
            processed_sequence = self.preprocess_landmarks_sequence(sequence)
            sequences.append(processed_sequence)
        
        return sequences
    
    def load_dataset_from_directory(self, 
                                  data_dir: Union[str, Path],
                                  real_subdir: str = "real",
                                  fake_subdir: str = "fake") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from directory structure containing individual landmark files.
        
        Expected structure:
        data_dir/
        ├── real/
        │   ├── real_00001.npy  # Each file contains landmarks of shape (478, 3)
        │   ├── real_00002.npy
        │   └── ...
        └── fake/
            ├── fake_00001.npy  # Each file contains landmarks of shape (478, 3)
            ├── fake_00002.npy
            └── ...
        
        Args:
            data_dir: Root data directory
            real_subdir: Subdirectory containing real face landmarks
            fake_subdir: Subdirectory containing fake face landmarks
        
        Returns:
            Tuple of (X, y) where X is flattened features (1434,) and y is labels
        """
        data_dir = Path(data_dir)
        real_dir = data_dir / real_subdir
        fake_dir = data_dir / fake_subdir
        
        self.logger.info(f"Loading dataset from {data_dir}")
        
        X_features = []
        y_labels = []
        
        # Load real face data
        if real_dir.exists():
            real_files = list(real_dir.glob("*.npy"))
            self.logger.info(f"Found {len(real_files)} real face files")
            
            for file_path in real_files:
                try:
                    landmarks = np.load(file_path)  # Shape: (478, 3)
                    
                    # Apply enhanced preprocessing
                    processed_landmarks = self._preprocess_single_landmark(landmarks)
                    
                    X_features.append(processed_landmarks)
                    y_labels.append(1)  # 1 for real (swapped for better class balance)
                    
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
        
        # Load fake face data
        if fake_dir.exists():
            fake_files = list(fake_dir.glob("*.npy"))
            self.logger.info(f"Found {len(fake_files)} fake face files")
            
            for file_path in fake_files:
                try:
                    landmarks = np.load(file_path)  # Shape: (478, 3)
                    
                    # Apply enhanced preprocessing
                    processed_landmarks = self._preprocess_single_landmark(landmarks)
                    
                    X_features.append(processed_landmarks)
                    y_labels.append(0)  # 0 for fake (swapped for better class balance)
                    
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
        
        if len(X_features) == 0:
            raise ValueError("No valid data found")
        
        # Convert to numpy arrays
        X = np.array(X_features)  # Shape: (num_samples, 1434)
        y = np.array(y_labels)    # Shape: (num_samples,)
        
        self.logger.info(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")
        self.logger.info(f"Real samples: {np.sum(y == 1)}, Fake samples: {np.sum(y == 0)}")
        self.logger.info(f"Feature dimension: {X.shape[1]} (enhanced with geometric features)")
        
        return X, y
    
    def load_single_video_landmarks(self, file_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load landmarks from a single video file.
        
        Args:
            file_path: Path to the landmarks file
        
        Returns:
            List of preprocessed sequences
        """
        landmarks_array = np.load(file_path)
        landmarks_frames = [landmarks_array[i] for i in range(len(landmarks_array))]
        
        return self.create_sequences_from_video_landmarks(landmarks_frames)
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """
        Fit the scaler on training data.
        
        Args:
            X: Training data of shape (samples, features) where features = 1434 (478*3)
        """
        if self.scaler is not None:
            self.scaler.fit(X)
            self.is_fitted = True
            self.logger.info(f"Scaler fitted on training data with shape {X.shape}")
        else:
            self.is_fitted = True
            self.logger.info(f"Manual normalization - no scaler fitting needed for data with shape {X.shape}")
    
    def transform_data(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            X: Data to transform of shape (samples, features)
        
        Returns:
            np.ndarray: Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        if self.scaler is not None:
            return self.scaler.transform(X)
        else:
            # Return data as-is for manual normalization (already applied in preprocessing)
            return X
    
    def fit_transform_data(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            X: Training data
        
        Returns:
            np.ndarray: Scaled data
        """
        self.fit_scaler(X)
        return self.transform_data(X)
    
    def split_dataset(self, 
                     X: np.ndarray, 
                     y: np.ndarray,
                     test_size: float = 0.2,
                     val_size: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data after test split)
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scaler(self, file_path: Union[str, Path]) -> None:
        """
        Save the fitted scaler.
        
        Args:
            file_path: Path to save the scaler
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if self.scaler is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info(f"Scaler saved to {file_path}")
        else:
            self.logger.info("Manual normalization - no scaler to save")
    
    def load_scaler(self, file_path: Union[str, Path]) -> None:
        """
        Load a fitted scaler.
        
        Args:
            file_path: Path to the saved scaler
        """
        if self.scaler is not None:
            with open(file_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            self.logger.info(f"Scaler loaded from {file_path}")
        else:
            self.is_fitted = True
            self.logger.info("Manual normalization - no scaler to load")
    
    def get_data_stats(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            X: Features of shape (samples, 1434)
            y: Labels of shape (samples,)
        
        Returns:
            Dict containing dataset statistics
        """
        stats = {
            "total_samples": len(X),
            "num_features": X.shape[1],
            "real_samples": np.sum(y == 0),
            "fake_samples": np.sum(y == 1),
            "class_balance": np.sum(y == 0) / len(y),
            "feature_stats": {
                "mean": np.mean(X),
                "std": np.std(X),
                "min": np.min(X),
                "max": np.max(X)
            }
        }
        
        return stats
    
    def load_and_split_dataset(self, 
                              data_dir: Union[str, Path],
                              validation_split: float = 0.2,
                              random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load dataset from directory and split into train/validation sets.
        
        Args:
            data_dir: Root data directory containing 'real' and 'fake' subdirectories
            validation_split: Proportion of data to use for validation (0.0-1.0)
            random_state: Random seed for reproducible splits
        
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val))
            where X has shape (samples, 1434) - flattened 3D landmarks
        """
        # Load the dataset
        X, y = self.load_dataset_from_directory(data_dir)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=random_state, 
            stratify=y
        )
        
        # Fit scaler on training data and transform both sets
        X_train_scaled = self.fit_transform_data(X_train)
        X_val_scaled = self.transform_data(X_val)
        
        self.logger.info(f"Dataset loaded and split:")
        self.logger.info(f"  Training: {X_train_scaled.shape[0]} samples")
        self.logger.info(f"  Validation: {X_val_scaled.shape[0]} samples")
        self.logger.info(f"  Feature dimension: {X_train_scaled.shape[1]} (flattened 3D landmarks)")
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val)
    
    def create_sample_data(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sample synthetic data for testing.
        
        Args:
            num_samples: Number of samples to create
        
        Returns:
            Tuple of (X, y) synthetic data
        """
        sequence_length = MODEL_CONFIG["sequence_length"]
        features_per_frame = MODEL_CONFIG["input_shape"][0] * MODEL_CONFIG["input_shape"][1]
        
        # Create random landmark-like data
        X = np.random.randn(num_samples, sequence_length, features_per_frame)
        
        # Add some patterns to distinguish real vs fake
        for i in range(num_samples):
            if i % 2 == 0:  # Real samples
                # Add some stability (less variation)
                X[i] = X[i] * 0.5
            else:  # Fake samples
                # Add more variation and some artifacts
                X[i] = X[i] * 1.5
                # Add some periodic artifacts
                X[i, :, :100] += 0.3 * np.sin(np.arange(sequence_length))[:, np.newaxis]
        
        # Create labels
        y = np.array([i % 2 for i in range(num_samples)])
        
        self.logger.info(f"Created sample data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y


def main():
    """
    Example usage of the LandmarkDataLoader.
    """
    loader = LandmarkDataLoader()
    
    # Create sample data
    print("Creating sample data...")
    X, y = loader.create_sample_data(num_samples=200)
    
    # Get data statistics
    stats = loader.get_data_stats(X, y)
    print(f"Data statistics: {stats}")
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(X, y)
    
    # Fit and transform data
    X_train_scaled = loader.fit_transform_data(X_train)
    X_val_scaled = loader.transform_data(X_val)
    X_test_scaled = loader.transform_data(X_test)
    
    print(f"Scaled data shapes:")
    print(f"Train: {X_train_scaled.shape}")
    print(f"Val: {X_val_scaled.shape}")
    print(f"Test: {X_test_scaled.shape}")


if __name__ == "__main__":
    main()
