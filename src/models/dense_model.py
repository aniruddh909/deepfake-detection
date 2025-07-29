"""
Dense neural network model for deepfake detection using individual face landmarks.
This module defines a feedforward neural network for classifying real vs fake faces
using flattened 3D face landmark coordinates.
"""

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, Model  # type: ignore
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from ..utils.config import MODEL_CONFIG
from ..utils.helpers import setup_logging


class DeepfakeDenseNet:
    """
    Dense neural network model for deepfake detection using individual face landmarks.
    """
    
    def __init__(self):
        """Initialize the deepfake dense model."""
        self.logger = setup_logging()
        self.model = None
        self.history = None
        
        self.logger.info("DeepfakeDenseNet initialized")
    
    def build_model(self, input_shape: Optional[Tuple[int,]] = None) -> Model:
        """
        Build the dense neural network architecture.
        
        Args:
            input_shape: Input shape (features,)
                        Default: (1434,) for 478 landmarks with xyz coordinates
        
        Returns:
            keras.Model: Compiled model
        """
        if input_shape is None:
            # Default: 478 landmarks * 3 coordinates = 1434 features
            landmarks_count = MODEL_CONFIG["input_shape"][0]
            coordinates = MODEL_CONFIG["input_shape"][1]
            input_shape = (landmarks_count * coordinates,)
        
        self.logger.info(f"Building model with input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name="landmark_input")
        
        # Dense layers with batch normalization and dropout
        x = layers.Dense(512, activation='relu', name="dense_1")(inputs)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        
        x = layers.Dense(256, activation='relu', name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        
        x = layers.Dense(128, activation='relu', name="dense_3")(x)
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.Dropout(0.2, name="dropout_3")(x)
        
        x = layers.Dense(64, activation='relu', name="dense_4")(x)
        x = layers.BatchNormalization(name="bn_4")(x)
        x = layers.Dropout(0.2, name="dropout_4")(x)
        
        # Output layer for binary classification
        outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="deepfake_densenet")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"]),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        self.logger.info(f"Model built successfully with {model.count_params():,} parameters")
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Dict containing training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if epochs is None:
            epochs = MODEL_CONFIG["epochs"]
        if batch_size is None:
            batch_size = MODEL_CONFIG["batch_size"]
            
        self.logger.info(f"Starting training with {len(X_train)} samples for {epochs} epochs")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        self.logger.info("Training completed")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Create results dictionary
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
        
        self.logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make class predictions on input data.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Class predictions (0 or 1)
        """
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.summary()


def create_deepfake_model(input_shape: Optional[Tuple[int,]] = None) -> DeepfakeDenseNet:
    """
    Factory function to create and build a deepfake detection model.
    
    Args:
        input_shape: Input shape for the model
        
    Returns:
        Configured DeepfakeDenseNet instance
    """
    model = DeepfakeDenseNet()
    model.build_model(input_shape)
    return model
