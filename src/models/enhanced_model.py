"""
Enhanced dense neural network model for deepfake detection with improved accuracy.
This module includes advanced techniques for better performance and generalization.
"""

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, Model, regularizers  # type: ignore
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path

from ..utils.config import MODEL_CONFIG
from ..utils.helpers import setup_logging


class EnhancedDeepfakeNet:
    """
    Enhanced dense neural network model for deepfake detection with improved accuracy.
    """
    
    def __init__(self):
        """Initialize the enhanced deepfake model."""
        self.logger = setup_logging()
        self.model = None
        self.history = None
        
        self.logger.info("EnhancedDeepfakeNet initialized")
    
    def build_model(self, input_shape: Optional[Tuple[int,]] = None) -> Model:
        """
        Build the enhanced dense neural network architecture.
        
        Args:
            input_shape: Input shape (features,)
                        Default: calculated from landmarks + geometric features
        
        Returns:
            keras.Model: Compiled model
        """
        if input_shape is None:
            # Default: 478 landmarks * 3 coordinates + geometric features ≈ 1456
            landmarks_count = MODEL_CONFIG["input_shape"][0]
            coordinates = MODEL_CONFIG["input_shape"][1]
            geometric_features = 22  # From enhanced preprocessing
            input_shape = (landmarks_count * coordinates + geometric_features,)
        
        self.logger.info(f"Building enhanced model with input shape: {input_shape}")
        
        # Input layer with noise for regularization
        inputs = layers.Input(shape=input_shape, name="landmark_input")
        
        # Add gaussian noise for better generalization
        x = layers.GaussianNoise(0.01, name="input_noise")(inputs)
        
        # First dense block with more neurons and better regularization
        x = layers.Dense(1024, activation='relu', 
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        name="dense_1")(x)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(0.4, name="dropout_1")(x)
        
        # Second dense block
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        
        # Third dense block
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        name="dense_3")(x)
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.Dropout(0.3, name="dropout_3")(x)
        
        # Fourth dense block
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        name="dense_4")(x)
        x = layers.BatchNormalization(name="bn_4")(x)
        x = layers.Dropout(0.2, name="dropout_4")(x)
        
        # Fifth dense block - smaller
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        name="dense_5")(x)
        x = layers.BatchNormalization(name="bn_5")(x)
        x = layers.Dropout(0.2, name="dropout_5")(x)
        
        # Final dense layer before output
        x = layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        name="dense_6")(x)
        x = layers.BatchNormalization(name="bn_6")(x)
        x = layers.Dropout(0.1, name="dropout_6")(x)
        
        # Output layer for binary classification
        outputs = layers.Dense(1, activation='sigmoid', name="output")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="enhanced_deepfake_net")
        
        # Use advanced optimizer with learning rate scheduling
        optimizer = keras.optimizers.Adam(
            learning_rate=MODEL_CONFIG["learning_rate"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=True  # Use AMSGrad variant for better convergence
        )
        
        # Compile model with simplified metrics for compatibility
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']  # Simplified metrics to avoid conflicts
        )
        
        self.model = model
        self.logger.info(f"Enhanced model built successfully with {model.count_params():,} parameters")
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              verbose: int = 1,
              class_weight: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """
        Train the enhanced model with advanced techniques.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            class_weight: Class weights for imbalanced data
            
        Returns:
            Dict containing training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if epochs is None:
            epochs = MODEL_CONFIG["epochs"]
        if batch_size is None:
            batch_size = MODEL_CONFIG["batch_size"]
        
        self.logger.info(f"Starting enhanced training with {len(X_train)} samples for {epochs} epochs")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        self.logger.info(f"Original class distribution: {class_distribution}")
        
        # Apply SMOTE for better class balance instead of class weights
        try:
            from imblearn.over_sampling import SMOTE
            # Use more conservative parameters to avoid memory issues
            smote = SMOTE(
                random_state=42,
                k_neighbors=min(5, min(counts) - 1) if min(counts) > 1 else 1,
                sampling_strategy='auto'
            )
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            self.logger.info(f"Applied SMOTE: {len(X_train)} -> {len(X_train_balanced)} samples")
            
            # Verify the resampling worked correctly
            unique_balanced, counts_balanced = np.unique(y_train_balanced, return_counts=True)
            balanced_distribution = dict(zip(unique_balanced, counts_balanced))
            self.logger.info(f"Balanced class distribution: {balanced_distribution}")
            
            X_train = X_train_balanced
            y_train = y_train_balanced
            
        except (ImportError, ValueError, Exception) as e:
            self.logger.warning(f"SMOTE failed ({type(e).__name__}: {str(e)}), using class weights instead")
            
            # Calculate class weights for imbalanced data
            class_counts = np.bincount(y_train.astype(int))
            total_samples = len(y_train)
            class_weight = {}
            
            for i, count in enumerate(class_counts):
                if count > 0:
                    weight = total_samples / (len(class_counts) * count)
                    class_weight[i] = weight
            
            self.logger.info(f"Using class weights: {class_weight}")
            # Store class weights for use in model.fit()
            self._class_weight = class_weight
        
        # Advanced callbacks
        callbacks = [
            # Early stopping with more patience
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-4
            ),
            
            # Learning rate reduction with more aggressive schedule
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=1e-8,
                verbose=1,
                min_delta=1e-4
            ),
            
            # Cosine annealing for learning rate
            keras.callbacks.LearningRateScheduler(
                lambda epoch: MODEL_CONFIG["learning_rate"] * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        # Determine whether to use class weights
        use_class_weight = hasattr(self, '_class_weight')
        
        # Train model with appropriate settings
        if use_class_weight:
            self.logger.info("Training with class weights due to SMOTE failure")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True,
                class_weight=self._class_weight
            )
        else:
            self.logger.info("Training with SMOTE-balanced data")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose,
                shuffle=True
            )
        
        self.history = history.history
        self.logger.info("Enhanced training completed")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the enhanced model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.logger.info(f"Evaluating enhanced model on {len(X_test)} test samples")
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Create results dictionary
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
        
        # Calculate additional metrics
        y_pred_prob = self.predict(X_test)
        y_pred = self.predict_classes(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # ROC AUC
        try:
            metrics['roc_auc_sklearn'] = float(roc_auc_score(y_test, y_pred_prob))
        except:
            metrics['roc_auc_sklearn'] = 0.0
        
        # Confusion matrix based metrics
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        self.logger.info(f"Enhanced evaluation results: {metrics}")
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
        self.logger.info(f"Enhanced model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        self.logger.info(f"Enhanced model loaded from {filepath}")
    
    def get_model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.summary()


def create_enhanced_deepfake_model(input_shape: Optional[Tuple[int,]] = None) -> EnhancedDeepfakeNet:
    """
    Factory function to create and build an enhanced deepfake detection model.
    
    Args:
        input_shape: Input shape for the model
        
    Returns:
        Configured EnhancedDeepfakeNet instance
    """
    model = EnhancedDeepfakeNet()
    model.build_model(input_shape)
    return model
