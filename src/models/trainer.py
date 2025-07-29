"""
Model training utilities for deepfake detection.
This module handles the complete training pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
import json
import time

from .cnn_model import CNNDeepfakeDetector
from ..preprocessing.data_loader import LandmarkDataLoader
from ..utils.config import MODEL_CONFIG, MODELS_DIR
from ..utils.helpers import setup_logging, save_json


class ModelTrainer:
    """
    Handles the complete training pipeline for deepfake detection models.
    """
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: Optional[str] = None,
                 lightweight: bool = False):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing training data
            output_dir: Directory to save trained models
            lightweight: Whether to use lightweight model architecture
        """
        self.logger = setup_logging()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else MODELS_DIR
        self.lightweight = lightweight
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = LandmarkDataLoader()
        self.model = CNNDeepfakeDetector()
        
        # Training state
        self.training_history = None
        self.model_metrics = None
        
        self.logger.info(f"ModelTrainer initialized - Data: {self.data_dir}, Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, ...]:
        """
        Load and prepare training data.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Loading and preparing data...")
        
        # Load dataset
        X, y = self.data_loader.load_dataset_from_directory(self.data_dir)
        
        # Get data statistics
        stats = self.data_loader.get_data_stats(X, y)
        self.logger.info(f"Dataset statistics: {stats}")
        
        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_dataset(X, y)
        
        # Fit and transform data
        X_train = self.data_loader.fit_transform_data(X_train)
        X_val = self.data_loader.transform_data(X_val)
        X_test = self.data_loader.transform_data(X_test)
        
        self.logger.info("Data preparation completed")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self) -> None:
        """Build the model architecture."""
        self.logger.info(f"Building {'lightweight' if self.lightweight else 'full'} model...")
        
        if self.lightweight:
            self.model.build_lightweight_model()
        else:
            self.model.build_model()
        
        # Print model summary
        self.logger.info("Model architecture:")
        self.logger.info(f"\n{self.model.get_model_summary()}")
    
    def train(self, 
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              validation_split: Optional[float] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
        
        Returns:
            Dict containing training history and metrics
        """
        start_time = time.time()
        
        # Use defaults from config if not specified
        epochs = epochs or MODEL_CONFIG["epochs"]
        batch_size = batch_size or MODEL_CONFIG["batch_size"]
        
        self.logger.info(f"Starting training - Epochs: {epochs}, Batch size: {batch_size}")
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
        
        # Build model
        self.build_model()
        
        # Train model
        self.training_history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate model
        self.logger.info("Evaluating model on test set...")
        test_metrics = self.model.evaluate(X_test, y_test)
        
        training_time = time.time() - start_time
        
        # Compile results
        results = {
            "training_history": self.training_history,
            "test_metrics": test_metrics,
            "training_time_seconds": training_time,
            "model_type": "lightweight" if self.lightweight else "full",
            "data_stats": self.data_loader.get_data_stats(X_train, y_train),
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "model_config": MODEL_CONFIG
            }
        }
        
        self.model_metrics = results
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Test metrics: {test_metrics}")
        
        return results
    
    def save_model(self, 
                   model_name: Optional[str] = None,
                   save_plots: bool = True) -> Dict[str, str]:
        """
        Save the trained model and associated files.
        
        Args:
            model_name: Name for the model files
            save_plots: Whether to save training plots
        
        Returns:
            Dict containing paths to saved files
        """
        if self.model.model is None:
            raise ValueError("No trained model to save")
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = int(time.time())
            model_type = "lightweight" if self.lightweight else "full"
            model_name = f"deepfake_model_{model_type}_{timestamp}"
        
        saved_files = {}
        
        # Save model
        model_path = self.output_dir / f"{model_name}.keras"
        self.model.save_model(str(model_path))
        saved_files["model"] = str(model_path)
        
        # Save scaler
        scaler_path = self.output_dir / f"{model_name}_scaler.pkl"
        self.data_loader.save_scaler(str(scaler_path))
        saved_files["scaler"] = str(scaler_path)
        
        # Save training metrics and history
        if self.model_metrics:
            metrics_path = self.output_dir / f"{model_name}_metrics.json"
            save_json(self.model_metrics, metrics_path)
            saved_files["metrics"] = str(metrics_path)
        
        # Save training plots
        if save_plots and self.training_history:
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_paths = self._save_training_plots(plots_dir, model_name)
            saved_files.update(plot_paths)
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "model_type": "lightweight" if self.lightweight else "full",
            "created_at": time.time(),
            "files": saved_files,
            "test_metrics": self.model_metrics.get("test_metrics", {}) if self.model_metrics else {}
        }
        
        info_path = self.output_dir / f"{model_name}_info.json"
        save_json(model_info, info_path)
        saved_files["info"] = str(info_path)
        
        self.logger.info(f"Model saved: {saved_files}")
        
        return saved_files
    
    def _save_training_plots(self, plots_dir: Path, model_name: str) -> Dict[str, str]:
        """
        Save training visualization plots.
        
        Args:
            plots_dir: Directory to save plots
            model_name: Model name for file naming
        
        Returns:
            Dict containing plot file paths
        """
        plot_paths = {}
        
        if not self.training_history:
            return plot_paths
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Training history plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision plot
        if 'precision' in self.training_history:
            axes[1, 0].plot(self.training_history['precision'], label='Training Precision')
            axes[1, 0].plot(self.training_history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall plot
        if 'recall' in self.training_history:
            axes[1, 1].plot(self.training_history['recall'], label='Training Recall')
            axes[1, 1].plot(self.training_history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save training history plot
        history_plot_path = plots_dir / f"{model_name}_training_history.png"
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths["training_history"] = str(history_plot_path)
        
        self.logger.info(f"Training plots saved to {plots_dir}")
        
        return plot_paths
    
    def create_sample_dataset(self, 
                            output_dir: str,
                            num_real: int = 50,
                            num_fake: int = 50) -> None:
        """
        Create a sample dataset for testing.
        
        Args:
            output_dir: Directory to save sample data
            num_real: Number of real samples
            num_fake: Number of fake samples
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        real_dir = output_path / "real"
        fake_dir = output_path / "fake"
        real_dir.mkdir(exist_ok=True)
        fake_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating sample dataset in {output_dir}")
        
        # Create sample data using data loader
        X_real, y_real = self.data_loader.create_sample_data(num_real * 2)  # Generate more, then filter
        X_fake, y_fake = self.data_loader.create_sample_data(num_fake * 2)
        
        # Filter to get real and fake samples
        real_samples = X_real[y_real == 0][:num_real]
        fake_samples = X_fake[y_fake == 1][:num_fake]
        
        # Save real samples
        for i, sample in enumerate(real_samples):
            # Convert back to landmark format (30, 468, 3)
            landmarks = sample.reshape(MODEL_CONFIG["sequence_length"], 468, 3)
            np.save(real_dir / f"real_video_{i:03d}.npy", landmarks)
        
        # Save fake samples
        for i, sample in enumerate(fake_samples):
            # Convert back to landmark format (30, 468, 3)
            landmarks = sample.reshape(MODEL_CONFIG["sequence_length"], 468, 3)
            np.save(fake_dir / f"fake_video_{i:03d}.npy", landmarks)
        
        self.logger.info(f"Sample dataset created: {num_real} real, {num_fake} fake samples")


def main():
    """
    Example usage of the ModelTrainer.
    """
    import tempfile
    
    # Create temporary directory for sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_data_dir = Path(temp_dir) / "sample_data"
        
        # Initialize trainer
        trainer = ModelTrainer(
            data_dir=str(sample_data_dir),
            lightweight=True  # Use lightweight model for quick testing
        )
        
        # Create sample dataset
        print("Creating sample dataset...")
        trainer.create_sample_dataset(str(sample_data_dir), num_real=20, num_fake=20)
        
        # Train model
        print("Training model...")
        results = trainer.train(epochs=5, batch_size=8)  # Quick training for demo
        
        # Save model
        print("Saving model...")
        saved_files = trainer.save_model("demo_model")
        
        print(f"Training completed!")
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.3f}")
        print(f"Files saved: {list(saved_files.keys())}")


if __name__ == "__main__":
    main()
