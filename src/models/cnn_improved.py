"""
CNN-based deepfake detection model for improved spatial pattern recognition
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class CNNDeepfakeDetector:
    """
    CNN-based architecture for deepfake detection using facial landmarks
    """
    
    def __init__(self, input_shape=(478, 3), num_classes=1):
        """
        Initialize CNN deepfake detector
        
        Args:
            input_shape: Shape of input landmarks (478, 3)
            num_classes: Number of output classes (1 for binary)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_cnn_model(self):
        """
        Build CNN architecture for spatial pattern recognition
        """
        # Input layer for landmarks (478 points with x,y,z coordinates)
        inputs = layers.Input(shape=self.input_shape, name='landmark_input')
        
        # Reshape for CNN processing - treat as 1D spatial sequence
        # Convert (478, 3) to (478, 3, 1) for Conv1D
        x = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)
        
        # First Conv2D block - detect local patterns
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Dropout(0.1)(x)
        
        # Second Conv2D block - detect mid-level patterns
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Dropout(0.2)(x)
        
        # Third Conv2D block - detect high-level patterns
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Dropout(0.3)(x)
        
        # Global patterns
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='cnn_deepfake_detector')
        
        return model
    
    def build_hybrid_model(self):
        """
        Build hybrid CNN + Dense model for better performance
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='landmark_input')
        
        # Spatial processing branch (CNN)
        spatial_branch = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)
        spatial_branch = layers.Conv2D(64, (5, 3), activation='relu', padding='same')(spatial_branch)
        spatial_branch = layers.BatchNormalization()(spatial_branch)
        spatial_branch = layers.MaxPooling2D((2, 1))(spatial_branch)
        
        spatial_branch = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(spatial_branch)
        spatial_branch = layers.BatchNormalization()(spatial_branch)
        spatial_branch = layers.MaxPooling2D((2, 1))(spatial_branch)
        
        spatial_branch = layers.GlobalAveragePooling2D()(spatial_branch)
        spatial_branch = layers.Dense(256, activation='relu')(spatial_branch)
        
        # Global processing branch (Dense)
        global_branch = layers.Flatten()(inputs)
        global_branch = layers.Dense(512, activation='relu')(global_branch)
        global_branch = layers.BatchNormalization()(global_branch)
        global_branch = layers.Dropout(0.3)(global_branch)
        
        global_branch = layers.Dense(256, activation='relu')(global_branch)
        global_branch = layers.BatchNormalization()(global_branch)
        global_branch = layers.Dropout(0.2)(global_branch)
        
        # Combine branches
        combined = layers.Concatenate()([spatial_branch, global_branch])
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.3)(combined)
        
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='sigmoid', name='output')(combined)
        
        model = Model(inputs=inputs, outputs=outputs, name='hybrid_deepfake_detector')
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile model with optimized settings
        """
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_path, patience=15):
        """
        Get training callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks

def create_cnn_model(input_shape=(478, 3), model_type='hybrid'):
    """
    Factory function to create CNN models
    """
    detector = CNNDeepfakeDetector(input_shape)
    
    if model_type == 'cnn':
        model = detector.build_cnn_model()
    elif model_type == 'hybrid':
        model = detector.build_hybrid_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return detector.compile_model(model), detector

if __name__ == "__main__":
    # Test model creation
    model, detector = create_cnn_model(model_type='hybrid')
    print("CNN Deepfake Detector created successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Print model summary
    model.summary()
