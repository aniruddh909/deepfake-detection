#!/usr/bin/env python3
"""
Dataset Processing Script for Deepfake Detection

This script processes all images in the sample_dataset folder and extracts
3D face landmarks using MediaPipe, saving them as numpy arrays for training.

Usage:
    python process_dataset.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional, Tuple
import time
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing.face_extractor import FaceLandmarkExtractor
from src.utils.helpers import setup_logging


class DatasetProcessor:
    """
    Processes image datasets to extract face landmarks for deepfake detection training.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the dataset processor.
        
        Args:
            input_dir: Input directory containing real/ and fake/ subdirectories
            output_dir: Output directory to save processed landmarks
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.logger = setup_logging()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "fake").mkdir(parents=True, exist_ok=True)
        
        # Initialize face extractor
        self.face_extractor = FaceLandmarkExtractor()
        
        # Statistics
        self.stats = {
            "real": {"total": 0, "processed": 0, "failed": 0},
            "fake": {"total": 0, "processed": 0, "failed": 0}
        }
    
    def process_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Process a single image and extract landmarks.
        
        Args:
            image_path: Path to input image
            output_path: Path to save landmarks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return False
            
            # Extract landmarks
            landmarks = self.face_extractor.extract_landmarks_from_frame(image)
            
            if landmarks is not None:
                # Save landmarks as numpy array
                np.save(output_path, landmarks)
                return True
            else:
                self.logger.warning(f"No face detected in image: {image_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_category(self, category: str) -> None:
        """
        Process all images in a category (real or fake).
        
        Args:
            category: Category name ('real' or 'fake')
        """
        input_category_dir = self.input_dir / category
        output_category_dir = self.output_dir / category
        
        if not input_category_dir.exists():
            self.logger.error(f"Input directory does not exist: {input_category_dir}")
            return
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in input_category_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        self.stats[category]["total"] = len(image_files)
        
        self.logger.info(f"Processing {len(image_files)} {category} images...")
        
        # Process images with progress bar
        with tqdm(image_files, desc=f"Processing {category} images") as pbar:
            for image_file in pbar:
                # Create output filename (replace extension with .npy)
                output_filename = image_file.stem + ".npy"
                output_path = output_category_dir / output_filename
                
                # Skip if already processed
                if output_path.exists():
                    self.stats[category]["processed"] += 1
                    pbar.set_postfix({"Processed": self.stats[category]["processed"], 
                                    "Failed": self.stats[category]["failed"]})
                    continue
                
                # Process image
                success = self.process_image(image_file, output_path)
                
                if success:
                    self.stats[category]["processed"] += 1
                else:
                    self.stats[category]["failed"] += 1
                
                # Update progress bar
                pbar.set_postfix({"Processed": self.stats[category]["processed"], 
                                "Failed": self.stats[category]["failed"]})
    
    def process_all(self) -> None:
        """Process all categories in the dataset."""
        start_time = time.time()
        
        self.logger.info("Starting dataset processing...")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Process each category
        for category in ["real", "fake"]:
            self.process_category(category)
        
        processing_time = time.time() - start_time
        
        # Print final statistics
        self.print_statistics(processing_time)
    
    def print_statistics(self, processing_time: float) -> None:
        """Print processing statistics."""
        self.logger.info("\n" + "="*60)
        self.logger.info("DATASET PROCESSING COMPLETE")
        self.logger.info("="*60)
        
        total_processed = 0
        total_failed = 0
        total_images = 0
        
        for category in ["real", "fake"]:
            stats = self.stats[category]
            total_processed += stats["processed"]
            total_failed += stats["failed"]
            total_images += stats["total"]
            
            success_rate = (stats["processed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            
            self.logger.info(f"\n{category.upper()} Images:")
            self.logger.info(f"  Total images: {stats['total']}")
            self.logger.info(f"  Successfully processed: {stats['processed']}")
            self.logger.info(f"  Failed to process: {stats['failed']}")
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        overall_success_rate = (total_processed / total_images * 100) if total_images > 0 else 0
        
        self.logger.info(f"\nOVERALL SUMMARY:")
        self.logger.info(f"  Total images: {total_images}")
        self.logger.info(f"  Successfully processed: {total_processed}")
        self.logger.info(f"  Failed to process: {total_failed}")
        self.logger.info(f"  Overall success rate: {overall_success_rate:.1f}%")
        self.logger.info(f"  Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"  Average time per image: {processing_time/total_images:.3f} seconds")
        
        self.logger.info(f"\nLandmarks saved to:")
        self.logger.info(f"  Real: {self.output_dir / 'real'}")
        self.logger.info(f"  Fake: {self.output_dir / 'fake'}")
    
    def verify_output(self) -> None:
        """Verify the processed output."""
        self.logger.info("\nVerifying processed data...")
        
        for category in ["real", "fake"]:
            output_dir = self.output_dir / category
            npy_files = list(output_dir.glob("*.npy"))
            
            if npy_files:
                # Check a sample file
                sample_file = npy_files[0]
                landmarks = np.load(sample_file)
                
                self.logger.info(f"{category.upper()} landmarks:")
                self.logger.info(f"  Files created: {len(npy_files)}")
                self.logger.info(f"  Sample shape: {landmarks.shape}")
                self.logger.info(f"  Expected shape: (468, 3)")
                self.logger.info(f"  Sample file: {sample_file.name}")
                
                # Verify shape
                if landmarks.shape == (468, 3):
                    self.logger.info(f"  ✅ Shape verification passed")
                else:
                    self.logger.warning(f"  ❌ Shape verification failed!")
            else:
                self.logger.warning(f"  ❌ No .npy files found for {category}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'face_extractor'):
            self.face_extractor.close()


def main():
    """Main function to run the dataset processing."""
    
    # Setup paths
    project_root = Path(__file__).parent
    input_dir = project_root / "data" / "sample_dataset"
    output_dir = project_root / "data" / "processed"
    
    # Verify input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Verify real and fake subdirectories exist
    if not (input_dir / "real").exists() or not (input_dir / "fake").exists():
        print(f"Error: Required subdirectories 'real' and 'fake' not found in {input_dir}")
        return
    
    print("🚀 Starting Face Landmark Extraction for Dataset")
    print(f"📁 Input: {input_dir}")
    print(f"💾 Output: {output_dir}")
    print("-" * 60)
    
    try:
        # Initialize processor
        processor = DatasetProcessor(str(input_dir), str(output_dir))
        
        # Process all images
        processor.process_all()
        
        # Verify output
        processor.verify_output()
        
        # Cleanup
        processor.cleanup()
        
        print("\n🎉 Dataset processing completed successfully!")
        print(f"📊 Processed landmarks are ready for training")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        if 'processor' in locals():
            processor.cleanup()
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if 'processor' in locals():
            processor.cleanup()
        raise


if __name__ == "__main__":
    main()
