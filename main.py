"""
Main entry point for the Deepfake Detection System.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.web.app import DeepfakeWebApp, create_templates
from src.detection.detector import DeepfakeDetector
from src.models.trainer import ModelTrainer
from src.utils.config import MODELS_DIR
from src.utils.helpers import setup_logging


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="Real-Time Deepfake Detection System")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web application command
    web_parser = subparsers.add_parser("web", help="Start web application")
    web_parser.add_argument("--model", type=str, help="Path to trained model file")
    web_parser.add_argument("--port", type=int, default=5000, help="Port number")
    web_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training data directory")
    train_parser.add_argument("--output", type=str, help="Output directory for trained model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lightweight", action="store_true", help="Train lightweight model")
    
    # Detection command
    detect_parser = subparsers.add_parser("detect", help="Run detection on video or webcam")
    detect_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    detect_parser.add_argument("--input", type=str, help="Input video file (if not specified, uses webcam)")
    detect_parser.add_argument("--output", type=str, help="Output video file path")
    detect_parser.add_argument("--duration", type=int, default=60, help="Webcam detection duration (seconds)")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup the application")
    setup_parser.add_argument("--templates", action="store_true", help="Create web templates")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.command == "web":
        logger.info("Starting web application")
        
        # Create templates if they don't exist
        create_templates()
        
        # Initialize web app
        app = DeepfakeWebApp(model_path=args.model)
        
        print(f"Starting Deepfake Detection Web Application...")
        print(f"Open your browser and go to: http://{args.host}:{args.port}")
        
        # Run the app
        app.run(host=args.host, port=args.port, debug=args.debug)
    
    elif args.command == "train":
        logger.info("Starting model training")
        
        output_dir = args.output or str(MODELS_DIR)
        
        # Initialize trainer
        trainer = ModelTrainer(
            data_dir=args.data,
            output_dir=output_dir,
            lightweight=args.lightweight
        )
        
        # Train model
        trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        
        print(f"Training completed. Model saved to: {output_dir}")
    
    elif args.command == "detect":
        logger.info("Starting detection")
        
        # Initialize detector
        detector = DeepfakeDetector(model_path=args.model)
        
        if args.input:
            # Video file detection
            print(f"Processing video: {args.input}")
            results = detector.detect_video_file(args.input, args.output)
            summary = detector.get_detection_summary(results)
            
            print(f"Detection completed!")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Frames with face: {summary['frames_with_face']}")
            print(f"Fake probability: {summary.get('fake_probability', 0) * 100:.1f}%")
        else:
            # Webcam detection
            print(f"Starting webcam detection for {args.duration} seconds")
            print("Press 'q' to quit early")
            detector.detect_webcam(duration_seconds=args.duration)
    
    elif args.command == "setup":
        if args.templates:
            print("Creating web templates...")
            create_templates()
            print("Templates created successfully!")
        else:
            print("Setup completed. Use --templates to create web templates.")
    
    else:
        print("Welcome to the Real-Time Deepfake Detection System!")
        print()
        print("Available commands:")
        print("  web     - Start the web application")
        print("  train   - Train the detection model")
        print("  detect  - Run detection on video or webcam")
        print("  setup   - Setup the application")
        print()
        print("Use --help with any command for more options.")
        print()
        print("Quick start:")
        print("  python main.py web                    # Start web interface")
        print("  python main.py detect --model <path>  # Start webcam detection")
        print("  python main.py train --data <path>    # Train model")


if __name__ == "__main__":
    main()
