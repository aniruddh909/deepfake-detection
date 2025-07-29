# Real-Time Deepfake Detection System

A comprehensive system for detecting deepfakes in real-time using MediaPipe Face Mesh and Conv1D CNN.

## 🎯 Project Overview

This project implements a real-time deepfake detection system that:

- Extracts 3D face landmarks using MediaPipe Face Mesh
- Uses a Conv1D CNN model trained on landmark coordinates
- Provides real-time detection via webcam or video upload
- Offers a web interface built with Flask

## 🏗️ Project Structure

```
Deepfake_detection/
├── docs/                       # Documentation
│   ├── activity.md             # Development activity log
│   └── README.md               # Project documentation
├── src/                        # Source code
│   ├── models/                 # Model definitions and training
│   │   ├── __init__.py
│   │   ├── cnn_model.py        # Conv1D CNN architecture
│   │   └── trainer.py          # Model training utilities
│   ├── preprocessing/          # Data preprocessing
│   │   ├── __init__.py
│   │   ├── face_extractor.py   # MediaPipe face landmark extraction
│   │   └── data_loader.py      # Data loading and preprocessing
│   ├── detection/              # Real-time detection
│   │   ├── __init__.py
│   │   ├── detector.py         # Main detection engine
│   │   └── video_processor.py  # Video processing utilities
│   ├── web/                    # Flask web application
│   │   ├── __init__.py
│   │   ├── app.py              # Flask application
│   │   ├── templates/          # HTML templates
│   │   └── static/             # CSS, JS, and assets
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config.py           # Configuration settings
│       └── helpers.py          # Helper functions
├── data/                       # Data storage
│   ├── raw/                    # Raw video data
│   ├── processed/              # Processed landmarks
│   └── models/                 # Saved model files
├── notebooks/                  # Jupyter notebooks for experimentation
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── main.py                     # Entry point
```

## 🚀 Quick Start

1. **Setup Environment**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Web Interface**

   ```bash
   python main.py
   ```

3. **Train Model (Optional)**
   ```bash
   python -m src.models.trainer
   ```

## 🔧 Configuration

See `src/utils/config.py` for configuration options.

## 📊 Model Architecture

- **Input:** MediaPipe Face Mesh landmarks (468 3D points)
- **Architecture:** Conv1D CNN with temporal processing
- **Output:** Binary classification (Real/Fake)

## 🤝 Contributing

This project follows modular development practices with clear separation of concerns.
