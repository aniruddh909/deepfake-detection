# 🕵 Real-Time Deepfake Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive real-time deepfake detection system that uses facial landmark analysis and CNN-based deep learning to identify manipulated videos and live webcam feeds.

##  Features

-  **Real-time Detection**: Live webcam analysis with instant results
-  **Video Processing**: Upload and analyze video files for deepfake content
-  **Advanced AI**: CNN-based model with 61%+ accuracy using facial landmark patterns
-  **Web Interface**: User-friendly Flask web application
-  **High Performance**: Optimized for real-time processing with MediaPipe
-  **Detailed Analysis**: Confidence scores and frame-by-frame results

##  Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Face Detection  │───▶│ Landmark Extract│
│ (Webcam/File)   │    │   (MediaPipe)    │    │ (468 points)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prediction    │◀───│   CNN Model      │◀───│ Feature Extract │
│ (Real/Fake)     │    │ (Hybrid Arch)    │    │ + Normalization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Set up the environment:**

   ```bash
   ./setup.sh
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

### Usage

#### Web Interface (Recommended)

```bash
# Start the web application
python main.py web --model data/models/latest_model.h5

# Open browser to: http://localhost:5000
```

#### Command Line Detection

```bash
# Webcam detection
python main.py detect --model data/models/latest_model.h5

# Video file detection
python main.py detect --model data/models/latest_model.h5 --input video.mp4 --output result.mp4
```

#### Training Custom Models

```bash
# Train on your dataset
python main.py train --data /path/to/dataset --epochs 100

# Train lightweight model
python main.py train --data /path/to/dataset --lightweight
```

## 📊 Model Performance

- **Accuracy**: 61.15% on validation set
- **Architecture**: Hybrid CNN + Dense layers
- **Input**: 468 facial landmarks (1,452 features)
- **Training**: Enhanced geometric features + MinMax normalization
- **Speed**: Real-time processing at 30+ FPS

## 🔧 Technical Details

### Dependencies

- **Python**: 3.8+
- **TensorFlow**: 2.10+
- **MediaPipe**: Latest
- **OpenCV**: 4.8+
- **Flask**: Web interface
- **NumPy, scikit-learn**: Data processing

### Model Architecture

- **Input**: Facial landmarks (478 points × 3 coordinates)
- **Features**: Enhanced with geometric relationships
- **Architecture**: Hybrid CNN + Dense layers
- **Output**: Binary classification (Real/Fake)

### Data Format

```
dataset/
├── real/
│   ├── real_video1_landmarks.npy
│   └── real_video2_landmarks.npy
└── fake/
    ├── fake_video1_landmarks.npy
    └── fake_video2_landmarks.npy
```

## 🛠️ System Requirements

- **OS**: macOS, Linux, Windows
- **RAM**: 8GB+ recommended
- **GPU**: Optional (recommended for training)
- **Webcam**: Required for real-time detection
- **Python**: 3.8 or higher

## 📁 Project Structure

```
deepfake-detection/
├── main.py                 # Main entry point
├── setup.sh               # Setup script
├── requirements.txt       # Dependencies
├── src/                   # Source code
│   ├── detection/         # Real-time detection engine
│   ├── models/           # CNN model architecture
│   ├── preprocessing/    # Data processing & feature extraction
│   ├── utils/           # Configuration & utilities
│   └── web/             # Flask web application
├── data/                # Data storage (gitignored)
│   ├── models/          # Trained models
│   ├── processed/       # Processed datasets
│   └── uploads/         # Web uploads
└── docs/               # Documentation
```

##  Development

### Running Tests

```bash
python -m pytest tests/
```

## ⚠️ Disclaimer

This system is designed for research and educational purposes. For production use, additional security and performance optimizations may be required. Results may vary depending on video quality and deepfake sophistication.

