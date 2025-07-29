# 🚀 Quick Start Guide

## Installation

1. **Clone or navigate to the project directory:**

   ```bash
   cd /Users/aniruddh/Documents/Deepfake_detection
   ```

2. **Run the setup script:**

   ```bash
   ./setup.sh
   ```

   This will:

   - Create a virtual environment
   - Install all dependencies
   - Set up project directories
   - Create web templates
   - Test the installation

## Usage

### 1. Web Interface (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Start the web application
python main.py web --model data/models/latest_model.h5

# Open browser to: http://localhost:5000
```

### 2. Command Line Detection

```bash
# Webcam detection (requires trained model)
python main.py detect --model path/to/model.h5

# Video file detection
python main.py detect --model path/to/model.h5 --input video.mp4 --output result.mp4
```

### 3. Training a Model

```bash
# Train on your dataset
python main.py train --data /path/to/dataset --epochs 100

# Train lightweight model
python main.py train --data /path/to/dataset --lightweight
```

## Project Structure

```
Deepfake_detection/
├── main.py                    # Main entry point
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
├── src/                       # Source code
│   ├── utils/                 # Configuration & utilities
│   ├── preprocessing/         # Data preparation
│   ├── models/               # CNN model & training
│   ├── detection/            # Real-time detection
│   └── web/                  # Flask web app
├── data/                     # Data storage
├── docs/                     # Documentation
└── logs/                     # Application logs
```

## Web Interface Features

- **📹 Video Upload**: Analyze uploaded videos for deepfake content
- **📷 Live Detection**: Real-time webcam analysis
- **📊 Results Dashboard**: Detailed analysis results and confidence scores
- **⚡ Real-time Processing**: Frame-by-frame detection with temporal smoothing

## Command Line Options

```bash
# Start web application
python main.py web [--port 5000] [--host 0.0.0.0] [--model path/to/model.h5]

# Train model
python main.py train --data /path/to/dataset [--epochs 100] [--batch-size 32] [--lightweight]

# Run detection
python main.py detect --model path/to/model.h5 [--input video.mp4] [--output result.mp4]

# Setup utilities
python main.py setup [--templates]
```

## Training Data Format

Organize your training data as follows:

```
dataset/
├── real/
│   ├── real_video1_landmarks.npy
│   ├── real_video2_landmarks.npy
│   └── ...
└── fake/
    ├── fake_video1_landmarks.npy
    ├── fake_video2_landmarks.npy
    └── ...
```

Each `.npy` file should contain landmarks of shape `(num_frames, 468, 3)`.

## System Requirements

- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **GPU**: Optional but recommended for training
- **Webcam**: Required for real-time detection
- **OS**: macOS, Linux, Windows

## Troubleshooting

### Common Issues

1. **Camera not working**: Check camera permissions and ensure no other applications are using it
2. **Model not loading**: Verify model file path and format
3. **Dependencies failing**: Try upgrading pip: `pip install --upgrade pip`
4. **Out of memory**: Reduce batch size or use lightweight model

### Getting Help

1. Check the logs in the `logs/` directory
2. Run the installation test: `python test_installation.py`
3. Verify all dependencies: `pip list`

## Next Steps

1. **Test the web interface** by running `python main.py web`
2. **Try webcam detection** (note: requires a trained model)
3. **Prepare your dataset** for training custom models
4. **Explore the API** for integration into other applications

---

**Note**: This system is designed for research and educational purposes. For production use, additional security and performance optimizations may be required.
