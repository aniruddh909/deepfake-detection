"""
Flask web application for deepfake detection.
Provides a web interface for uploading videos and real-time webcam detection.
"""

import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from werkzeug.utils import secure_filename
import tempfile
import threading
import time
from pathlib import Path
import logging
from typing import Optional

# Add the parent directory to the path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, src_dir)

try:
    from src.detection.detector import DeepfakeDetector
    from src.utils.config import WEB_CONFIG, MODELS_DIR
    from src.utils.helpers import setup_logging, validate_video_file, get_video_info
except ImportError:
    # Fallback for when running directly
    from detection.detector import DeepfakeDetector
    from utils.config import WEB_CONFIG, MODELS_DIR
    from utils.helpers import setup_logging, validate_video_file, get_video_info


class DeepfakeWebApp:
    """
    Flask web application for deepfake detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the web application.
        
        Args:
            model_path: Path to the trained model
        """
        self.app = Flask(__name__)
        self.logger = setup_logging()
        
        # Configure Flask app
        self.app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG["max_content_length"]
        self.app.config['UPLOAD_FOLDER'] = WEB_CONFIG["upload_folder"]
        
        # Create upload folder if it doesn't exist
        Path(WEB_CONFIG["upload_folder"]).mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self.detector = DeepfakeDetector(model_path)
        
        # Webcam state
        self.webcam_active = False
        self.webcam_thread = None
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("DeepfakeWebApp initialized")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html')
        
        @self.app.route('/upload', methods=['GET', 'POST'])
        def upload_video():
            """Handle video upload and processing."""
            if request.method == 'GET':
                return render_template('upload.html')
            
            # Check if file was uploaded
            if 'video' not in request.files:
                return jsonify({'error': 'No video file uploaded'}), 400
            
            file = request.files['video']
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            
            # Validate file extension
            if not self._allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
            
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = str(int(time.time()))
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Validate video file
                if not validate_video_file(file_path):
                    os.remove(file_path)
                    return jsonify({'error': 'Invalid video file'}), 400
                
                # Get video info
                video_info = get_video_info(file_path)
                
                # Process video
                self.logger.info(f"Processing uploaded video: {filename}")
                results = self.detector.detect_video_file(file_path)
                
                # Generate summary
                summary = self.detector.get_detection_summary(results)
                
                # Clean up uploaded file
                os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'video_info': video_info,
                    'summary': summary,
                    'total_frames': len(results),
                    'frames_analyzed': summary.get('frames_with_prediction', 0)
                })
            
            except Exception as e:
                self.logger.error(f"Error processing video: {e}")
                # Clean up file if it exists
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        @self.app.route('/webcam')
        def webcam_page():
            """Webcam detection page."""
            return render_template('webcam.html')
        
        @self.app.route('/start_webcam')
        def start_webcam():
            """Start webcam detection."""
            if self.webcam_active:
                return jsonify({'status': 'already_active'})
            
            try:
                self.webcam_active = True
                self.detector.reset_detection_state()
                return jsonify({'status': 'started'})
            except Exception as e:
                self.webcam_active = False
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/stop_webcam')
        def stop_webcam():
            """Stop webcam detection."""
            self.webcam_active = False
            return jsonify({'status': 'stopped'})
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route."""
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/status')
        def get_status():
            """Get application status."""
            return jsonify({
                'model_loaded': self.detector.is_model_loaded,
                'webcam_active': self.webcam_active,
                'scaler_fitted': self.detector.data_loader.is_fitted
            })
        
        @self.app.route('/api/model/load', methods=['POST'])
        def load_model():
            """Load a model file."""
            data = request.get_json()
            model_path = data.get('model_path')
            scaler_path = data.get('scaler_path')
            
            if not model_path:
                return jsonify({'error': 'Model path required'}), 400
            
            try:
                self.detector.load_model(model_path, scaler_path)
                return jsonify({'success': True, 'message': 'Model loaded successfully'})
            except Exception as e:
                return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return ('.' in filename and 
                Path(filename).suffix.lower() in WEB_CONFIG["allowed_extensions"])
    
    def _generate_frames(self):
        """Generate frames for video streaming."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.logger.error("Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while self.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect deepfake
                detection_result = self.detector.detect_frame(frame)
                
                # Annotate frame
                annotated_frame = self.detector.annotate_frame(frame, detection_result)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        finally:
            cap.release()
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None, debug: Optional[bool] = None):
        """
        Run the Flask application.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        host = host or WEB_CONFIG["host"]
        port = port or WEB_CONFIG["port"]
        debug = debug if debug is not None else WEB_CONFIG["debug"]
        
        self.logger.info(f"Starting web application on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


# Create Flask templates
def create_templates():
    """Create HTML templates for the web application."""
    
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Base template
    base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Deepfake Detection System{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .alert-container { position: fixed; top: 20px; right: 20px; z-index: 1050; }
        .video-container { max-width: 640px; margin: 0 auto; }
        .detection-info { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">🕵️ Deepfake Detector</a>
            <div class="navbar-nav">
                <a class="nav-link" href="{{ url_for('index') }}">Home</a>
                <a class="nav-link" href="{{ url_for('upload_video') }}">Upload Video</a>
                <a class="nav-link" href="{{ url_for('webcam_page') }}">Webcam</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <div class="alert-container" id="alertContainer"></div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function showAlert(message, type = 'info') {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            document.getElementById('alertContainer').appendChild(alertDiv);
            
            setTimeout(() => alertDiv.remove(), 5000);
        }
    </script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""
    
    # Index template
    index_template = """
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="text-center mb-5">
            <h1 class="display-4">🕵️ Real-Time Deepfake Detection</h1>
            <p class="lead">Detect deepfakes in videos and real-time webcam streams using AI</p>
        </div>
        
        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">📹 Upload Video</h5>
                        <p class="card-text">Upload a video file to analyze for deepfake content</p>
                        <a href="{{ url_for('upload_video') }}" class="btn btn-primary">Upload Video</a>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <h5 class="card-title">📷 Live Detection</h5>
                        <p class="card-text">Real-time deepfake detection using your webcam</p>
                        <a href="{{ url_for('webcam_page') }}" class="btn btn-success">Start Webcam</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-body">
                <h5 class="card-title">🧠 How It Works</h5>
                <ul>
                    <li><strong>Face Detection:</strong> Uses MediaPipe to extract 3D face landmarks</li>
                    <li><strong>Temporal Analysis:</strong> Analyzes sequences of frames using Conv1D CNN</li>
                    <li><strong>Real-time Processing:</strong> Provides instant feedback on video streams</li>
                    <li><strong>High Accuracy:</strong> Trained on landmark coordinate patterns</li>
                </ul>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""
    
    # Upload template
    upload_template = """
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-8">
        <h2>📹 Upload Video for Analysis</h2>
        
        <div class="card">
            <div class="card-body">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="videoFile" class="form-label">Select Video File</label>
                        <input type="file" class="form-control" id="videoFile" name="video" 
                               accept=".mp4,.avi,.mov,.mkv,.webm" required>
                        <div class="form-text">Supported formats: MP4, AVI, MOV, MKV, WEBM (Max: 16MB)</div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary" id="uploadBtn">
                        <span class="spinner-border spinner-border-sm d-none" id="uploadSpinner"></span>
                        Analyze Video
                    </button>
                </form>
            </div>
        </div>
        
        <div id="results" class="mt-4" style="display: none;">
            <div class="card">
                <div class="card-header">
                    <h5>Analysis Results</h5>
                </div>
                <div class="card-body" id="resultsContent">
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('videoFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadSpinner = document.getElementById('uploadSpinner');
    
    if (!fileInput.files[0]) {
        showAlert('Please select a video file', 'warning');
        return;
    }
    
    formData.append('video', fileInput.files[0]);
    
    // Show loading state
    uploadBtn.disabled = true;
    uploadSpinner.classList.remove('d-none');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
            showAlert('Video analysis completed!', 'success');
        } else {
            showAlert(result.error || 'Upload failed', 'danger');
        }
    } catch (error) {
        showAlert('Error uploading file: ' + error.message, 'danger');
    } finally {
        uploadBtn.disabled = false;
        uploadSpinner.classList.add('d-none');
    }
});

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const contentDiv = document.getElementById('resultsContent');
    
    const fakeProbability = (result.summary.fake_probability * 100).toFixed(1);
    const avgPrediction = (result.summary.average_prediction * 100).toFixed(1);
    
    contentDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Video Information</h6>
                <ul class="list-unstyled">
                    <li><strong>Duration:</strong> ${result.video_info.duration.toFixed(1)}s</li>
                    <li><strong>FPS:</strong> ${result.video_info.fps.toFixed(1)}</li>
                    <li><strong>Dimensions:</strong> ${result.video_info.width}x${result.video_info.height}</li>
                    <li><strong>Total Frames:</strong> ${result.video_info.frame_count}</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6>Detection Results</h6>
                <ul class="list-unstyled">
                    <li><strong>Frames Analyzed:</strong> ${result.frames_analyzed}</li>
                    <li><strong>Face Detection Rate:</strong> ${(result.summary.face_detection_rate * 100).toFixed(1)}%</li>
                    <li><strong>Average Prediction:</strong> ${avgPrediction}%</li>
                    <li><strong>Fake Probability:</strong> <span class="badge ${fakeProbability > 50 ? 'bg-danger' : 'bg-success'}">${fakeProbability}%</span></li>
                </ul>
            </div>
        </div>
    `;
    
    resultsDiv.style.display = 'block';
}
</script>
{% endblock %}
"""
    
    # Webcam template
    webcam_template = """
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-10">
        <h2>📷 Real-Time Deepfake Detection</h2>
        
        <div class="video-container">
            <div class="card">
                <div class="card-body text-center">
                    <div id="videoPlaceholder" class="bg-light p-5 mb-3" style="display: block;">
                        <h5>Click "Start Detection" to begin</h5>
                        <p class="text-muted">Make sure your camera is connected and permissions are granted</p>
                    </div>
                    
                    <img id="videoFeed" src="" style="display: none; width: 100%; max-width: 640px;" />
                    
                    <div class="mt-3">
                        <button id="startBtn" class="btn btn-success">Start Detection</button>
                        <button id="stopBtn" class="btn btn-danger" style="display: none;">Stop Detection</button>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="detection-info">
            <h6>Detection Information</h6>
            <div class="row">
                <div class="col-md-4">
                    <small class="text-muted">Status:</small>
                    <div id="status">Inactive</div>
                </div>
                <div class="col-md-4">
                    <small class="text-muted">Model:</small>
                    <div id="modelStatus">Loading...</div>
                </div>
                <div class="col-md-4">
                    <small class="text-muted">Instructions:</small>
                    <div><small>Position your face in the camera view</small></div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
let isActive = false;

document.getElementById('startBtn').addEventListener('click', startDetection);
document.getElementById('stopBtn').addEventListener('click', stopDetection);

async function startDetection() {
    try {
        const response = await fetch('/start_webcam');
        const result = await response.json();
        
        if (result.status === 'started' || result.status === 'already_active') {
            isActive = true;
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('videoPlaceholder').style.display = 'none';
            document.getElementById('videoFeed').style.display = 'block';
            document.getElementById('videoFeed').src = '/video_feed';
            document.getElementById('status').textContent = 'Active';
            
            showAlert('Detection started!', 'success');
        } else {
            showAlert('Failed to start detection: ' + result.message, 'danger');
        }
    } catch (error) {
        showAlert('Error starting detection: ' + error.message, 'danger');
    }
}

async function stopDetection() {
    try {
        const response = await fetch('/stop_webcam');
        const result = await response.json();
        
        isActive = false;
        document.getElementById('startBtn').style.display = 'inline-block';
        document.getElementById('stopBtn').style.display = 'none';
        document.getElementById('videoPlaceholder').style.display = 'block';
        document.getElementById('videoFeed').style.display = 'none';
        document.getElementById('status').textContent = 'Inactive';
        
        showAlert('Detection stopped', 'info');
    } catch (error) {
        showAlert('Error stopping detection: ' + error.message, 'danger');
    }
}

// Check application status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        const modelStatus = status.model_loaded ? 'Loaded ✅' : 'Not loaded ❌';
        document.getElementById('modelStatus').textContent = modelStatus;
    } catch (error) {
        document.getElementById('modelStatus').textContent = 'Error checking status';
    }
}

// Check status on page load
checkStatus();
</script>
{% endblock %}
"""
    
    # Write templates to files
    with open(templates_dir / "base.html", "w") as f:
        f.write(base_template)
    
    with open(templates_dir / "index.html", "w") as f:
        f.write(index_template)
    
    with open(templates_dir / "upload.html", "w") as f:
        f.write(upload_template)
    
    with open(templates_dir / "webcam.html", "w") as f:
        f.write(webcam_template)


def main():
    """
    Example usage of the DeepfakeWebApp.
    """
    # Create templates
    create_templates()
    
    # Initialize web app with trained model
    model_path = str(MODELS_DIR / "latest_model.h5")
    app = DeepfakeWebApp(model_path=model_path)
    
    print("Starting Deepfake Detection Web Application...")
    print("Open your browser and go to: http://localhost:8080")
    
    # Run the app
    app.run()


if __name__ == "__main__":
    main()
