#!/bin/bash

# Setup script for Deepfake Detection System
# This script installs dependencies and initializes the project

echo "🚀 Setting up Deepfake Detection System..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,models,uploads}
mkdir -p logs
mkdir -p notebooks

# Initialize web templates
echo "🌐 Setting up web templates..."
python -c "
import sys
sys.path.append('src')
from src.web.app import create_templates
create_templates()
print('Web templates created!')
"

# Create a simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify installation.
"""

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct."""
    import os
    
    required_dirs = [
        'src/utils',
        'src/preprocessing', 
        'src/models',
        'src/detection',
        'src/web',
        'data',
        'logs'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing installation...")
    print()
    
    print("Testing imports:")
    imports_ok = test_imports()
    print()
    
    print("Testing project structure:")
    structure_ok = test_project_structure()
    print()
    
    if imports_ok and structure_ok:
        print("🎉 Installation test passed!")
        print()
        print("You can now run:")
        print("  python main.py web          # Start web interface")
        print("  python main.py setup        # Additional setup options")
    else:
        print("❌ Installation test failed!")
        print("Please check the error messages above.")
EOF

# Run installation test
echo "🧪 Testing installation..."
python test_installation.py

echo ""
echo "🎉 Setup completed!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start the web interface: python main.py web"
echo "  3. Open your browser to: http://localhost:5000"
echo ""
echo "For more options: python main.py --help"
