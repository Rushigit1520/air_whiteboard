#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  AI Air Whiteboard — Quick Setup Script
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "=== AI Air Whiteboard Setup ==="

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Core (always install)
pip install opencv-python mediapipe numpy

# Optional features
echo ""
echo "Installing optional packages (PDF, OCR, Voice)..."
pip install fpdf2 pytesseract SpeechRecognition || true

# pyaudio needs system libs
if command -v apt-get &>/dev/null; then
    echo "Detected apt — installing system audio/OCR libs..."
    sudo apt-get install -y portaudio19-dev python3-pyaudio tesseract-ocr libportaudio2 || true
    pip install pyaudio || true
elif command -v brew &>/dev/null; then
    echo "Detected Homebrew — installing system audio/OCR libs..."
    brew install portaudio tesseract || true
    pip install pyaudio || true
fi

echo ""
echo "=== Setup complete ==="
echo "Activate venv:  source .venv/bin/activate"
echo "Run:            python main.py"
echo "  Options:      python main.py --width 1920 --height 1080 --cam 0"
