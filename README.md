# AI Air Whiteboard created by rushikesh farakate .

>  This project is not ready yet — currently working on it. Stay tuned for updates!**

A full-screen, gesture-controlled, AI-powered whiteboard built with Python, OpenCV, and MediaPipe.

---

## Features

| Category | Feature |
|---|---|
| **Gestures** | 1 finger → draw, 2 fingers → erase, open palm → clear (confirmed), 2 hands → save (confirmed) 
| **Drawing** | Smooth continuous lines (no dots), interpolated strokes, EMA-smoothed cursor 
| **Canvas** | Full-screen black canvas, 40-step undo/redo, 10-colour palette 
| **AI** | Auto shape correction (circle, rect, triangle, line), Tesseract OCR, Gaussian smoothing 
| **Voice** | "clear", "save", "undo", "redo", "camera", "ocr", "smooth" |
| **Export** | PNG and PDF export |
| **Multi-user** | Two hands draw simultaneously in different colours |
| **Camera** | Annotated hand-landmark preview in bottom-right (toggleable) |

---

## Project Structure

```
air_whiteboard/
├── main.py                    # Orchestrator & main loop
├── requirements.txt
├── setup.sh                   # Quick install script
│
├── agents/
│   ├── gesture_agent.py       # MediaPipe hand tracking + gesture classification
│   ├── canvas_agent.py        # Canvas, palette, undo/redo, rendering
│   ├── ai_agent.py            # Shape detection, OCR, smoothing
│   └── control_agent.py       # Voice, export, confirmation dialogs
│
└── utils/
    └── smoothing.py           # PointSmoother, interpolate_points, Catmull-Rom
```

---

## Installation

### Automatic (Linux / macOS)

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
python main.py
```

### Manual

```bash
pip install opencv-python mediapipe numpy
pip install fpdf2 pytesseract SpeechRecognition pyaudio   # optional

# Tesseract binary (for OCR)
# Linux:   sudo apt install tesseract-ocr
# macOS:   brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# PortAudio (for voice)
# Linux:   sudo apt install portaudio19-dev
# macOS:   brew install portaudio
```

---

## Running

```bash
python main.py                          # auto-detect screen size, camera 0
python main.py --width 1920 --height 1080
python main.py --cam 1                  # use second camera
```

---

## Gesture Reference

| Gesture | Action |
|---|---|
| ☝️  Index finger only | **Draw** |
| ✌️  Index + middle fingers | **Erase** |
| 🖐  Open palm (4+ fingers) | **Request clear** (hold 1.5 s to confirm) |
| 🙌  Two hands detected | **Request save** (hold 1.5 s to confirm) |

### Confirmation Dialogs

When a destructive gesture is held long enough, a confirmation popup appears:
- **Hold 1-finger** → YES (confirm)
- **Hold 2-fingers** → NO (cancel)
- **Enter / Esc** on keyboard also work

---

## Keyboard Shortcuts

| Key | Action |
| `Z` | Undo |
| `Y` | Redo |
| `S` | Save PNG |
| `P` | Save PDF |
| `C` | Toggle camera preview |
| `A` | Auto-correct shapes |
| `O` | Run OCR (requires pytesseract) |
| `M` | Smooth canvas |
| `+` / `-` | Brush size up / down |
| `X` | Toggle eraser size (32 / 60 px) |
| `Q` / `Esc` | Quit |



## Voice Commands

Requires `SpeechRecognition` and `pyaudio`.

| Say | Action |
|---|---|
| "clear" | Request clear |
| "save" | Save PNG |
| "save pdf" | Save PDF |
| "undo" | Undo last stroke |
| "redo" | Redo last undo |
| "camera" | Toggle camera preview |
| "ocr" / "read text" | Run OCR on canvas |
| "smooth" | Smooth canvas |


## Multi-User Drawing

When MediaPipe detects **two hands** both making a draw gesture, each hand draws in its own colour (user 0 uses the selected palette colour; user 1 defaults to lime green). The two-hands *save* gesture still works — it's triggered by the dominant gesture classification when both hands aren't in draw mode.

---

## Troubleshooting

| Problem | Fix |

| Camera not opening | Try `--cam 1` or `--cam 2` |
| Dotted lines | Update to latest version; ensure adequate lighting |
| Gestures not detected | Ensure good lighting; keep hand 40–80 cm from camera |
| Voice not working | `pip install SpeechRecognition pyaudio`; check microphone permissions |
| OCR poor results | Write larger; use white colour on black background |
| Left-side drawing off | Already fixed — frame is mirrored before landmark extraction |

