"""
AI Air Whiteboard — main entry point
=====================================

Architecture
------------
  GestureAgent   →  reads webcam, runs MediaPipe, classifies gestures
  CanvasAgent    →  owns the drawing canvas, renders palette/UI
  AIAgent        →  shape detection, OCR, smoothing
  ControlAgent   →  voice, export, confirmation dialogs

Usage
-----
  python main.py [--width W] [--height H] [--cam INDEX]

Keyboard shortcuts (also shown in the bottom bar while running)
---------------------------------------------------------------
  Z / Y   – undo / redo
  S       – save PNG
  P       – save PDF
  C       – toggle camera preview
  A       – auto-correct shapes
  O       – run OCR (requires pytesseract)
  M       – smooth canvas
  + / -   – increase / decrease brush size
  Q / Esc – quit
"""

from __future__ import annotations

import argparse
import sys
import os

import cv2
import numpy as np

# Make sure project root is on path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from agents.gesture_agent import GestureAgent
from agents.canvas_agent  import CanvasAgent
from agents.ai_agent      import AIAgent
from agents.control_agent import ControlAgent


# ─────────────────────────────────────────────────────────────────────────────
#  Screen resolution detection
# ─────────────────────────────────────────────────────────────────────────────

def _screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1280, 720


# ─────────────────────────────────────────────────────────────────────────────
#  Camera helper
# ─────────────────────────────────────────────────────────────────────────────

def _open_camera(index: int, w: int, h: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ─────────────────────────────────────────────────────────────────────────────
#  Camera preview rendering
# ─────────────────────────────────────────────────────────────────────────────

PREVIEW_W = 240
PREVIEW_H = 180
PREVIEW_MARGIN = 12


def _draw_camera_preview(
    frame: np.ndarray,
    preview_src: np.ndarray,
    screen_w: int,
    screen_h: int,
):
    """Paste the resized camera preview into the bottom-right corner."""
    preview = cv2.resize(preview_src, (PREVIEW_W, PREVIEW_H))
    px = screen_w  - PREVIEW_W  - PREVIEW_MARGIN
    py = screen_h  - PREVIEW_H  - 38            # 38 px above hint bar

    # Blend slightly (cam preview is opaque by default)
    frame[py : py + PREVIEW_H, px : px + PREVIEW_W] = preview

    # Border + label
    cv2.rectangle(
        frame,
        (px - 2, py - 2),
        (px + PREVIEW_W + 2, py + PREVIEW_H + 2),
        (60, 60, 70), 1,
    )
    cv2.putText(
        frame, "CAM",
        (px + 4, py + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    screen_w = args.width
    screen_h = args.height

    if screen_w == 0 or screen_h == 0:
        screen_w, screen_h = _screen_size()

    print(f"[Air Whiteboard] Resolution: {screen_w}×{screen_h}")

    # ── Init agents ───────────────────────────────────────────────────────────
    gesture_agent = GestureAgent(max_hands=2)
    ai_agent      = AIAgent()
    canvas_agent  = CanvasAgent(screen_w, screen_h, ai_agent=ai_agent)
    control_agent = ControlAgent(canvas_agent, ai_agent)

    caps = ai_agent.capabilities()
    print(f"[Air Whiteboard] AI caps  -- shapes: OK  OCR: {'OK' if caps['ocr'] else 'NO (install pytesseract)'}  smooth: OK")
    print(f"[Air Whiteboard] Voice    -- {'OK enabled' if control_agent._voice_enabled else 'NO (install SpeechRecognition + pyaudio)'}")

    # ── Camera ────────────────────────────────────────────────────────────────
    try:
        cap = _open_camera(args.cam, 1280, 720)
    except RuntimeError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    # ── Window ────────────────────────────────────────────────────────────────
    WIN = "AI Air Whiteboard"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("[Air Whiteboard] Running — press Q or Esc to quit.")
    print("  Gestures: 1-finger=Draw  2-fingers=Erase  3-fingers=OCR-Scan  Open-palm=Clear  Both-open=Zoom")

    ocr_hold_counter = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[Warning] Frame grab failed — retrying…")
            continue

        # Mirror so left/right match user expectation
        frame = cv2.flip(raw_frame, 1)

        # Resize camera feed to match canvas dimensions
        cam_frame = cv2.resize(frame, (screen_w, screen_h))

        # ── Gesture processing ────────────────────────────────────────────────
        gesture_info = gesture_agent.process(cam_frame)
        gesture = gesture_info.get("gesture", "idle")

        # ── Zoom gesture handling ────────────────────────────────────────────
        if gesture == "zoom":
            c1 = gesture_info.get("cursor")
            c2 = gesture_info.get("cursor2")
            if c1 and c2:
                dist = np.hypot(c2[0] - c1[0], c2[1] - c1[1])
                mid  = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                if canvas_agent._zoom_initial_dist is None:
                    canvas_agent.start_zoom(dist)
                else:
                    canvas_agent.update_zoom(dist, midpoint=mid)
        else:
            if canvas_agent._zoom_initial_dist is not None:
                canvas_agent.end_zoom()

        # ── OCR gesture handling (3 fingers held ~1 sec) ─────────────────────
        if gesture == "ocr":
            ocr_hold_counter += 1
            if ocr_hold_counter == 30:     # ~1 second at 30fps
                canvas_agent.scan_canvas_ocr()
        else:
            ocr_hold_counter = 0

        # ── Canvas update (skip during confirmation, zoom, or OCR) ───────────
        if not control_agent.pending_action and gesture not in ("zoom", "ocr"):
            canvas_agent.update(gesture_info)

        # ── Compose result frame ──────────────────────────────────────────────
        result = canvas_agent.get_canvas()

        # Camera preview overlay
        if control_agent.show_camera:
            preview_src = (
                gesture_agent.annotated_frame
                if gesture_agent.annotated_frame is not None
                else cam_frame
            )
            _draw_camera_preview(result, preview_src, screen_w, screen_h)

        # UI (palette, cursor, hints, gesture badge)
        canvas_agent.draw_ui(result, gesture_info)

        # Confirmation popup / OCR overlay
        control_agent.draw_overlay(result)

        cv2.imshow(WIN, result)

        # ── Keyboard input ────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):       # Q or Esc → quit
            break
        elif key == ord("z"):
            canvas_agent.undo()
        elif key == ord("y"):
            canvas_agent.redo()
        elif key in (ord("+"), ord("=")):
            canvas_agent.brush_size = min(60, canvas_agent.brush_size + 2)
            canvas_agent.set_status(f"Brush: {canvas_agent.brush_size}", 45)
        elif key == ord("-"):
            canvas_agent.brush_size = max(2, canvas_agent.brush_size - 2)
            canvas_agent.set_status(f"Brush: {canvas_agent.brush_size}", 45)
        elif key == ord("a"):
            control_agent.run_auto_shape()
        elif key == ord("x"):
            # Toggle eraser size
            canvas_agent.eraser_size = 60 if canvas_agent.eraser_size == 32 else 32
            canvas_agent.set_status(f"Eraser: {canvas_agent.eraser_size}px", 45)
        elif key == ord("t"):
            canvas_agent.auto_shape_enabled = not canvas_agent.auto_shape_enabled
            state = "ON" if canvas_agent.auto_shape_enabled else "OFF"
            canvas_agent.set_status(f"🔷 Auto-Shape: {state}", 80)
        elif key == ord("r"):
            canvas_agent.reset_zoom()
        elif key == ord("f"):
            # Temporarily exit fullscreen so file dialog is visible
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            canvas_agent.load_file_dialog()
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        control_agent.process(gesture_info, key)

    cap.release()
    cv2.destroyAllWindows()
    print("[Air Whiteboard] Exited cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Air Whiteboard")
    parser.add_argument("--width",  type=int, default=0,
                        help="Canvas width  (0 = auto-detect screen)")
    parser.add_argument("--height", type=int, default=0,
                        help="Canvas height (0 = auto-detect screen)")
    parser.add_argument("--cam",    type=int, default=0,
                        help="OpenCV camera index (default 0)")
    main(parser.parse_args())
