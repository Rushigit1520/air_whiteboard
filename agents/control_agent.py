"""
Control & Interaction Agent
---------------------------
Handles all side-channel control:
  • Confirmation popups (gesture + keyboard) for destructive actions.
  • Export to PNG and PDF.
  • Voice commands in a background thread (requires SpeechRecognition + pyaudio).
  • Camera preview toggle.
  • OCR result display overlay.
"""

from __future__ import annotations

import os
import time
import threading
import cv2
import numpy as np

# ── Optional: PDF export ──────────────────────────────────────────────────────
try:
    from fpdf import FPDF  # type: ignore
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

# ── Optional: Voice commands ──────────────────────────────────────────────────
try:
    import speech_recognition as sr  # type: ignore
    _SR_OK = True
except ImportError:
    _SR_OK = False


class ControlAgent:
    CONFIRM_FRAMES   = 120    # ~4 s at 30 fps
    CONFIRM_HOLD     = 18     # frames to hold gesture for yes/no

    def __init__(self, canvas_agent, ai_agent=None):
        self.canvas_agent = canvas_agent
        self.ai_agent     = ai_agent

        # Confirmation state
        self.pending_action: str | None = None
        self._confirm_timer  = 0
        self._confirm_hold   = 0

        # Camera toggle
        self.show_camera = True

        # OCR overlay
        self._ocr_text       = ""
        self._ocr_frames     = 0

        # Gesture debounce for destructive gestures (clear / save)
        self._gesture_debounce: dict[str, int] = {}
        self._DEBOUNCE_FRAMES = 45  # must hold gesture N frames to trigger

        # Voice
        self._voice_enabled = _SR_OK
        if _SR_OK:
            self._start_voice_thread()

    # ─────────────────────────────────────────────────────────────────────────
    #  Voice
    # ─────────────────────────────────────────────────────────────────────────

    def _start_voice_thread(self):
        t = threading.Thread(target=self._voice_loop, daemon=True)
        t.start()

    def _voice_loop(self):
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except Exception:
            return
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower().strip()
                self._handle_voice(text)
            except Exception:
                pass

    def _handle_voice(self, text: str):
        if "clear" in text:
            self.request_action("clear")
        elif "save" in text and "pdf" in text:
            self.save_pdf()
        elif "save" in text:
            self.save_png()
        elif "undo" in text:
            self.canvas_agent.undo()
        elif "redo" in text:
            self.canvas_agent.redo()
        elif "camera" in text:
            self.show_camera = not self.show_camera
        elif "text" in text or "ocr" in text or "read" in text:
            self._run_ocr()
        elif "smooth" in text and self.ai_agent:
            self._run_smooth()

    # ─────────────────────────────────────────────────────────────────────────
    #  Confirmation flow
    # ─────────────────────────────────────────────────────────────────────────

    def request_action(self, action: str):
        """Queue a destructive action for confirmation."""
        if self.pending_action == action:
            return  # already pending
        self.pending_action   = action
        self._confirm_timer   = self.CONFIRM_FRAMES
        self._confirm_hold    = 0

    def _confirm_yes(self):
        if self.pending_action == "clear":
            self.canvas_agent.clear()
        elif self.pending_action == "save":
            self.save_png()
        self.pending_action = None

    def _confirm_no(self):
        self.canvas_agent.set_status("Cancelled", 50)
        self.pending_action = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Export
    # ─────────────────────────────────────────────────────────────────────────

    def save_png(self, path: str | None = None) -> str:
        if path is None:
            path = f"whiteboard_{int(time.time())}.png"
        cv2.imwrite(path, self.canvas_agent.canvas)
        self.canvas_agent.set_status(f"💾 Saved: {os.path.basename(path)}")
        return path

    def save_pdf(self, path: str | None = None) -> str:
        if path is None:
            path = f"whiteboard_{int(time.time())}.pdf"

        if not _PDF_OK:
            self.canvas_agent.set_status("⚠  fpdf2 not installed — PDF skipped")
            return ""

        tmp_png = f"_tmp_wb_{int(time.time())}.png"
        self.save_png(tmp_png)

        try:
            pdf = FPDF(orientation="L", unit="mm", format="A4")
            pdf.add_page()
            pdf.image(tmp_png, x=0, y=0, w=297, h=210)
            pdf.output(path)
            self.canvas_agent.set_status(f"📄 PDF saved: {os.path.basename(path)}")
        except Exception as exc:
            self.canvas_agent.set_status(f"PDF error: {exc}")
        finally:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)

        return path

    # ─────────────────────────────────────────────────────────────────────────
    #  AI helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_ocr(self):
        if self.ai_agent is None:
            return
        text = self.ai_agent.ocr_canvas(self.canvas_agent.canvas)
        self._ocr_text   = text[:120] if text else "(nothing recognised)"
        self._ocr_frames = 180

    def _run_smooth(self):
        if self.ai_agent is None:
            return
        self.canvas_agent._save_undo()
        self.canvas_agent.canvas = self.ai_agent.smooth_canvas(
            self.canvas_agent.canvas, ksize=5
        )
        self.canvas_agent.set_status("🖌  Canvas smoothed")

    def run_auto_shape(self):
        if self.ai_agent is None:
            return
        self.canvas_agent._save_undo()
        _, n = self.ai_agent.run_shape_correction(
            self.canvas_agent.canvas, self.canvas_agent.current_color
        )
        self.canvas_agent.set_status(f"🔷 Auto-corrected {n} shape(s)")

    # ─────────────────────────────────────────────────────────────────────────
    #  Main update
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, gesture_info: dict, key: int | None = None):
        """
        Called each frame.  Handles:
          - Counting hold-frames for destructive gestures.
          - Confirming / cancelling pending actions.
          - Keyboard shortcuts.
        """
        gesture = gesture_info.get("gesture", "idle")

        # ── Debounce destructive gestures ─────────────────────────────────────
        if not self.pending_action:
            for g in ("clear", "save"):
                if gesture == g:
                    self._gesture_debounce[g] = self._gesture_debounce.get(g, 0) + 1
                    if self._gesture_debounce[g] >= self._DEBOUNCE_FRAMES:
                        self.request_action(g)
                        self._gesture_debounce[g] = 0
                else:
                    self._gesture_debounce[g] = 0

        # ── Confirmation hold logic ───────────────────────────────────────────
        if self.pending_action:
            self._confirm_timer -= 1
            if self._confirm_timer <= 0:
                self._confirm_no()
            else:
                if gesture == "draw":
                    self._confirm_hold += 1
                    if self._confirm_hold >= self.CONFIRM_HOLD:
                        self._confirm_yes()
                        self._confirm_hold = 0
                elif gesture == "erase":
                    self._confirm_hold -= 1
                    if self._confirm_hold <= -self.CONFIRM_HOLD:
                        self._confirm_no()
                        self._confirm_hold = 0
                else:
                    self._confirm_hold = max(0, self._confirm_hold - 1)

        # ── Keyboard ──────────────────────────────────────────────────────────
        if key is None:
            return

        if key == ord("s"):
            self.save_png()
        elif key == ord("p"):
            self.save_pdf()
        elif key == ord("c"):
            self.show_camera = not self.show_camera
        elif key == ord("o"):
            self._run_ocr()
        elif key == ord("m"):
            self._run_smooth()
        elif key == 13:   # Enter → yes
            if self.pending_action:
                self._confirm_yes()
        elif key == 27:   # Esc → no / cancel
            if self.pending_action:
                self._confirm_no()

    # ─────────────────────────────────────────────────────────────────────────
    #  Overlay rendering
    # ─────────────────────────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray):
        """Draw confirmation popup and OCR overlay on frame in-place."""
        if self.pending_action:
            self._draw_confirm_popup(frame)
        if self._ocr_frames > 0:
            self._draw_ocr_overlay(frame)
            self._ocr_frames -= 1

    def _draw_confirm_popup(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        bw, bh = 540, 160
        bx = (w - bw) // 2
        by = (h - bh) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Border
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 120, 220), 2, cv2.LINE_AA)

        # Title
        title = f"Confirm  {self.pending_action.upper()}?"
        cv2.putText(frame, title, (bx + 20, by + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

        # Instructions
        cv2.putText(frame,
                    "Hold  1-finger (draw)  =  YES      Hold  2-fingers  =  NO",
                    (bx + 14, by + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "or press  Enter / Esc",
                    (bx + 14, by + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (130, 130, 150), 1)

        # Progress bar for auto-cancel
        ratio  = self._confirm_timer / self.CONFIRM_FRAMES
        bar_w  = int((bw - 28) * ratio)
        bar_y  = by + bh - 22
        cv2.rectangle(frame, (bx + 14, bar_y), (bx + 14 + bw - 28, bar_y + 10), (40, 40, 55), -1)
        bar_color = (0, 200, 255) if ratio > 0.3 else (0, 80, 255)
        cv2.rectangle(frame, (bx + 14, bar_y), (bx + 14 + bar_w, bar_y + 10), bar_color, -1)

        # Hold-gesture indicator
        if self._confirm_hold > 0:
            fill = int((bw - 28) * min(1.0, self._confirm_hold / self.CONFIRM_HOLD))
            cv2.rectangle(frame, (bx + 14, bar_y - 14), (bx + 14 + fill, bar_y - 4),
                          (50, 255, 80), -1)

    def _draw_ocr_overlay(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        lines = self._ocr_text.split("\n")
        by    = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, by - 30), (w - 10, by + 28 * len(lines) + 10),
                      (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "OCR Result:", (20, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (20, by + 26 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
