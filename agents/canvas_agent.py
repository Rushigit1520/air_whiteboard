"""
Drawing & Canvas Agent
----------------------
Owns the whiteboard canvas (numpy array) and all rendering logic:
  • Full-screen black canvas.
  • Compact top palette bar (colors + brush-size controls).
  • Smooth continuous line drawing (EMA-smoothed + interpolated).
  • Per-stroke undo / redo stack.
  • Eraser mode.
  • Status message overlay.
  • Cursor visual indicator.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque

from utils.smoothing import interpolate_points


# ── Palette configuration ────────────────────────────────────────────────────
PALETTE_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("Red",    (0,   0,   255)),
    ("Green",  (0,   255,  0)),
    ("Blue",   (255,  0,   0)),
    ("Yellow", (0,   255, 255)),
    ("White",  (255, 255, 255)),
    ("Purple", (255,  0,  200)),
    ("Orange", (0,   165, 255)),
    ("Cyan",   (255, 255,  0)),
    ("Pink",   (147,  20, 255)),
    ("Lime",   (50,  205,  50)),
]

ERASER_COLOR = (0, 0, 0)
PALETTE_H    = 62          # pixels
CIRCLE_R     = 18
CIRCLE_STEP  = CIRCLE_R * 2 + 12
CIRCLE_CY    = PALETTE_H // 2


class CanvasAgent:
    MAX_UNDO = 40

    def __init__(self, width: int, height: int, ai_agent=None):
        self.width  = width
        self.height = height

        self.canvas: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

        self.current_color: tuple[int, int, int] = PALETTE_COLORS[0][1]
        self.brush_size:  int = 5
        self.eraser_size: int = 32

        # AI agent for auto-shape detection
        self.ai_agent = ai_agent
        self.auto_shape_enabled: bool = True   # toggle with 'T' key

        # Undo / Redo
        self._undo_stack: deque[np.ndarray] = deque(maxlen=self.MAX_UNDO)
        self._redo_stack: deque[np.ndarray] = deque(maxlen=self.MAX_UNDO)
        self._stroke_started = False  # save state once per stroke

        # Drawing state
        self.prev_point: tuple[int, int] | None = None

        # Stroke tracking for auto-shape
        self._stroke_points: list[tuple[int, int]] = []
        self._stroke_color: tuple[int, int, int] = self.current_color
        self._pre_stroke_canvas: np.ndarray | None = None

        # Precompute palette circle centers
        self._palette_cx: list[tuple[int, tuple[int, int, int], str]] = []
        self._compute_palette()

        # Brush-size button rects (x1, y1, x2, y2, delta)
        self._brush_btns: list[tuple[int, int, int, int, int]] = []
        self._compute_brush_btns()

        # Status overlay
        self._status_text   = ""
        self._status_frames = 0

        # Multi-user colours (indexed by hand slot)
        self._user_colors: dict[int, tuple[int, int, int]] = {
            0: self.current_color,
            1: (50, 200, 50),   # second user gets green
        }

        # Zoom state
        self.zoom_level: float = 1.0
        self.view_cx: float = width / 2.0    # view center in canvas coords
        self.view_cy: float = height / 2.0
        self._zoom_initial_dist: float | None = None
        self._zoom_initial_level: float = 1.0

        # OCR overlay state
        self._ocr_overlay: dict | None = None

        # Undo/Redo button cooldown (prevent rapid-fire)
        self._undo_redo_cooldown: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    #  Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_palette(self):
        """Precompute (cx, color, name) for each palette swatch."""
        self._palette_cx = []
        for i, (name, color) in enumerate(PALETTE_COLORS):
            cx = 14 + CIRCLE_R + i * CIRCLE_STEP
            self._palette_cx.append((cx, color, name))

    def _compute_brush_btns(self):
        """Brush ±  buttons, placed after last color swatch."""
        last_cx = self._palette_cx[-1][0] if self._palette_cx else 200
        bx = last_cx + CIRCLE_R + 30
        by = PALETTE_H // 2
        bsize = 20
        self._brush_btns = [
            (bx,          by - bsize, bx + bsize * 2, by + bsize, +2),  # '+'
            (bx + bsize * 2 + 10, by - bsize, bx + bsize * 4 + 10, by + bsize, -2),  # '−'
        ]
        # Undo / Redo buttons after brush buttons
        ubx = self._brush_btns[-1][2] + 20
        btn_w, btn_h = 48, 32
        self._undo_btn = (ubx, by - btn_h // 2, ubx + btn_w, by + btn_h // 2)
        self._redo_btn = (ubx + btn_w + 8, by - btn_h // 2, ubx + btn_w * 2 + 8, by + btn_h // 2)

    # ──────────────────────────────────────────────────────────────────────────
    #  Undo / Redo
    # ──────────────────────────────────────────────────────────────────────────

    def _save_undo(self):
        self._undo_stack.append(self.canvas.copy())
        self._redo_stack.clear()

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(self.canvas.copy())
            self.canvas = self._undo_stack.pop()
            self.set_status("↩  Undo")

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(self.canvas.copy())
            self.canvas = self._redo_stack.pop()
            self.set_status("↪  Redo")

    # ──────────────────────────────────────────────────────────────────────────
    #  Canvas operations
    # ──────────────────────────────────────────────────────────────────────────

    def clear(self):
        self._save_undo()
        self.canvas[:] = 0
        self.prev_point = None
        self.set_status("🗑  Canvas Cleared")

    def set_status(self, msg: str, frames: int = 100):
        self._status_text   = msg
        self._status_frames = frames

    # ──────────────────────────────────────────────────────────────────────────
    #  File loading (images & PDFs)
    # ──────────────────────────────────────────────────────────────────────────

    def load_file_dialog(self):
        """Open a file dialog and load the selected image or PDF onto the canvas.
        Runs synchronously — blocks until user picks a file or cancels."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            root.focus_force()

            filetypes = [
                ("All supported", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*"),
            ]
            path = filedialog.askopenfilename(
                title="Load Image or PDF onto Whiteboard",
                filetypes=filetypes,
            )
            root.destroy()

            if path:
                self._load_file(path)
            else:
                self.set_status("File load cancelled", 60)
        except Exception as e:
            self.set_status(f"File dialog error: {e}", 120)

    def _load_file(self, path: str):
        """Load an image or PDF and paste it onto the canvas."""
        import os
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext == ".pdf":
            img = self._load_pdf_page(path)
        else:
            img = cv2.imread(path)

        if img is None:
            self.set_status(f"Failed to load: {name}", 120)
            return

        self._save_undo()
        self._paste_image_on_canvas(img)
        self._loaded_file_name = name
        self.set_status(f"Loaded: {name}", 120)

    def _load_pdf_page(self, path: str, page_num: int = 0) -> np.ndarray | None:
        """Try to render the first page of a PDF as an image."""
        # Try PyMuPDF (fitz)
        try:
            import fitz  # type: ignore
            doc = fitz.open(path)
            page = doc.load_page(page_num)
            # Render at 2x for quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA → BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB → BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            doc.close()
            return img
        except ImportError:
            pass

        # Try pdf2image (poppler)
        try:
            from pdf2image import convert_from_path  # type: ignore
            images = convert_from_path(path, first_page=1, last_page=1, dpi=200)
            if images:
                from PIL import Image
                pil_img = images[0]
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        self.set_status("PDF needs: pip install PyMuPDF", 200)
        return None

    def _paste_image_on_canvas(self, img: np.ndarray):
        """Resize image to fit canvas (maintain aspect ratio) and center it."""
        ih, iw = img.shape[:2]
        ch, cw = self.height, self.width

        # Compute scale to fit
        scale = min(cw / iw, ch / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center on canvas
        x_off = (cw - new_w) // 2
        y_off = (ch - new_h) // 2

        self.canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # ──────────────────────────────────────────────────────────────────────────
    #  Palette interaction
    # ──────────────────────────────────────────────────────────────────────────

    def _hit_palette(self, cursor: tuple[int, int]) -> bool:
        x, y = cursor
        if y > PALETTE_H:
            return False

        for cx, color, name in self._palette_cx:
            if abs(x - cx) <= CIRCLE_R + 4:
                self.current_color = color
                self._user_colors[0] = color
                self.set_status(f"Color: {name}", 60)
                return True

        for x1, y1, x2, y2, delta in self._brush_btns:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.brush_size = max(2, min(60, self.brush_size + delta))
                self.set_status(f"Brush size: {self.brush_size}", 45)
                return True

        # Undo/Redo button hit detection (with cooldown)
        if self._undo_redo_cooldown <= 0:
            ux1, uy1, ux2, uy2 = self._undo_btn
            if ux1 <= x <= ux2 and uy1 <= y <= uy2:
                self.undo()
                self._undo_redo_cooldown = 15   # ~0.5s cooldown
                return True

            rx1, ry1, rx2, ry2 = self._redo_btn
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                self.redo()
                self._undo_redo_cooldown = 15
                return True

        return True  # absorb any click in palette zone

    # ──────────────────────────────────────────────────────────────────────────
    #  Drawing
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_stroke(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
    ):
        """Interpolated anti-aliased stroke so no dots appear at low frame rates."""
        pts = interpolate_points(p1, p2)
        for i in range(len(pts) - 1):
            cv2.line(self.canvas, pts[i], pts[i + 1], color, size, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    #  Zoom helpers
    # ──────────────────────────────────────────────────────────────────────────

    def screen_to_canvas(self, sx: int, sy: int) -> tuple[int, int]:
        """Convert screen coordinates to canvas coordinates (accounting for zoom)."""
        if self.zoom_level == 1.0:
            return sx, sy
        vis_w = self.width / self.zoom_level
        vis_h = self.height / self.zoom_level
        cx = self.view_cx - vis_w / 2 + sx / self.zoom_level
        cy = self.view_cy - vis_h / 2 + sy / self.zoom_level
        cx = max(0, min(self.width - 1, int(cx)))
        cy = max(0, min(self.height - 1, int(cy)))
        return cx, cy

    def start_zoom(self, dist: float):
        """Begin zoom gesture — record initial distance between hands."""
        self._zoom_initial_dist = dist
        self._zoom_initial_level = self.zoom_level

    def update_zoom(self, current_dist: float, midpoint: tuple[int, int] | None = None):
        """Update zoom level based on current distance between hands."""
        if self._zoom_initial_dist is None or self._zoom_initial_dist < 20:
            return
        ratio = current_dist / self._zoom_initial_dist
        new_zoom = self._zoom_initial_level * ratio
        self.zoom_level = max(0.5, min(5.0, new_zoom))

        # Optionally pan toward midpoint
        if midpoint is not None:
            target_cx, target_cy = self.screen_to_canvas(*midpoint)
            # Gently drift view center toward where user is looking
            self.view_cx += (target_cx - self.view_cx) * 0.05
            self.view_cy += (target_cy - self.view_cy) * 0.05
            # Clamp so we don't go out of bounds
            self._clamp_view()

    def end_zoom(self):
        """End zoom gesture."""
        self._zoom_initial_dist = None

    def reset_zoom(self):
        """Reset zoom to 1.0 and center the view."""
        self.zoom_level = 1.0
        self.view_cx = self.width / 2.0
        self.view_cy = self.height / 2.0
        self._zoom_initial_dist = None
        self.set_status("🔍  Zoom reset", 60)

    def _clamp_view(self):
        """Clamp view center so the visible region stays within the canvas."""
        vis_w = self.width / self.zoom_level / 2
        vis_h = self.height / self.zoom_level / 2
        self.view_cx = max(vis_w, min(self.width - vis_w, self.view_cx))
        self.view_cy = max(vis_h, min(self.height - vis_h, self.view_cy))

    # ──────────────────────────────────────────────────────────────────────────
    #  Main update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, gesture_info: dict):
        """
        Called every frame with the output of GestureAgent.process().
        Handles draw / erase for ALL detected hands (multi-user).
        """
        hands = gesture_info.get("hands", [])

        # Tick undo/redo button cooldown
        if self._undo_redo_cooldown > 0:
            self._undo_redo_cooldown -= 1

        # ── multi-user: iterate over all detected hands ───────────────────────
        primary_handled = False
        for slot_idx, hand in enumerate(hands[:2]):
            gesture = hand.gesture
            cursor  = hand.cursor

            if cursor is None:
                continue

            # Transform cursor to canvas coordinates if zoomed
            canvas_cursor = self.screen_to_canvas(*cursor) if self.zoom_level != 1.0 else cursor

            # Palette zone (primary hand only for colour selection)
            if slot_idx == 0 and cursor[1] < PALETTE_H and gesture in ("draw", "idle"):
                if gesture == "draw" and self.prev_point is None:
                    self._hit_palette(cursor)
                self.prev_point = None
                continue

            if gesture == "draw":
                color = self._user_colors.get(slot_idx, self.current_color)
                size  = self.brush_size

                if not self._stroke_started:
                    self._save_undo()
                    self._stroke_started = True
                    # Save canvas state before stroke for potential shape replacement
                    if slot_idx == 0 and self.auto_shape_enabled and self.ai_agent:
                        self._pre_stroke_canvas = self.canvas.copy()
                        self._stroke_points = []
                        self._stroke_color = color

                prev = self.prev_point if slot_idx == 0 else None
                if prev is not None:
                    self._draw_stroke(prev, canvas_cursor, color, size)
                else:
                    cv2.circle(self.canvas, canvas_cursor, max(1, size // 2), color, -1, cv2.LINE_AA)

                if slot_idx == 0:
                    self.prev_point = canvas_cursor
                    # Collect points for shape analysis
                    if self.auto_shape_enabled and self.ai_agent:
                        self._stroke_points.append(canvas_cursor)
                primary_handled = True

            elif gesture == "erase":
                if not self._stroke_started:
                    self._save_undo()
                    self._stroke_started = True

                prev = self.prev_point if slot_idx == 0 else None
                if prev is not None:
                    self._draw_stroke(prev, canvas_cursor, ERASER_COLOR, self.eraser_size)
                else:
                    cv2.circle(self.canvas, canvas_cursor, self.eraser_size // 2, ERASER_COLOR, -1)

                if slot_idx == 0:
                    self.prev_point = cursor

            else:
                if slot_idx == 0:
                    if self._stroke_started:
                        self._stroke_started = False
                        # ── Auto-shape: analyse completed stroke ──────────
                        self._try_auto_shape()
                    self.prev_point = None

    # ──────────────────────────────────────────────────────────────────────────
    #  Auto-shape: analyse stroke on completion
    # ──────────────────────────────────────────────────────────────────────────

    def _try_auto_shape(self):
        """If the stroke looks like a shape, replace freehand with perfect one.
        If not a shape, try OCR to recognize letters/digits."""
        if not self.auto_shape_enabled or self.ai_agent is None:
            self._stroke_points = []
            self._pre_stroke_canvas = None
            return

        if len(self._stroke_points) < 8:
            self._stroke_points = []
            self._pre_stroke_canvas = None
            return

        from agents.ai_agent import AIAgent
        result = AIAgent.classify_stroke(self._stroke_points)

        if result is not None and self._pre_stroke_canvas is not None:
            shape_name, params = result
            # Restore canvas to pre-stroke state (erasing freehand drawing)
            self.canvas = self._pre_stroke_canvas.copy()
            # Draw the perfect shape
            AIAgent.draw_perfect_shape(
                self.canvas, shape_name, params,
                self._stroke_color,
                thickness=max(2, self.brush_size),
            )
            emoji = {"circle": "⬤", "rectangle": "▬", "triangle": "△", "line": "╱"}
            self.set_status(
                f"{emoji.get(shape_name, '🔷')}  Auto → {shape_name.title()}", 80
            )
        elif self._pre_stroke_canvas is not None:
            # Not a shape → try letter/character recognition
            letter_result = self.ai_agent.recognize_letter(
                self.canvas, self._stroke_points
            )
            if letter_result is not None:
                _, letter_params = letter_result
                char = letter_params["char"]
                bbox = letter_params["bbox"]
                # Restore canvas to pre-stroke state (erase freehand)
                self.canvas = self._pre_stroke_canvas.copy()
                # Draw the clean letter
                AIAgent.draw_perfect_letter(
                    self.canvas, char, bbox,
                    self._stroke_color,
                    thickness=max(2, self.brush_size),
                )
                self.set_status(
                    f"🔤  Auto → \"{char}\"", 80
                )

        self._stroke_points = []
        self._pre_stroke_canvas = None

    def scan_canvas_ocr(self):
        """Scan entire canvas for text — triggered by 3-finger gesture.
        Only scans areas where drawings exist (non-black regions)."""
        if not self.ai_agent or not self.ai_agent.ocr_available:
            self.set_status("OCR not available (install pytesseract)", 120)
            return

        # Find non-black (drawn) region on canvas
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Find bounding box of all drawn content
        coords = cv2.findNonZero(mask)
        if coords is None:
            self.set_status("Nothing to scan — canvas is empty", 80)
            return

        x, y, w, h = cv2.boundingRect(coords)

        # Add padding
        pad = 40
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(self.width, x + w + pad)
        y2 = min(self.height, y + h + pad)

        roi = self.canvas[y1:y2, x1:x2]

        try:
            text = self.ai_agent.ocr_canvas(roi)
            text = text.strip()
            if text and any(c.isalnum() for c in text):
                self._ocr_overlay = {
                    "text": text,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "frames": 300,   # show for ~10 seconds
                }
                self.set_status(f"OCR: \"{text}\"", 150)
            else:
                self.set_status("No text recognised", 80)
        except Exception as e:
            self.set_status(f"OCR error: {e}", 120)

    # ──────────────────────────────────────────────────────────────────────────
    #  Rendering helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_canvas(self) -> np.ndarray:
        """Return the canvas view, applying zoom if needed."""
        if self.zoom_level <= 1.01 and self.zoom_level >= 0.99:
            return self.canvas.copy()

        self._clamp_view()

        vis_w = int(self.width / self.zoom_level)
        vis_h = int(self.height / self.zoom_level)

        x1 = int(self.view_cx - vis_w / 2)
        y1 = int(self.view_cy - vis_h / 2)

        # Clamp to canvas bounds
        x1 = max(0, min(x1, self.width - vis_w))
        y1 = max(0, min(y1, self.height - vis_h))
        x2 = min(x1 + vis_w, self.width)
        y2 = min(y1 + vis_h, self.height)

        roi = self.canvas[y1:y2, x1:x2]
        return cv2.resize(roi, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def draw_ui(self, frame: np.ndarray, gesture_info: dict) -> np.ndarray:
        """Overlay palette, cursor, status, and hints onto `frame` in-place."""
        self._draw_palette_bar(frame)
        self._draw_cursor(frame, gesture_info)
        self._draw_zoom_badge(frame, gesture_info)
        self._draw_ocr_overlay(frame)
        self._draw_status(frame)
        self._draw_hints(frame)
        self._draw_gesture_badge(frame, gesture_info)
        return frame

    def _draw_ocr_overlay(self, frame: np.ndarray):
        """Render the OCR text recognition overlay near the stroke area."""
        if self._ocr_overlay is None:
            return

        ov = self._ocr_overlay
        ov["frames"] -= 1
        if ov["frames"] <= 0:
            self._ocr_overlay = None
            return

        text = ov["text"]
        ox   = ov["x"]
        oy   = ov["y"] + ov["h"] + 10   # below the stroke

        # Clamp position
        oy = min(oy, self.height - 80)
        ox = max(10, min(ox, self.width - 300))

        # Measure text
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Split multi-line
        lines = text.split("\n")[:3]  # max 3 lines
        line_h = 30
        max_tw = 0
        for line in lines:
            tw, _ = cv2.getTextSize(line, font, 0.7, 2)[0]
            max_tw = max(max_tw, tw)

        panel_w = max_tw + 30
        panel_h = len(lines) * line_h + 20

        # Draw semi-transparent dark panel
        alpha = min(1.0, ov["frames"] / 30.0)  # fade out
        overlay = frame.copy()

        # Panel background
        cv2.rectangle(overlay,
                      (ox - 5, oy - 5),
                      (ox + panel_w, oy + panel_h),
                      (20, 40, 20), -1)
        # Green border
        cv2.rectangle(overlay,
                      (ox - 5, oy - 5),
                      (ox + panel_w, oy + panel_h),
                      (0, 220, 100), 2, cv2.LINE_AA)

        # "OCR" label
        cv2.putText(overlay, "OCR", (ox + 2, oy + 14),
                    font, 0.4, (0, 200, 100), 1)

        # Text lines
        for i, line in enumerate(lines):
            ty = oy + 20 + (i + 1) * line_h - 8
            cv2.putText(overlay, line, (ox + 8, ty),
                        font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_palette_bar(self, frame: np.ndarray):
        # Background
        cv2.rectangle(frame, (0, 0), (self.width, PALETTE_H), (22, 22, 28), -1)
        cv2.line(frame, (0, PALETTE_H), (self.width, PALETTE_H), (60, 60, 70), 1)

        # Color swatches
        for cx, color, _ in self._palette_cx:
            selected = (color == self.current_color)
            cv2.circle(frame, (cx, CIRCLE_CY), CIRCLE_R, color, -1, cv2.LINE_AA)
            border_color = (255, 255, 255) if selected else (80, 80, 90)
            border_thick = 3 if selected else 1
            cv2.circle(frame, (cx, CIRCLE_CY), CIRCLE_R, border_color, border_thick, cv2.LINE_AA)
            if selected:
                cv2.circle(frame, (cx, CIRCLE_CY), CIRCLE_R + 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Brush size label + buttons
        for x1, y1, x2, y2, delta in self._brush_btns:
            label = "+" if delta > 0 else "−"
            cx_b  = (x1 + x2) // 2
            cy_b  = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 70), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 130), 1, cv2.LINE_AA)
            cv2.putText(frame, label, (cx_b - 8, cy_b + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        # Undo / Redo buttons
        for btn, lbl, clr_bg, clr_txt in [
            (self._undo_btn, "<-", (80, 50, 30), (255, 180, 80)),
            (self._redo_btn, "->", (30, 60, 50), (80, 220, 140)),
        ]:
            bx1, by1, bx2, by2 = btn
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), clr_bg, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), clr_txt, 1, cv2.LINE_AA)
            tcx = (bx1 + bx2) // 2 - 10
            tcy = (by1 + by2) // 2 + 5
            cv2.putText(frame, lbl, (tcx, tcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr_txt, 2)

        # Current brush preview (right side of palette)
        bx = self.width - 90
        by_center = CIRCLE_CY
        r = min(CIRCLE_R, self.brush_size)
        cv2.circle(frame, (bx, by_center), r, self.current_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"B:{self.brush_size}", (bx + CIRCLE_R + 6, by_center + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _draw_cursor(self, frame: np.ndarray, gesture_info: dict):
        cursor  = gesture_info.get("cursor")
        gesture = gesture_info.get("gesture", "idle")
        if cursor is None:
            return

        x, y = cursor
        sz = 22  # crosshair arm length
        BLK = (0, 0, 0)  # shadow color

        if gesture == "draw":
            color = self.current_color
            r = max(4, self.brush_size)
            # Shadow outlines (thick black behind everything)
            cv2.circle(frame, (x, y), r + 6, BLK, 3, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, BLK, 4, cv2.LINE_AA)
            cv2.line(frame, (x - sz, y), (x - r - 3, y), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x + r + 3, y), (x + sz, y), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x, y - sz), (x, y - r - 3), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x, y + r + 3), (x, y + sz), BLK, 3, cv2.LINE_AA)
            # Color layer
            cv2.circle(frame, (x, y), r + 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), r, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(frame, (x - sz, y), (x - r - 3, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x + r + 3, y), (x + sz, y), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - sz), (x, y - r - 3), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y + r + 3), (x, y + sz), color, 1, cv2.LINE_AA)

        elif gesture == "erase":
            er = self.eraser_size // 2
            # Shadow
            cv2.rectangle(frame, (x - er, y - er), (x + er, y + er), BLK, 4, cv2.LINE_AA)
            # Color
            cv2.rectangle(frame, (x - er, y - er), (x + er, y + er),
                          (180, 180, 180), 2, cv2.LINE_AA)
            cv2.line(frame, (x - 8, y - 8), (x + 8, y + 8), (0, 100, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x + 8, y - 8), (x - 8, y + 8), (0, 100, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "ERASE", (x - 22, y - er - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLK, 3)
            cv2.putText(frame, "ERASE", (x - 22, y - er - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)

        elif gesture == "zoom":
            # Shadow
            cv2.circle(frame, (x, y), 14, BLK, 4, cv2.LINE_AA)
            cv2.line(frame, (x - sz, y), (x + sz, y), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x, y - sz), (x, y + sz), BLK, 3, cv2.LINE_AA)
            # Color
            cv2.circle(frame, (x, y), 14, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x - sz, y), (x + sz, y), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - sz), (x, y + sz), (255, 0, 255), 1, cv2.LINE_AA)

        elif gesture == "ocr":
            # Green crosshair for OCR scanning mode
            cv2.circle(frame, (x, y), 16, BLK, 4, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 16, (0, 220, 100), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.putText(frame, "SCAN", (x - 18, y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLK, 3)
            cv2.putText(frame, "SCAN", (x - 18, y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

        else:
            # Idle / clear / save — bright crosshair with black outline
            cv2.circle(frame, (x, y), 12, BLK, 4, cv2.LINE_AA)
            cv2.line(frame, (x - sz, y), (x - 6, y), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x + 6, y), (x + sz, y), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x, y - sz), (x, y - 6), BLK, 3, cv2.LINE_AA)
            cv2.line(frame, (x, y + 6), (x, y + sz), BLK, 3, cv2.LINE_AA)
            # Color
            cv2.circle(frame, (x, y), 12, (100, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(frame, (x - sz, y), (x - 6, y), (100, 200, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x + 6, y), (x + sz, y), (100, 200, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x, y - sz), (x, y - 6), (100, 200, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x, y + 6), (x, y + sz), (100, 200, 255), 1, cv2.LINE_AA)

    def _draw_gesture_badge(self, frame: np.ndarray, gesture_info: dict):
        gesture = gesture_info.get("gesture", "idle")
        num     = gesture_info.get("num_hands", 0)
        colors  = {
            "draw":  (50, 220, 80),
            "erase": (30, 160, 255),
            "ocr":   (0, 220, 100),
            "clear": (30,  30, 255),
            "save":  (220, 200, 0),
            "zoom":  (255, 0, 255),
            "idle":  (90,  90, 100),
        }
        c = colors.get(gesture, (90, 90, 100))
        label = gesture.upper()
        bx, by = self.width - 170, PALETTE_H + 20
        cv2.putText(frame, f"✋ {label}", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        if num > 1 and gesture != "zoom":
            cv2.putText(frame, "DUAL HAND", (bx, by + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 200, 0), 1)

    def _draw_zoom_badge(self, frame: np.ndarray, gesture_info: dict):
        """Show zoom level indicator + visual feedback during zoom."""
        gesture = gesture_info.get("gesture", "idle")
        cursor1 = gesture_info.get("cursor")
        cursor2 = gesture_info.get("cursor2")

        # Always show zoom level when not 1.0
        if self.zoom_level < 0.99 or self.zoom_level > 1.01:
            zoom_text = f"ZOOM {self.zoom_level:.1f}x"
            tw, _ = cv2.getTextSize(zoom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            zx = (self.width - tw) // 2
            zy = PALETTE_H + 30
            cv2.putText(frame, zoom_text, (zx, zy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        # Draw visual feedback during zoom gesture
        if gesture == "zoom" and cursor1 and cursor2:
            c1 = cursor1
            c2 = cursor2
            # Draw circles on both hands
            cv2.circle(frame, c1, 20, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, c2, 20, (255, 0, 255), 2, cv2.LINE_AA)
            # Draw line between hands
            cv2.line(frame, c1, c2, (255, 0, 255), 2, cv2.LINE_AA)
            # Draw midpoint
            mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            cv2.circle(frame, mid, 6, (255, 255, 0), -1, cv2.LINE_AA)
            # Distance label
            dist = int(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
            cv2.putText(frame, f"{dist}px", (mid[0] + 10, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def _draw_status(self, frame: np.ndarray):
        if self._status_frames <= 0:
            return
        alpha = min(1.0, self._status_frames / 25.0)
        txt   = self._status_text
        tw, th = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
        tx = (self.width  - tw) // 2
        ty = self.height - 70
        overlay = frame.copy()
        cv2.putText(overlay, txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        self._status_frames -= 1

    def _draw_hints(self, frame: np.ndarray):
        auto_label = "ON" if self.auto_shape_enabled else "OFF"
        zoom_label = f"{self.zoom_level:.1f}x" if self.zoom_level != 1.0 else "1x"
        hints = f"Z:Undo  Y:Redo  S:PNG  F:LoadFile  T:AutoSnap[{auto_label}]  R:ResetZoom[{zoom_label}]  +/-:Brush  Q:Quit"
        cv2.putText(frame, hints,
                    (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 80), 1)
