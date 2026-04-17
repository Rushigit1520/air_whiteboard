"""
AI Processing Agent
-------------------
Provides three optional AI-powered post-processing features:

1. Shape auto-correction  – detects circles, rectangles, triangles drawn
                             freehand and replaces them with perfect geometry.
2. Handwriting OCR        – extracts recognised text from the canvas using
                             Tesseract (optional dependency).
3. Drawing smoothing      – applies a light Gaussian pass to reduce noise.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Optional: Tesseract OCR ───────────────────────────────────────────────────
try:
    import pytesseract  # type: ignore
    _TESS_OK = True

    # Auto-detect Tesseract on Windows (common install paths)
    import os as _os
    _tess_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        _os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for _p in _tess_paths:
        if _os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
except ImportError:
    _TESS_OK = False


# ── Shape classification thresholds ──────────────────────────────────────────
# NOTE: These are tuned for AIR-DRAWING which is far shakier than pen/mouse.
_APPROX_EPS          = 0.05    # polyDP epsilon factor (higher = more forgiving)
_CIRCLE_ROUNDNESS    = 0.55    # min area ratio contour/enclosing-circle (was 0.82)
_MIN_SHAPE_AREA      = 400     # ignore tiny scribbles (px²)  (was 1000)
_STROKE_CLOSE_RATIO  = 0.40    # start↔end gap / perimeter to count as closed (was 0.25)
_MIN_STROKE_POINTS   = 10      # minimum points for shape detection (was 20)


class AIAgent:
    """Stateless helper; call methods as needed from main loop."""

    def __init__(self):
        self.shape_correction_enabled = False
        self.ocr_available            = _TESS_OK

    # ─────────────────────────────────────────────────────────────────────────
    #  1.  Shape detection & auto-correction
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_contour(cnt) -> str:
        area = cv2.contourArea(cnt)
        if area < _MIN_SHAPE_AREA:
            return "noise"

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, _APPROX_EPS * peri, True)
        n      = len(approx)

        if n == 3:
            return "triangle"

        if n == 4:
            _, _, w, h = cv2.boundingRect(approx)
            ar = w / float(h) if h > 0 else 0
            return "square" if 0.85 <= ar <= 1.15 else "rectangle"

        # Circle test: compare area to bounding enclosing circle
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        circle_area   = np.pi * radius ** 2
        if circle_area > 0 and area / circle_area >= _CIRCLE_ROUNDNESS:
            return "circle"

        # Arrow / line test (very elongated bounding box)
        _, _, w, h = cv2.boundingRect(cnt)
        long, short = max(w, h), min(w, h)
        if short > 0 and long / short > 5:
            return "line"

        return "unknown"

    def detect_shapes(self, canvas: np.ndarray) -> list[tuple]:
        """
        Return list of (contour, shape_name) for all meaningful shapes on canvas.
        Shapes: 'circle', 'rectangle', 'square', 'triangle', 'line', 'unknown'.
        """
        gray    = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        # Small morphological close to join near-touching strokes
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for cnt in contours:
            label = self._classify_contour(cnt)
            if label not in ("noise", "unknown"):
                shapes.append((cnt, label))
        return shapes

    def auto_correct_shape(
        self,
        canvas: np.ndarray,
        contour,
        shape: str,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Erase the freehand shape and redraw a perfect geometric version."""
        x, y, w, h = cv2.boundingRect(contour)
        # Erase original pixels inside bounding rect
        cv2.drawContours(canvas, [contour], -1, (0, 0, 0), cv2.FILLED)

        thick = 3

        if shape == "circle":
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            cv2.circle(canvas, (int(cx), int(cy)), int(r), color, thick, cv2.LINE_AA)

        elif shape in ("rectangle", "square"):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thick, cv2.LINE_AA)

        elif shape == "triangle":
            peri   = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, _APPROX_EPS * peri, True)
            if len(approx) == 3:
                cv2.polylines(canvas, [approx], True, color, thick, cv2.LINE_AA)

        elif shape == "line":
            # Fit a line and draw it
            vx, vy, cx_, cy_ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            length = max(w, h)
            p1 = (int(cx_ - vx * length), int(cy_ - vy * length))
            p2 = (int(cx_ + vx * length), int(cy_ + vy * length))
            cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)

        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    #  1b. Real-time stroke shape classification  (tuned for air-drawing)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _subsample_points(pts: np.ndarray, max_pts: int = 120) -> np.ndarray:
        """Down-sample noisy stroke to reduce jitter; keeps start & end."""
        n = len(pts)
        if n <= max_pts:
            return pts
        indices = np.linspace(0, n - 1, max_pts, dtype=int)
        return pts[indices]

    @staticmethod
    def _smooth_points(pts: np.ndarray, window: int = 5) -> np.ndarray:
        """Simple moving-average smooth to reduce hand-shake noise."""
        if len(pts) < window:
            return pts
        kernel = np.ones(window) / window
        sx = np.convolve(pts[:, 0].astype(float), kernel, mode='valid')
        sy = np.convolve(pts[:, 1].astype(float), kernel, mode='valid')
        smoothed = np.stack([sx, sy], axis=1).astype(np.int32)
        return smoothed

    @staticmethod
    def classify_stroke(
        points: list[tuple[int, int]],
    ) -> tuple[str, dict] | None:
        """
        Analyse a freshly-completed stroke and return (shape_name, params)
        if it looks like a recognisable shape.  Tuned for shaky air-drawing.

        Returns None when the stroke doesn't match any shape.

        Supported shapes and their params:
          'circle'    → {'center': (cx,cy), 'radius': r}
          'rectangle' → {'x': x, 'y': y, 'w': w, 'h': h}
          'triangle'  → {'points': np.array}
          'line'      → {'p1': (x1,y1), 'p2': (x2,y2)}
        """
        if len(points) < _MIN_STROKE_POINTS:
            return None

        raw_pts = np.array(points, dtype=np.int32)

        # ── Pre-process: sub-sample + smooth to remove jitter ──────────────
        pts = AIAgent._subsample_points(raw_pts, max_pts=150)
        pts = AIAgent._smooth_points(pts, window=5)

        if len(pts) < 6:
            return None

        contour = pts.reshape((-1, 1, 2))

        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)

        if peri < 40:           # too small
            return None

        # ── Check if stroke is closed (start ≈ end) ────────────────────────
        start = raw_pts[0]       # use original start/end for gap check
        end   = raw_pts[-1]
        gap   = np.linalg.norm(start.astype(float) - end.astype(float))
        closed = gap / max(peri, 1) < _STROKE_CLOSE_RATIO

        # ── Closed shape analysis ──────────────────────────────────────────
        if closed and area > _MIN_SHAPE_AREA:
            # Use convex hull for more robust analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)

            # ── Circle detection (multi-metric) ───────────────────────────
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            enclosing_area = np.pi * radius ** 2

            # Metric 1: area ratio vs enclosing circle
            ratio1 = hull_area / enclosing_area if enclosing_area > 0 else 0

            # Metric 2: circularity = 4π × area / perimeter²
            circularity = (4 * np.pi * hull_area) / (hull_peri ** 2) if hull_peri > 0 else 0

            # Metric 3: aspect ratio of bounding rect (circles ≈ 1.0)
            bx, by, bw, bh = cv2.boundingRect(hull)
            aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

            # Accept as circle if any two of the three metrics pass
            circle_votes = 0
            if ratio1 >= _CIRCLE_ROUNDNESS:       # 0.55
                circle_votes += 1
            if circularity >= 0.65:                # perfect circle = 1.0
                circle_votes += 1
            if aspect >= 0.65:                     # roughly square bounding box
                circle_votes += 1

            if circle_votes >= 2 and radius > 15:
                return ("circle", {
                    "center": (int(cx), int(cy)),
                    "radius": int(radius),
                })

            # ── Triangle detection ────────────────────────────────────────
            approx = cv2.approxPolyDP(hull, 0.06 * hull_peri, True)
            n_verts = len(approx)

            if n_verts == 3:
                return ("triangle", {"points": approx})

            # ── Rectangle detection (via minAreaRect for rotated rects) ───
            if 4 <= n_verts <= 8:
                rect = cv2.minAreaRect(hull)
                (rx, ry), (rw, rh), angle = rect

                # Check how well the hull fills the min-area rectangle
                rect_area = rw * rh
                fill_ratio = hull_area / rect_area if rect_area > 0 else 0

                if fill_ratio > 0.70 and min(rw, rh) > 25:
                    # Use axis-aligned bounding box for the output
                    x, y, w, h = cv2.boundingRect(hull)
                    return ("rectangle", {"x": x, "y": y, "w": w, "h": h})

        # ── Open shape: straight line ──────────────────────────────────────
        if not closed:
            x, y, w, h = cv2.boundingRect(contour)
            diag = np.hypot(w, h)
            # A line's arc length should be close to the endpoint distance
            if diag > 50 and peri < diag * 3.5:
                vx, vy, cx_, cy_ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                # Compute max deviation from the fitted line
                line_dir = np.array([vx[0], vy[0]])
                line_pt  = np.array([cx_[0], cy_[0]])
                diffs = pts.astype(float) - line_pt
                cross = np.abs(diffs[:, 0] * line_dir[1] - diffs[:, 1] * line_dir[0])
                max_dev = np.max(cross)
                if max_dev < diag * 0.15:  # 15% deviation (was 10%)
                    p1 = tuple(raw_pts[0])
                    p2 = tuple(raw_pts[-1])
                    return ("line", {"p1": p1, "p2": p2})

        return None

    @staticmethod
    def draw_perfect_shape(
        canvas: np.ndarray,
        shape: str,
        params: dict,
        color: tuple[int, int, int],
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw a perfect geometric shape onto the canvas."""
        if shape == "circle":
            cv2.circle(
                canvas,
                params["center"],
                params["radius"],
                color, thickness, cv2.LINE_AA,
            )
        elif shape == "rectangle":
            x, y, w, h = params["x"], params["y"], params["w"], params["h"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
        elif shape == "triangle":
            cv2.polylines(canvas, [params["points"]], True, color, thickness, cv2.LINE_AA)
        elif shape == "line":
            cv2.line(canvas, params["p1"], params["p2"], color, thickness, cv2.LINE_AA)
        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    #  1c. Letter / character recognition from a single stroke
    # ─────────────────────────────────────────────────────────────────────────

    def recognize_letter(
        self,
        canvas: np.ndarray,
        stroke_points: list[tuple[int, int]],
    ) -> tuple[str, dict] | None:
        """
        Try to recognise a single handwritten letter/digit from a stroke.

        Process:
          1. Compute bounding box of the stroke points.
          2. Extract that ROI from the canvas, preprocess for OCR.
          3. Run Tesseract in single-character mode (--psm 10).
          4. Return ('letter', {'char': 'A', 'bbox': (x,y,w,h)}) or None.
        """
        if not _TESS_OK:
            return None

        if len(stroke_points) < 8:
            return None

        pts = np.array(stroke_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        # Ignore tiny scribbles or very narrow strokes
        if w < 15 or h < 15:
            return None

        # Pad the ROI so Tesseract has breathing room
        pad = max(20, max(w, h) // 3)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(canvas.shape[1], x + w + pad)
        y2 = min(canvas.shape[0], y + h + pad)

        roi = canvas[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Preprocess: grayscale → invert → threshold (black text on white bg)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)

        # Resize up if too small (Tesseract works better on larger images)
        min_dim = 64
        rh, rw = binary.shape
        if rh < min_dim or rw < min_dim:
            scale = max(min_dim / rw, min_dim / rh, 2.0)
            binary = cv2.resize(
                binary,
                (int(rw * scale), int(rh * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Add white border for Tesseract
        binary = cv2.copyMakeBorder(
            binary, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=255,
        )

        try:
            # --psm 10 = single character
            # --psm 8  = single word
            # Try single char first, fall back to single word
            char = pytesseract.image_to_string(
                binary,
                config="--psm 10 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            ).strip()

            if not char:
                # Try single word mode
                char = pytesseract.image_to_string(
                    binary,
                    config="--psm 8 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                ).strip()

            if char and len(char) <= 3 and any(c.isalnum() for c in char):
                # Keep only alphanumeric characters
                char = ''.join(c for c in char if c.isalnum())
                if char:
                    return ("letter", {
                        "char": char,
                        "bbox": (x, y, w, h),
                    })
        except Exception:
            pass

        return None

    @staticmethod
    def draw_perfect_letter(
        canvas: np.ndarray,
        char: str,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw a clean, perfectly rendered letter/digit on the canvas.
        The letter is scaled to fit the bounding box of the original stroke.
        """
        x, y, w, h = bbox

        # Choose font scale to fill the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Start with a guess and adjust
        target_h = int(h * 0.9)
        font_scale = 1.0

        # Binary search for the right font scale
        for _ in range(15):
            (tw, th), baseline = cv2.getTextSize(char, font, font_scale, thickness)
            if th < target_h * 0.9:
                font_scale *= 1.2
            elif th > target_h * 1.1:
                font_scale *= 0.85
            else:
                break

        (tw, th), baseline = cv2.getTextSize(char, font, font_scale, thickness)

        # Center the text in the bounding box
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2

        # Draw with anti-aliasing
        cv2.putText(
            canvas, char, (tx, ty),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )
        return canvas

    def run_shape_correction(
        self,
        canvas: np.ndarray,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> tuple[np.ndarray, int]:
        """
        Detect all shapes and auto-correct them in-place.
        Returns (modified_canvas, number_corrected).
        """
        shapes = self.detect_shapes(canvas)
        for cnt, label in shapes:
            self.auto_correct_shape(canvas, cnt, label, color)
        return canvas, len(shapes)

    # ─────────────────────────────────────────────────────────────────────────
    #  2.  Handwriting OCR
    # ─────────────────────────────────────────────────────────────────────────

    def ocr_canvas(self, canvas: np.ndarray) -> str:
        """
        Run Tesseract OCR on the full canvas and return recognised text.
        Returns empty string if Tesseract is not installed.
        """
        if not _TESS_OK:
            return "(pytesseract not installed)"

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # Invert: white text on black → black text on white for Tesseract
        inverted  = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 200, 255, cv2.THRESH_BINARY)

        try:
            text = pytesseract.image_to_string(
                binary,
                config="--psm 6 --oem 1",
            )
            return text.strip()
        except Exception as exc:
            return f"OCR error: {exc}"

    def ocr_region(self, canvas: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """OCR a sub-region of the canvas."""
        roi = canvas[y : y + h, x : x + w]
        return self.ocr_canvas(roi)

    # ─────────────────────────────────────────────────────────────────────────
    #  3.  Drawing smoothing
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def smooth_canvas(canvas: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Apply a gentle Gaussian blur to reduce jagged lines.
        ksize must be odd; larger values produce more smoothing.
        """
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(canvas, (ksize, ksize), 0)
        # Only apply blur where pixels are already painted (preserve black bg)
        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        result = canvas.copy()
        result[mask > 0] = blurred[mask > 0]
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Capability info
    # ─────────────────────────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        return {
            "shape_correction": True,
            "ocr":              _TESS_OK,
            "smoothing":        True,
        }
