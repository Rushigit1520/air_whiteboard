"""
Gesture Detection Agent
-----------------------
Uses MediaPipe Hands to:
  • Track index-finger tip for cursor positioning.
  • Classify hand gestures:
      1 finger  (index up)          → 'draw'
      2 fingers (index + middle)    → 'erase'
      4+ fingers (open palm)        → 'clear'
      2 hands detected              → 'save'
      Pinch (thumb + index close)   → 'palette_select'
  • Support basic multi-user differentiation via hand chirality.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from utils.smoothing import PointSmoother


# Landmark indices (MediaPipe convention)
FINGER_TIPS = [8, 12, 16, 20]       # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]       # corresponding PIP joints
THUMB_TIP   = 4
THUMB_IP    = 3
INDEX_TIP   = 8
INDEX_MCP   = 5


class HandInfo:
    """Encapsulates per-hand data."""

    def __init__(self):
        self.landmarks = None
        self.handedness: str = "Right"   # 'Left' | 'Right'
        self.fingers: list[int] = [0, 0, 0, 0]  # index, middle, ring, pinky
        self.gesture: str = "idle"
        self.cursor: tuple[int, int] | None = None
        self.smoother = PointSmoother(window_size=6, alpha=0.5)


class GestureAgent:
    """
    Processes raw video frames and returns structured gesture information
    for all detected hands.
    """

    # How many consecutive frames a gesture must persist before being reported
    GESTURE_CONFIRM_FRAMES = 3

    def __init__(self, max_hands: int = 2):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.65,
        )

        # Per-slot smoothers (up to max_hands slots)
        self._slots: list[HandInfo] = [HandInfo() for _ in range(max_hands)]
        self._gesture_counters: list[deque] = [
            deque(maxlen=self.GESTURE_CONFIRM_FRAMES) for _ in range(max_hands)
        ]

        # The annotated frame for the camera preview
        self.annotated_frame: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fingers_up(self, lm, handedness: str) -> list[int]:
        """
        Returns [index, middle, ring, pinky] up-states (1=up, 0=down).
        Thumb is excluded to avoid false positives caused by hand rotation.
        """
        tips = FINGER_TIPS
        pips = FINGER_PIPS
        states = []
        for tip, pip in zip(tips, pips):
            # A finger is 'up' when its tip is above its PIP joint (smaller y)
            states.append(1 if lm[tip].y < lm[pip].y else 0)
        return states

    def _is_pinch(self, lm) -> bool:
        """Thumb tip and index tip close together → pinch."""
        tx, ty = lm[THUMB_TIP].x, lm[THUMB_TIP].y
        ix, iy = lm[INDEX_TIP].x, lm[INDEX_TIP].y
        return np.hypot(tx - ix, ty - iy) < 0.06

    def _classify(self, fingers: list[int]) -> str:
        """Classify a SINGLE hand gesture (multi-hand logic handled later)."""
        total = sum(fingers)
        idx, mid, ring, pinky = fingers
        if total == 1 and idx == 1:
            return "draw"
        if total == 2 and idx == 1 and mid == 1:
            return "erase"
        if total == 3 and idx == 1 and mid == 1 and ring == 1:
            return "ocr"       # 3 fingers = scan text
        if total >= 4:
            return "clear"
        return "idle"

    def _stable_gesture(self, slot_idx: int, raw: str) -> str:
        """Only report a gesture once it has appeared N consecutive frames."""
        q = self._gesture_counters[slot_idx]
        q.append(raw)
        if len(q) == self.GESTURE_CONFIRM_FRAMES and len(set(q)) == 1:
            return raw
        # Fall back to the most recent confirmed gesture in the deque
        return q[-1] if q else "idle"

    def _get_cursor(self, lm, w: int, h: int, slot: HandInfo) -> tuple[int, int]:
        raw_x = int(lm[INDEX_TIP].x * w)
        raw_y = int(lm[INDEX_TIP].y * h)
        return slot.smoother.smooth((raw_x, raw_y))

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray) -> dict:
        """
        Process a (flipped) BGR frame.

        Returns:
            {
              'gesture'   : str,               # dominant gesture
              'cursor'    : (x,y) | None,      # primary hand cursor
              'num_hands' : int,
              'hands'     : [HandInfo, ...],   # per-hand details
              'fingers'   : [int, int, int, int],
            }
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        annotated = frame.copy()

        info: dict = {
            "gesture": "idle",
            "cursor": None,
            "cursor2": None,       # second hand cursor (for zoom)
            "num_hands": 0,
            "hands": [],
            "fingers": [0, 0, 0, 0],
        }

        if not results.multi_hand_landmarks:
            # Reset smoothers on no detection
            for slot in self._slots:
                slot.smoother.reset()
            self.annotated_frame = annotated
            return info

        num_hands = len(results.multi_hand_landmarks)
        info["num_hands"] = num_hands

        for i, (hand_lm, hand_class) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            slot = self._slots[i] if i < len(self._slots) else HandInfo()
            slot.landmarks  = hand_lm.landmark
            slot.handedness = hand_class.classification[0].label  # 'Left'|'Right'

            fingers = self._fingers_up(hand_lm.landmark, slot.handedness)
            slot.fingers = fingers

            raw_gesture = self._classify(fingers)
            slot.gesture = self._stable_gesture(i, raw_gesture)

            slot.cursor = self._get_cursor(hand_lm.landmark, w, h, slot)
            info["hands"].append(slot)

            # Draw landmarks on preview
            self._mp_draw.draw_landmarks(
                annotated,
                hand_lm,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

            # Highlight index tip
            ix = int(hand_lm.landmark[INDEX_TIP].x * w)
            iy = int(hand_lm.landmark[INDEX_TIP].y * h)
            color_map = {
                "draw": (0, 255, 0),
                "erase": (0, 165, 255),
                "ocr":  (0, 220, 100),
                "clear": (0, 0, 255),
                "save": (255, 220, 0),
                "zoom": (255, 0, 255),
                "idle": (180, 180, 180),
            }
            cv2.circle(annotated, (ix, iy), 8, color_map.get(slot.gesture, (255,255,255)), -1)

        # ── Determine combined gesture ────────────────────────────────────
        primary = info["hands"][0]
        info["cursor"]  = primary.cursor
        info["fingers"] = primary.fingers

        if num_hands >= 2 and len(info["hands"]) >= 2:
            g0 = info["hands"][0].gesture
            g1 = info["hands"][1].gesture
            info["cursor2"] = info["hands"][1].cursor

            if g0 == "clear" and g1 == "clear":
                # Both open palms → ZOOM
                info["gesture"] = "zoom"
            else:
                info["gesture"] = "save"
        else:
            info["gesture"] = primary.gesture

        self.annotated_frame = annotated
        return info
