"""
Smoothing utilities for cursor stabilization and line rendering.
"""

import numpy as np
from collections import deque


class PointSmoother:
    """One-Euro Filter — adaptive smoothing for cursor positions.
    
    Smooth for slow movements (reduces jitter) but fast for quick swipes.
    This is far superior to simple EMA for hand-tracking cursors.
    """

    def __init__(self, window_size: int = 6, alpha: float = 0.55,
                 min_cutoff: float = 1.5, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff   # min smoothing (lower = smoother)
        self.beta = beta               # speed response (higher = faster response)
        self.d_cutoff = d_cutoff       # derivative smoothing
        self.alpha = alpha             # fallback alpha

        # State
        self._x_prev: float | None = None
        self._y_prev: float | None = None
        self._dx_prev: float = 0.0
        self._dy_prev: float = 0.0
        self._initialized = False

        # Additional moving-average buffer for extra stability
        self.buffer: deque = deque(maxlen=window_size)

    def _one_euro_alpha(self, cutoff: float, dt: float = 1.0 / 30.0) -> float:
        """Compute alpha from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])

        if not self._initialized:
            self._x_prev = x
            self._y_prev = y
            self._initialized = True
            self.buffer.append((int(x), int(y)))
            return int(x), int(y)

        # Compute speed (derivative)
        dx = x - self._x_prev
        dy = y - self._y_prev

        # Smooth the derivative
        a_d = self._one_euro_alpha(self.d_cutoff)
        self._dx_prev = a_d * dx + (1 - a_d) * self._dx_prev
        self._dy_prev = a_d * dy + (1 - a_d) * self._dy_prev

        # Compute speed magnitude
        speed = np.hypot(self._dx_prev, self._dy_prev)

        # Adaptive cutoff: faster movement → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * speed

        # Filter
        a = self._one_euro_alpha(cutoff)
        self._x_prev = a * x + (1 - a) * self._x_prev
        self._y_prev = a * y + (1 - a) * self._y_prev

        result = (int(self._x_prev), int(self._y_prev))
        self.buffer.append(result)
        return result

    def reset(self):
        self.buffer.clear()
        self._x_prev = None
        self._y_prev = None
        self._dx_prev = 0.0
        self._dy_prev = 0.0
        self._initialized = False


def interpolate_points(
    p1: tuple[int, int], p2: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Generate densely-interpolated points between p1 and p2
    so cv2.line calls are never spaced too far apart.
    """
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(1, int(dist))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        pts.append(
            (
                int(p1[0] + (p2[0] - p1[0]) * t),
                int(p1[1] + (p2[1] - p1[1]) * t),
            )
        )
    return pts


def catmull_rom_chain(points: list[tuple[int, int]], samples: int = 8) -> list[tuple[int, int]]:
    """
    Generate a Catmull-Rom spline chain through a list of control points.
    Returns a smooth polyline with `samples` intermediate points per segment.
    """
    if len(points) < 4:
        return points

    result = []
    for i in range(1, len(points) - 2):
        p0 = np.array(points[i - 1], dtype=float)
        p1 = np.array(points[i], dtype=float)
        p2 = np.array(points[i + 1], dtype=float)
        p3 = np.array(points[i + 2], dtype=float)

        for t in np.linspace(0, 1, samples, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            result.append((int(pt[0]), int(pt[1])))

    result.append(points[-2])
    return result
