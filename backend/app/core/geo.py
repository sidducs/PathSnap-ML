"""Geometry helpers.

GPS coordinates are angles, so Euclidean distance in degrees is meaningless. Every trip lives inside a
single city (<~30 km), so we project lon/lat onto a local equirectangular plane (metres) centred on the
road network. Over that extent the distortion is well below GPS noise (<0.1 %), and all distance
maths becomes plain NumPy.  Coordinate order in the raw files is (longitude, latitude).
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


class LocalProjector:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._ky = np.deg2rad(1.0) * EARTH_RADIUS_M
        self._kx = self._ky * np.cos(np.deg2rad(self.lat0))

    def to_xy(self, lon, lat):
        return (np.asarray(lon, float) - self.lon0) * self._kx, (np.asarray(lat, float) - self.lat0) * self._ky

    def to_lonlat(self, x, y):
        return np.asarray(x, float) / self._kx + self.lon0, np.asarray(y, float) / self._ky + self.lat0


def project_to_segments(px, py, x0, y0, x1, y1):
    """Distance from points to segments (all arrays broadcast element-wise).

    Returns (distance_m, t) where t in [0, 1] is the position of the foot point along the segment.
    Zero-length segments (self-loop arcs exist in the data) give t = 0.
    """
    dx, dy = x1 - x0, y1 - y0
    len2 = dx * dx + dy * dy
    safe = np.where(len2 > 0, len2, 1.0)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / safe, 0.0, 1.0)
    t = np.where(len2 > 0, t, 0.0)
    fx, fy = x0 + t * dx, y0 + t * dy
    return np.hypot(px - fx, py - fy), t


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
