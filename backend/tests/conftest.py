"""Synthetic fixture: a 12x12 street grid (100 m blocks) with a noisy GPS trace along a known route.
Lets the whole API be tested in CI without the (git-ignored) hackathon data."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

LON0, LAT0 = 76.35, 10.20   # near Angamaly
N = 12
M_PER_DEG_LAT = 111_195.0


def _write_trip(root, trip_id="00000001", seed=0, with_route=True):
    rng = np.random.default_rng(seed)
    d = root / trip_id
    d.mkdir(parents=True)
    m_lon = M_PER_DEG_LAT * np.cos(np.deg2rad(LAT0))
    nodes = np.array([[LON0 + c * 100 / m_lon, LAT0 + r * 100 / M_PER_DEG_LAT] for r in range(N) for c in range(N)])
    nid = lambda r, c: r * N + c
    arcs, arc_id = [], {}
    for r in range(N):
        for c in range(N):
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    arc_id[(nid(r, c), nid(rr, cc))] = len(arcs)
                    arcs.append((nid(r, c), nid(rr, cc)))
    # route: east along row 2, then north along column 9
    path = [nid(2, c) for c in range(0, 10)] + [nid(r, 9) for r in range(3, 11)]
    route = [arc_id[(a, b)] for a, b in zip(path[:-1], path[1:])]
    pts = []
    for a in route:
        f, t = arcs[a]
        for s in np.arange(0, 100, 12.0):  # ~12 m per second
            pts.append(nodes[f] + (nodes[t] - nodes[f]) * s / 100)
    pts = np.array(pts)
    noise = rng.normal(0, 8, pts.shape) / np.array([m_lon, M_PER_DEG_LAT])
    track = np.column_stack([pts + noise, np.arange(1, len(pts) + 1, dtype=float)])
    np.savetxt(d / f"{trip_id}.track", track, fmt="%.6f", delimiter="\t")
    np.savetxt(d / f"{trip_id}.nodes", nodes, fmt="%.6f", delimiter="\t")
    np.savetxt(d / f"{trip_id}.arcs", np.array(arcs), fmt="%d", delimiter="\t")
    if with_route:
        np.savetxt(d / f"{trip_id}.route", np.array(route), fmt="%d")
    return route


@pytest.fixture()
def data_root(tmp_path):
    _write_trip(tmp_path / "student_test", with_route=True)
    return tmp_path


@pytest.fixture()
def make_client(data_root, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    def _make(**env):
        monkeypatch.setenv("DATA_ROOT", str(data_root))
        monkeypatch.setenv("TRIP_SET", "student_test")
        monkeypatch.setenv("ARTIFACTS_DIR", str(env.pop("artifacts", tmp_path / "no_artifacts")))
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        from backend.app import config
        config.get_settings.cache_clear()
        import backend.app.main as main
        importlib.reload(main)
        return TestClient(main.app)
    return _make
