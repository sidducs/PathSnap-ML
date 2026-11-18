"""Road network = directed arcs between nodes, projected to local metres, with a spatial index."""
from __future__ import annotations

from collections import deque

import numpy as np
import shapely
from shapely import STRtree

from .geo import LocalProjector
from .trip_io import TripInput


class RoadNetwork:
    HOP_CAP = 8  # BFS depth cached for HMM transitions

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray):
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.proj = LocalProjector(nodes_lonlat[:, 0].mean(), nodes_lonlat[:, 1].mean())
        nx, ny = self.proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.node_xy = np.column_stack([nx, ny])
        self.arc_from = arcs[:, 0].astype(np.int64)
        self.arc_to = arcs[:, 1].astype(np.int64)
        self.x0, self.y0 = nx[self.arc_from], ny[self.arc_from]
        self.x1, self.y1 = nx[self.arc_to], ny[self.arc_to]
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        self.arc_len = np.hypot(dx, dy)
        self.arc_bearing = np.arctan2(dy, dx)  # math convention (rad, CCW from +x/east)

        # CSR adjacency: outgoing arcs per node
        order = np.argsort(self.arc_from, kind="stable")
        self._out_arcs = order
        self._out_ptr = np.searchsorted(self.arc_from[order], np.arange(self.n_nodes + 1))
        self.out_degree = np.diff(self._out_ptr)                              # per node
        self.in_degree = np.bincount(self.arc_to, minlength=self.n_nodes)      # per node

        # reverse twin (arc B with from/to swapped), -1 if none
        key = self.arc_from * self.n_nodes + self.arc_to
        rkey = self.arc_to * self.n_nodes + self.arc_from
        sk = np.argsort(key, kind="stable")
        pos = np.searchsorted(key[sk], rkey)
        pos_c = np.clip(pos, 0, len(key) - 1)
        hit = key[sk][pos_c] == rkey
        self.reverse_of = np.where(hit, sk[pos_c], -1)

        self._tree: STRtree | None = None
        self._hops_cache: dict[int, dict[int, int]] = {}

    @classmethod
    def from_trip(cls, trip: TripInput) -> "RoadNetwork":
        return cls(trip.nodes_lonlat, trip.arcs)

    # ---------------- topology ----------------
    def successors(self, arc: int) -> np.ndarray:
        n = self.arc_to[arc]
        return self._out_arcs[self._out_ptr[n]:self._out_ptr[n + 1]]

    def connected(self, a: int, b: int) -> bool:
        """True if arc b can directly follow arc a (or is a itself)."""
        return a == b or self.arc_to[a] == self.arc_from[b]

    def hops_from(self, arc: int, cap: int | None = None) -> dict[int, int]:
        """{reachable_arc: min #arc-hops} for up to HOP_CAP hops (BFS, cached; excludes `arc` itself)."""
        cached = self._hops_cache.get(arc)
        if cached is not None:
            return cached
        cap = self.HOP_CAP if cap is None else cap
        seen = {arc: 0}
        q = deque([arc])
        while q:
            a = q.popleft()
            h = seen[a]
            if h >= cap:
                continue
            for b in self.successors(a):
                b = int(b)
                if b not in seen:
                    seen[b] = h + 1
                    q.append(b)
        del seen[arc]
        self._hops_cache[arc] = seen
        return seen

    def shortest_arc_path(self, a: int, b: int, max_hops: int = 12) -> list[int] | None:
        """Arcs strictly between a and b on the fewest-hop path, [] if adjacent, None if unreachable."""
        if self.connected(a, b):
            return []
        parent = {a: -1}
        q = deque([(a, 0)])
        while q:
            cur, h = q.popleft()
            if h >= max_hops:
                continue
            for nxt in self.successors(cur):
                nxt = int(nxt)
                if nxt in parent:
                    continue
                parent[nxt] = cur
                if nxt == b:
                    path, c = [], parent[b]
                    while c != a:
                        path.append(c)
                        c = parent[c]
                    return path[::-1]
                q.append((nxt, h + 1))
        return None

    # ---------------- spatial queries ----------------
    @property
    def tree(self) -> STRtree:
        if self._tree is None:
            coords = np.stack([np.column_stack([self.x0, self.y0]), np.column_stack([self.x1, self.y1])], axis=1)
            self._tree = STRtree(shapely.linestrings(coords))
        return self._tree

    def query_radius(self, x: np.ndarray, y: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """All (point_idx, arc_idx) pairs with point-to-arc distance <= radius."""
        pts = shapely.points(x, y)
        pi, ai = self.tree.query(pts, predicate="dwithin", distance=radius)
        return pi.astype(np.int64), ai.astype(np.int64)

    def nearest_arc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = shapely.points(x, y)
        pi, ai = self.tree.query_nearest(pts, all_matches=False)
        out = np.full(len(x), -1, dtype=np.int64)
        out[pi] = ai
        return out

    # ---------------- output geometry ----------------
    def arc_lonlat(self, arc: int) -> list[list[float]]:
        """[[lat, lon], [lat, lon]] for map libraries (Leaflet/Folium use lat, lon)."""
        lon, lat = self.proj.to_lonlat([self.x0[arc], self.x1[arc]], [self.y0[arc], self.y1[arc]])
        return [[float(lat[0]), float(lon[0])], [float(lat[1]), float(lon[1])]]
