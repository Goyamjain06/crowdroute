"""GTFS loader: builds per-route stop sequences, trip schedules and a stop index."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config


def normalize_name(s: str) -> str:
    """Canonical form for stop names: lowercase, alnum only, common tokens unified."""
    s = str(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\b(xing|x-ing|crossing|xng)\b", "crossing", s)
    s = re.sub(r"\b(stn|station)\b", "station", s)
    s = re.sub(r"\b(terminal|tml|term)\b", "terminal", s)
    s = re.sub(r"\b(nh8|nh 8)\b", "nh8", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # collapse initials: "s j hospital" -> "sj hospital"
    s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)
    return s


def _hms_to_sec(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


@dataclass
class GTFS:
    routes: pd.DataFrame
    stops: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    # route_short_name -> canonical stop sequence (list of stop_id) from the longest trip
    route_seq: dict[str, list[int]] = field(default_factory=dict)
    # route_short_name -> {stop_id: seq_index}
    route_pos: dict[str, dict[int, int]] = field(default_factory=dict)
    # route_short_name -> list of (trip_id, [departure_sec per seq index])
    route_trips: dict[str, list[tuple[str, np.ndarray]]] = field(default_factory=dict)
    # stop_id -> set(route_short_name)
    stop_routes: dict[int, set[str]] = field(default_factory=dict)
    stop_name: dict[int, str] = field(default_factory=dict)
    stop_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    route_id_by_name: dict[str, int] = field(default_factory=dict)
    # route_short_name -> {normalized stop name: stop_id} for stops on that route
    route_norm_names: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def route_names(self) -> list[str]:
        return list(self.route_seq)

    def opposite_route(self, name: str) -> str | None:
        if name.endswith("UP"):
            cand = name[:-2] + "DOWN"
        elif name.endswith("DOWN"):
            cand = name[:-4] + "UP"
        else:
            return None
        return cand if cand in self.route_seq else None


def load_gtfs(path=config.RAW_GTFS) -> GTFS:
    routes = pd.read_csv(path / "routes.txt")
    stops = pd.read_csv(path / "stops.txt")
    trips = pd.read_csv(path / "trips.txt")
    st = pd.read_csv(path / "stop_times.txt", usecols=["trip_id", "departure_time", "stop_id", "stop_sequence"])
    st["dep_sec"] = st.departure_time.map(_hms_to_sec)
    st = st.sort_values(["trip_id", "stop_sequence"])

    g = GTFS(routes, stops, trips, st)
    g.stop_name = dict(zip(stops.stop_id, stops.stop_name))
    g.stop_xy = dict(zip(stops.stop_id, zip(stops.stop_lat, stops.stop_lon)))
    g.route_id_by_name = dict(zip(routes.route_short_name, routes.route_id))
    rname_by_id = dict(zip(routes.route_id, routes.route_short_name))

    trip_route = dict(zip(trips.trip_id, trips.route_id.map(rname_by_id)))
    st["route"] = st.trip_id.map(trip_route)

    # Canonical stop sequence per route = stop sequence of the trip with most stops
    # (FULL trips); SHORT trips are sub-sequences and are positioned against it.
    for route, grp in st.groupby("route", sort=False):
        per_trip = {t: (sub.stop_id.to_numpy(), sub.dep_sec.to_numpy()) for t, sub in grp.groupby("trip_id", sort=False)}
        longest = max(per_trip, key=lambda t: len(per_trip[t][0]))
        seq = list(dict.fromkeys(per_trip[longest][0].tolist()))  # dedupe loops, keep order
        pos = {s: i for i, s in enumerate(seq)}
        g.route_seq[route] = seq
        g.route_pos[route] = pos
        trips_list = []
        for t, (sids, deps) in per_trip.items():
            arr = np.full(len(seq), np.nan)
            for sid, d in zip(sids, deps):
                if sid in pos:
                    arr[pos[sid]] = d
            trips_list.append((t, arr))
        trips_list.sort(key=lambda x: np.nanmin(x[1]))
        g.route_trips[route] = trips_list
        for s in seq:
            g.stop_routes.setdefault(s, set()).add(route)
        g.route_norm_names[route] = {normalize_name(g.stop_name[s]): s for s in seq}
    return g


@lru_cache(maxsize=1)
def get_gtfs() -> GTFS:
    return load_gtfs()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
