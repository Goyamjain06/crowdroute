"""Journey planner on static GTFS with crowding overlay.

Deliberately simple (the brief says the crowding model is the main contribution):
  * origin/destination = a point (lat, lon) or a stop name; stops within WALK_RADIUS are candidates
  * direct journeys and one-transfer journeys
  * departure = next scheduled trip after the requested time (from GTFS stop_times)
  * each leg is annotated with the crowding of the stretches it rides, in the
    time band the bus is actually on them
  * alternatives: the same journeys 15/30/45 minutes later, so the user can
    trade time for comfort
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .gtfs import GTFS, get_gtfs, haversine_km, normalize_name

WALK_RADIUS_KM = 0.6
WALK_KMH = 4.5
TRANSFER_PENALTY_S = 300
MAX_RESULTS = 6
MAX_WAIT_MIN = 60
SHIFTS_MIN = (15, 30, 45)


class CrowdIndex:
    """Fast lookup of crowding cells: (route, seg, daytype, band) -> row."""

    def __init__(self, cell: pd.DataFrame):
        self.df = cell
        self.lookup: dict[tuple, dict] = {}
        # route x daytype x band fallback profile for stretches with no cell
        self.route_band = cell.groupby(["route", "daytype", "band"]).agg(
            load=("load", "mean"), load_factor=("load_factor", "mean"), n=("n", "sum")).to_dict("index")
        self.route_day = cell.groupby(["route", "daytype"]).agg(
            load=("load", "mean"), load_factor=("load_factor", "mean"), n=("n", "sum")).to_dict("index")
        cols = ["load", "load_factor", "level", "label", "confidence", "n", "trips"]
        for key, row in zip(zip(cell.route, cell.seg, cell.daytype, cell.band), cell[cols].to_dict("records")):
            self.lookup[key] = row

    def get(self, route: str, seg: int, daytype: str, band: int) -> dict:
        r = self.lookup.get((route, seg, daytype, band))
        if r:
            return r
        for d in (1, -1, 2, -2):
            r = self.lookup.get((route, seg, daytype, (band + d) % config.N_BANDS))
            if r:
                return {**r, "confidence": "low", "source": "neighbour_band"}
        p = self.route_band.get((route, daytype, band)) or self.route_day.get((route, daytype))
        if p:
            lf = float(p["load_factor"])
            return {"load": float(p["load"]), "load_factor": lf, "level": _level(lf)[0], "label": _level(lf)[1],
                    "confidence": "low", "n": int(p["n"]), "trips": None, "source": "route_profile"}
        return {"load": None, "load_factor": None, "level": 0, "label": "No data", "confidence": "none", "n": 0, "trips": None}


def _level(lf: float):
    for ub, code, label in config.CROWD_LEVELS:
        if lf <= ub:
            return code, label
    return config.CROWD_LEVELS[-1][1], config.CROWD_LEVELS[-1][2]


def daytype_for(ts: pd.Timestamp) -> str:
    if ts.strftime("%Y-%m-%d") in config.HOLIDAYS or ts.dayofweek == 6:
        return "sunday"
    return "saturday" if ts.dayofweek == 5 else "weekday"


@dataclass
class Leg:
    route: str
    from_stop: int
    to_stop: int
    from_name: str
    to_name: str
    depart: str
    arrive: str
    n_stops: int
    in_vehicle_min: float
    crowd_max: dict
    crowd_avg_lf: float
    stretch_levels: list[int]
    confidence: str
    trip_id: str
    dep_sec: float = 0.0
    arr_sec: float = 0.0


MAX_LEG_MIN = 150


@dataclass
class Journey:
    legs: list[Leg]
    depart: str
    arrive: str
    total_min: float
    wait_min: float
    walk_min: float
    transfers: int
    crowd_level: int
    crowd_label: str
    crowd_lf: float
    confidence: str
    summary: str


class Planner:
    def __init__(self, g: GTFS, crowd: CrowdIndex):
        self.g = g
        self.crowd = crowd
        self.stops = g.stops.copy()
        self.stops["norm"] = self.stops.stop_name.map(normalize_name)

    # ---- geography -------------------------------------------------------
    def nearby_stops(self, lat: float, lon: float, radius_km: float = WALK_RADIUS_KM) -> list[tuple[int, float]]:
        d = haversine_km(lat, lon, self.stops.stop_lat.to_numpy(), self.stops.stop_lon.to_numpy())
        idx = np.where(d <= radius_km)[0]
        if len(idx) == 0:
            idx = np.argsort(d)[:3]
        return sorted([(int(self.stops.stop_id.iloc[i]), float(d[i])) for i in idx], key=lambda x: x[1])[:12]

    def search_stops(self, q: str, limit: int = 8) -> list[dict]:
        from rapidfuzz import process, fuzz
        qn = normalize_name(q)
        names = self.stops.norm.tolist()
        hits = process.extract(qn, names, scorer=fuzz.WRatio, limit=limit)
        out = []
        for _, score, i in hits:
            r = self.stops.iloc[i]
            out.append({"stop_id": int(r.stop_id), "name": r.stop_name, "lat": float(r.stop_lat),
                        "lon": float(r.stop_lon), "score": score, "routes": sorted(self.g.stop_routes.get(int(r.stop_id), []))[:20]})
        return out

    # ---- schedule --------------------------------------------------------
    def next_trip(self, route: str, s_pos: int, e_pos: int, after_sec: float):
        """Earliest trip on route serving both positions with departure >= after_sec."""
        best = None
        for tid, arr in self.g.route_trips[route]:
            d, a = arr[s_pos], arr[e_pos]
            if np.isnan(d) or np.isnan(a) or d < after_sec or a <= d:
                continue
            if best is None or d < best[1]:
                best = (tid, float(d), float(a))
        return best

    # ---- crowding for a ride ---------------------------------------------
    def ride_crowding(self, route: str, s_pos: int, e_pos: int, depart_sec: float, daytype: str, offsets):
        off = offsets[route]
        levels, lfs, confs, worst = [], [], [], None
        for seg in range(s_pos, e_pos):
            t = depart_sec + (off[seg] - off[s_pos])
            band = int((t // 60) // config.TIMEBAND_MIN) % config.N_BANDS
            c = self.crowd.get(route, seg, daytype, band)
            if c["load_factor"] is None:
                continue
            levels.append(int(c["level"])); lfs.append(float(c["load_factor"])); confs.append(c["confidence"])
            if worst is None or c["load_factor"] > worst["load_factor"]:
                worst = {**c, "seg": seg, "from": self.g.stop_name[self.g.route_seq[route][seg]],
                         "to": self.g.stop_name[self.g.route_seq[route][seg + 1]]}
        if not lfs:
            return {"load_factor": None, "level": 0, "label": "No data", "confidence": "none"}, 0.0, [], "none"
        conf = "high" if confs.count("high") >= len(confs) / 2 else ("medium" if "low" not in confs or confs.count("low") < len(confs) / 2 else "low")
        return worst, float(np.mean(lfs)), levels, conf

    # ---- planning --------------------------------------------------------
    def plan(self, o_lat, o_lon, d_lat, d_lon, when: pd.Timestamp, offsets, max_results=MAX_RESULTS) -> list[Journey]:
        g = self.g
        daytype = daytype_for(when)
        t0 = when.hour * 3600 + when.minute * 60
        o_stops = self.nearby_stops(o_lat, o_lon)
        d_stops = self.nearby_stops(d_lat, d_lon)
        d_set = {s: dist for s, dist in d_stops}
        journeys: list[Journey] = []
        seen = set()

        def make_leg(route, s_pos, e_pos, after):
            nt = self.next_trip(route, s_pos, e_pos, after)
            if not nt:
                return None
            tid, dep, arr = nt
            if (arr - dep) / 60 > MAX_LEG_MIN:
                return None
            worst, avg_lf, levels, conf = self.ride_crowding(route, s_pos, e_pos, dep, daytype, offsets)
            fs, ts_ = g.route_seq[route][s_pos], g.route_seq[route][e_pos]
            return Leg(route, fs, ts_, g.stop_name[fs], g.stop_name[ts_], _hms(dep), _hms(arr), e_pos - s_pos,
                       round((arr - dep) / 60, 1), worst, round(avg_lf, 2), levels, conf, tid, dep, arr)

        # direct
        for os_, odist in o_stops:
            walk1 = odist / WALK_KMH * 3600
            for route in g.stop_routes.get(os_, ()):
                pos = g.route_pos[route]
                sp = pos[os_]
                for ds_, ddist in d_stops:
                    ep = pos.get(ds_)
                    if ep is None or ep <= sp:
                        continue
                    key = (route, os_, ds_)
                    if key in seen:
                        continue
                    seen.add(key)
                    leg = make_leg(route, sp, ep, t0 + walk1)
                    if leg:
                        journeys.append(self._assemble([leg], t0, walk1, ddist / WALK_KMH * 3600))

        # one transfer
        if len(journeys) < max_results:
            for os_, odist in o_stops[:6]:
                walk1 = odist / WALK_KMH * 3600
                for rA in g.stop_routes.get(os_, ()):
                    seqA = g.route_seq[rA]; sp = g.route_pos[rA][os_]
                    for xp in range(sp + 1, len(seqA)):
                        x = seqA[xp]
                        for rB in g.stop_routes.get(x, ()):
                            if rB == rA or rB == g.opposite_route(rA):
                                continue
                            posB = g.route_pos[rB]; xb = posB[x]
                            for ds_, ddist in d_stops[:6]:
                                ep = posB.get(ds_)
                                if ep is None or ep <= xb:
                                    continue
                                key = (rA, os_, x, rB, ds_)
                                if key in seen:
                                    continue
                                seen.add(key)
                                leg1 = make_leg(rA, sp, xp, t0 + walk1)
                                if not leg1:
                                    continue
                                leg2 = make_leg(rB, xb, ep, leg1.arr_sec + TRANSFER_PENALTY_S)
                                if not leg2:
                                    continue
                                journeys.append(self._assemble([leg1, leg2], t0, walk1, ddist / WALK_KMH * 3600))
                        if len(journeys) > 400:
                            break
        journeys = [j for j in _dedupe(journeys) if j.wait_min <= MAX_WAIT_MIN]
        journeys.sort(key=lambda j: (j.total_min + 8 * j.transfers))
        return journeys[:max_results]

    def _assemble(self, legs: list[Leg], t0: float, walk1: float, walk2: float) -> Journey:
        dep = legs[0].dep_sec; arr = legs[-1].arr_sec + walk2
        wait = dep - (t0 + walk1)
        with_lf = [l for l in legs if l.crowd_max.get("load_factor") is not None]
        lfs = [l.crowd_max["load_factor"] for l in with_lf]
        worst_lf = max(lfs) if lfs else None
        worst_leg = with_lf[int(np.argmax(lfs))] if lfs else None
        level, label = _level(worst_lf) if worst_lf is not None else (0, "No data")
        confs = [l.confidence for l in legs]
        conf = "low" if "low" in confs or "none" in confs else ("medium" if "medium" in confs else "high")
        routes = " → ".join(l.route for l in legs)
        return Journey(legs, _hms(dep), _hms(arr), round((arr - t0) / 60, 1), round(wait / 60, 1),
                       round((walk1 + walk2) / 60, 1), len(legs) - 1, level, label,
                       round(worst_lf, 2) if worst_lf is not None else None, conf,
                       f"{routes}: {label.lower()}" + (f" on the {worst_leg.crowd_max['from']} stretch" if worst_leg and level >= 3 and worst_leg.crowd_max.get("from") else ""))

    def plan_with_alternatives(self, o, d, when: pd.Timestamp, offsets) -> dict:
        base = self.plan(*o, *d, when, offsets)
        alts = []
        for m in SHIFTS_MIN:
            js = self.plan(*o, *d, when + pd.Timedelta(minutes=m), offsets, max_results=3)
            for j in js:
                alts.append({"shift_min": m, **asdict(j)})
        # keep alternatives that are calmer than the fastest option now (the one a commuter would default to)
        ref_lvl = base[0].crowd_level if base else 0
        calmer = [a for a in alts if a["crowd_level"] and a["crowd_level"] < ref_lvl and a["confidence"] != "none"]
        calmer.sort(key=lambda a: (a["transfers"], a["crowd_level"], a["shift_min"], a["total_min"]))
        return {"daytype": daytype_for(when), "requested": when.strftime("%Y-%m-%d %H:%M"),
                "journeys": [asdict(j) for j in base], "calmer_later": calmer[:3]}


def _hms(sec: float) -> str:
    sec = int(round(sec)); h, m = divmod(sec // 60, 60)
    return f"{h % 24:02d}:{m:02d}"


def _sec(hms: str) -> float:
    h, m = hms.split(":")
    return int(h) * 3600 + int(m) * 60


def _dedupe(js: list[Journey]) -> list[Journey]:
    out, seen = [], set()
    for j in js:
        k = tuple(l.route for l in j.legs)
        if k in seen:
            continue
        seen.add(k); out.append(j)
    return out


@lru_cache(maxsize=1)
def get_planner() -> tuple[Planner, dict]:
    from .loads import route_time_offsets
    g = get_gtfs()
    cell = pd.read_parquet(config.PROCESSED / "crowding.parquet")
    planner = Planner(g, CrowdIndex(cell))
    return planner, route_time_offsets(g)
