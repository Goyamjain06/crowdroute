"""FastAPI service: journey planning with crowding, plus operator views.

Run:  uvicorn crowdroute.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .planner import get_planner, daytype_for

app = FastAPI(title="CrowdRoute API", version="0.1.0",
              description="Crowdedness-aware bus journey planner for Delhi (Transport Stack Hackathon 2026, Problem 2).")

WEB = config.ROOT / "web"


@app.on_event("startup")
def _warm():
    get_planner()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/meta")
def meta():
    info = json.load(open(config.PROCESSED / "model_info.json"))
    return {"model": info, "timeband_min": config.TIMEBAND_MIN, "levels": [
        {"code": c, "label": l, "max_load_factor": ub} for ub, c, l in config.CROWD_LEVELS], "capacity": config.CAPACITY}


@app.get("/api/stops/search")
def stops_search(q: str = Query(..., min_length=2), limit: int = 8):
    p, _ = get_planner()
    return p.search_stops(q, limit)


@app.get("/api/plan")
def plan(o_lat: float, o_lon: float, d_lat: float, d_lon: float,
         when: str = Query(..., description="Local time, e.g. 2026-10-28T18:00")):
    p, off = get_planner()
    try:
        ts = pd.Timestamp(when)
    except Exception:
        raise HTTPException(400, "when must be an ISO datetime")
    res = p.plan_with_alternatives((o_lat, o_lon), (d_lat, d_lon), ts, off)
    # attach geometry for the map
    for j in res["journeys"] + res["calmer_later"]:
        for leg in j["legs"]:
            leg["path"] = _leg_path(p, leg["route"], leg["from_stop"], leg["to_stop"])
    return res


def _leg_path(p, route, from_stop, to_stop):
    seq = p.g.route_seq[route]; pos = p.g.route_pos[route]
    return [[*p.g.stop_xy[s]] for s in seq[pos[from_stop]: pos[to_stop] + 1]]


@lru_cache(maxsize=256)
def _band_layer(daytype: str, band: int, min_conf: str):
    p, _ = get_planner()
    df = p.crowd.df
    order = {"low": 0, "medium": 1, "high": 2}
    sel = df[(df.daytype == daytype) & (df.band == band) & (df.confidence.map(order) >= order[min_conf])]
    feats = []
    for r in sel.itertuples(index=False):
        a, b = p.g.stop_xy[r.from_stop], p.g.stop_xy[r.to_stop]
        feats.append({"r": r.route, "s": int(r.seg), "a": [a[0], a[1]], "b": [b[0], b[1]],
                      "lf": round(float(r.load_factor), 2), "lv": int(r.level), "n": int(r.n), "c": r.confidence[0],
                      "load": round(float(r.load))})
    return feats


@app.get("/api/crowding/heatmap")
def heatmap(daytype: str = "weekday", band: int = 18, min_conf: str = "medium"):
    if daytype not in config.DAYTYPES or not 0 <= band < config.N_BANDS:
        raise HTTPException(400, "bad daytype/band")
    return {"daytype": daytype, "band": band, "segments": _band_layer(daytype, band, min_conf)}


@app.get("/api/crowding/route/{route}")
def route_profile(route: str, daytype: str = "weekday"):
    p, _ = get_planner()
    if route not in p.g.route_seq:
        raise HTTPException(404, "unknown route")
    df = p.crowd.df
    sel = df[(df.route == route) & (df.daytype == daytype)]
    seq = p.g.route_seq[route]
    grid = np.full((len(seq) - 1, config.N_BANDS), np.nan)
    conf = np.full((len(seq) - 1, config.N_BANDS), "", dtype=object)
    for r in sel.itertuples(index=False):
        grid[r.seg, r.band] = r.load_factor; conf[r.seg, r.band] = r.confidence[0]
    return {"route": route, "daytype": daytype,
            "stops": [{"id": s, "name": p.g.stop_name[s], "lat": p.g.stop_xy[s][0], "lon": p.g.stop_xy[s][1]} for s in seq],
            "load_factor": [[None if np.isnan(x) else round(float(x), 2) for x in row] for row in grid],
            "confidence": conf.tolist(), "n_tickets": int(sel.n.sum())}


@app.get("/api/crowding/top")
def top_overloaded(daytype: str = "weekday", band: int | None = None, limit: int = 25):
    """Operator view: most overloaded route stretches (well-supported cells only)."""
    p, _ = get_planner()
    df = p.crowd.df
    sel = df[(df.daytype == daytype) & (df.confidence != "low")]
    if band is not None:
        sel = sel[sel.band == band]
    # one row per route: its worst stretch, so one overloaded route doesn't fill the table
    sel = sel.sort_values("load_factor", ascending=False).drop_duplicates("route").head(limit)
    return [{"route": r.route, "from": p.g.stop_name[r.from_stop], "to": p.g.stop_name[r.to_stop],
             "band": int(r.band), "time": f"{(r.band * config.TIMEBAND_MIN) // 60:02d}:{(r.band * config.TIMEBAND_MIN) % 60:02d}",
             "load": round(float(r.load)), "load_factor": round(float(r.load_factor), 2), "label": r.label,
             "buses_per_band": round(float(r.trips), 1), "n_tickets": int(r.n), "confidence": r.confidence}
            for r in sel.itertuples(index=False)]


@app.get("/api/crowding/hourly")
def hourly(daytype: str = "weekday"):
    """Network-wide load by time band (ticket-weighted) for the dashboard chart."""
    p, _ = get_planner()
    df = p.crowd.df
    sel = df[(df.daytype == daytype) & (df.confidence != "low")]
    w = sel.assign(w=sel.load * sel.n).groupby("band")[["w", "n"]].sum()
    share = sel.groupby("band").apply(lambda d: float(np.average(d.level >= 3, weights=d.n)), include_groups=False)
    return [{"band": int(b), "time": f"{(b * config.TIMEBAND_MIN) // 60:02d}:{(b * config.TIMEBAND_MIN) % 60:02d}",
             "mean_load": round(float(w.w[b] / w.n[b]), 1), "crowded_share": round(float(share[b]), 3)}
            for b in w.index]


@app.get("/api/routes")
def routes():
    p, _ = get_planner()
    counts = p.crowd.df.groupby("route").n.sum().sort_values(ascending=False)
    return [{"route": r, "n_tickets": int(n)} for r, n in counts.items()]


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
