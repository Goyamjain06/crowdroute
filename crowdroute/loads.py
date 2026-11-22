"""Crowding model: ticket records -> on-board load by route, stretch, day type and time band.

Method (see docs/technical_note.md):
 1. Every ticket occupies the stretches between its start and end stop. The time at
    which the passenger is on each stretch is shifted by the scheduled run time
    from GTFS, so a 40-minute ride is counted in the right time bands.
 2. Observed on-board passengers are summed per (route, stretch, daytype, band)
    and divided by the number of days of that type -> sampled on-board per day.
 3. Divided by the number of scheduled trips crossing that stretch in that band
    (from GTFS) -> sampled load per bus.
 4. Multiplied by a global scale factor S = target daily ridership / observed
    daily ticketed passengers, because e-tickets are a small sample of all riders.
 5. Thin cells are shrunk towards the route's own profile (empirical Bayes), and
    every cell carries a confidence flag based on raw ticket support.
 6. Load / seats -> load factor -> commuter-facing crowding level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .gtfs import GTFS


def route_time_offsets(g: GTFS) -> dict[str, np.ndarray]:
    """Median scheduled seconds from the first stop to each stop position, per route."""
    out = {}
    for route, trips in g.route_trips.items():
        mat = np.vstack([arr for _, arr in trips])
        first = np.nanmin(mat, axis=1, keepdims=True)
        rel = mat - first
        med = np.nanmedian(rel, axis=0)
        # fill gaps by interpolation over position
        idx = np.arange(len(med))
        ok = ~np.isnan(med)
        if ok.sum() >= 2:
            med = np.interp(idx, idx[ok], med[ok])
        else:
            med = idx * 120.0  # fallback: 2 min per stop
        med = np.maximum.accumulate(med)  # monotone
        out[route] = med
    return out


def trips_per_band(g: GTFS, offsets: dict[str, np.ndarray]) -> pd.DataFrame:
    """Scheduled trips crossing each (route, seg) in each band. seg k = stretch k -> k+1."""
    R, SG, B = [], [], []
    for route, trips in g.route_trips.items():
        off = offsets[route]
        if len(off) < 2:
            continue
        for _, arr in trips:
            served = ~np.isnan(arr)
            if served.sum() < 2:
                continue
            lo, hi = np.argmax(served), len(served) - 1 - np.argmax(served[::-1])
            segs = np.arange(lo, hi)
            t = np.nanmin(arr) + off[segs] - off[lo]
            bands = ((t // 60) // config.TIMEBAND_MIN).astype(int) % config.N_BANDS
            R.append(np.full(len(segs), route, dtype=object)); SG.append(segs); B.append(bands)
    df = pd.DataFrame({"route": np.concatenate(R), "seg": np.concatenate(SG), "band": np.concatenate(B)})
    return df.groupby(["route", "seg", "band"]).size().rename("trips").reset_index()


def observed_vehicles_per_band(tickets: pd.DataFrame) -> pd.DataFrame:
    """Estimated buses in service per (route, daytype, band), averaged per day.

    Within a cell-day, k tickets carry a vehicle number and v distinct vehicles
    appear. If N buses were running and tickets fell on them at random,
    E[v] = N * (1 - (1 - 1/N)^k). We solve this for N (an occupancy / capture
    estimator), which corrects for buses that happened to sell no e-ticket.
    N is capped at 3v to keep it tame when k is tiny.
    """
    v = tickets.dropna(subset=["vehicle_number"])
    per_day = (v.groupby(["route", "daytype", "band", "date"]).vehicle_number
               .agg(v="nunique", k="size").reset_index())
    vv, kk = per_day.v.to_numpy(float), per_day.k.to_numpy(float)
    N = vv.copy()
    for _ in range(25):
        N = np.where(kk > vv, vv / (1 - (1 - 1 / np.maximum(N, 1.0)) ** kk), np.minimum(3 * vv, N * 1.5))
        N = np.minimum(N, 3 * vv)
    per_day["n_bus"] = N
    ndays = tickets.groupby("daytype").date.nunique()
    out = per_day.groupby(["route", "daytype", "band"]).n_bus.sum().reset_index()
    out["veh_per_day"] = out.n_bus / out.daytype.map(ndays)
    return out[["route", "daytype", "band", "veh_per_day"]]


def expand_tickets(tickets: pd.DataFrame, offsets: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per (ticket, stretch) with the band at which the passenger is on that stretch."""
    t = tickets.reset_index(drop=True)
    s = t.s_pos.to_numpy(); e = t.e_pos.to_numpy()
    n = e - s
    rep = np.repeat(np.arange(len(t)), n)
    seg = s[rep] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    # flatten all route offset arrays into one vector with per-route base index
    rcat = pd.Categorical(t.route)
    cats = list(rcat.categories)
    lens = np.array([len(offsets[r]) for r in cats])
    base = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = np.concatenate([offsets[r] for r in cats])
    code = rcat.codes[rep]
    shift = flat[base[code] + seg] - flat[base[code] + s[rep]]
    minute_of_day = (t.ts.dt.hour * 60 + t.ts.dt.minute).to_numpy()[rep] + shift / 60
    band = (minute_of_day // config.TIMEBAND_MIN).astype(np.int16) % config.N_BANDS
    del shift, minute_of_day
    dcat = pd.Categorical(t.daytype, categories=list(config.DAYTYPES))
    return pd.DataFrame({
        "route": pd.Categorical.from_codes(code, cats),
        "seg": seg.astype(np.int16), "band": band,
        "daytype": pd.Categorical.from_codes(dcat.codes[rep], dcat.categories),
        "day": t.ts.dt.day.to_numpy().astype(np.int8)[rep],
        "week": t.week.to_numpy().astype(np.int8)[rep],
        "pax": t.passenger_count.to_numpy().astype(np.int8)[rep],
        "is_ac": t.is_ac.to_numpy()[rep],
    })


def days_per_daytype(tickets: pd.DataFrame) -> dict[str, int]:
    return tickets.groupby("daytype").date.nunique().to_dict()


def scale_factor(raw_tickets: pd.DataFrame) -> tuple[float, dict]:
    """Global scale from sampled riders to all riders, with its derivation."""
    wk = raw_tickets[raw_tickets.daytype == "weekday"]
    per_day = wk.groupby("date").passenger_count.sum()
    observed = float(per_day.mean())
    S = config.TARGET_WEEKDAY_RIDERSHIP / observed
    return S, {"observed_weekday_pax_per_day": round(observed), "n_weekdays": int(per_day.size),
               "target_weekday_ridership": config.TARGET_WEEKDAY_RIDERSHIP,
               "implied_sampling_rate": round(1 / S, 4), "scale_factor": round(S, 2)}


def crowd_level(lf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = np.full(len(lf), config.CROWD_LEVELS[-1][1]); labels = np.full(len(lf), config.CROWD_LEVELS[-1][2], dtype=object)
    for ub, code, label in reversed(config.CROWD_LEVELS):
        m = lf <= ub
        codes[m] = code; labels[m] = label
    return codes, labels


def confidence(n: np.ndarray) -> np.ndarray:
    out = np.full(len(n), "high", dtype=object)
    out[n < config.CONF_THRESHOLDS["medium"]] = "medium"
    out[n < config.CONF_THRESHOLDS["low"]] = "low"
    return out


def build_crowding(tickets: pd.DataFrame, raw_for_scale: pd.DataFrame, g: GTFS,
                   offsets: dict[str, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict]:
    offsets = offsets or route_time_offsets(g)
    ex = expand_tickets(tickets, offsets)
    ndays = days_per_daytype(tickets)
    S, scale_info = scale_factor(raw_for_scale)

    # observed on-board pax and ticket support per cell
    cell = ex.groupby(["route", "seg", "daytype", "band"], observed=True).agg(pax=("pax", "sum"), n=("pax", "size"),
                                                               ac_share=("is_ac", "mean")).reset_index()
    cell["route"] = cell.route.astype(str); cell["daytype"] = cell.daytype.astype(str)
    cell["seg"] = cell.seg.astype(int); cell["band"] = cell.band.astype(int)
    cell["pax_per_day"] = cell.pax / cell.daytype.map(ndays)

    tpb = trips_per_band(g, offsets)
    cell = cell.merge(tpb, on=["route", "seg", "band"], how="left")
    cell["no_schedule"] = cell.trips.isna()
    cell["trips_gtfs"] = cell.trips.fillna(0)
    veh = observed_vehicles_per_band(tickets)
    cell = cell.merge(veh, on=["route", "daytype", "band"], how="left")
    cell["veh_per_day"] = cell.veh_per_day.fillna(0)
    # frequency = max(scheduled trips, buses proven by tickets); never below 1
    cell["trips"] = np.maximum(np.maximum(cell.trips_gtfs, cell.veh_per_day), 1.0)
    cell["load_sampled"] = cell.pax_per_day / cell.trips        # per bus, before scaling
    ac_mask = cell.ac_share >= 0.5
    crush_v = np.where(ac_mask, config.CAPACITY["ac"]["crush"], config.CAPACITY["nonac"]["crush"])
    well = (cell.daytype == "weekday") & (cell.n >= config.CONF_THRESHOLDS["low"])
    q = float(np.quantile((cell.load_sampled / crush_v)[well], config.CAL_QUANTILE))
    S_cap = 1.0 / q
    scale_info.update({"scale_ridership": round(S, 2), "scale_capacity": round(S_cap, 2),
                       "calibration_quantile": config.CAL_QUANTILE, "method": config.SCALE_METHOD})
    S = S_cap if config.SCALE_METHOD == "capacity" else S
    scale_info["scale_factor"] = round(S, 2)
    scale_info["implied_sampling_rate"] = round(1 / S, 4)
    cell["load_raw"] = cell.load_sampled * S

    # Empirical-Bayes shrinkage towards the route x daytype x band profile (mean over stretches),
    # then towards the route x daytype mean. Weights are ticket counts.
    def wmean(df, keys):
        tmp = df.assign(_w=df.load_raw * df.n).groupby(keys)[["_w", "n"]].sum()
        return (tmp._w / tmp.n).rename("prior").reset_index()
    p1 = wmean(cell, ["route", "daytype", "band"]).rename(columns={"prior": "prior_band"})
    p2 = wmean(cell, ["route", "daytype"]).rename(columns={"prior": "prior_route"})
    cell = cell.merge(p1, on=["route", "daytype", "band"]).merge(p2, on=["route", "daytype"])
    K = config.SHRINK_K
    cell["load"] = (cell.n * cell.load_raw + K * cell.prior_band) / (cell.n + K)

    seats = np.where(cell.ac_share >= 0.5, config.CAPACITY["ac"]["seats"], config.CAPACITY["nonac"]["seats"])
    crush = np.where(cell.ac_share >= 0.5, config.CAPACITY["ac"]["crush"], config.CAPACITY["nonac"]["crush"])
    cell["seats"] = seats
    cell["load"] = np.minimum(cell.load, crush * config.MAX_LOAD_FACTOR)
    cell["load_factor"] = cell.load / seats
    cell["level"], cell["label"] = crowd_level(cell.load_factor.to_numpy())
    cell["confidence"] = confidence(cell.n.to_numpy())
    cell["from_stop"] = [g.route_seq[r][s] for r, s in zip(cell.route, cell.seg)]
    cell["to_stop"] = [g.route_seq[r][s + 1] if s + 1 < len(g.route_seq[r]) else g.route_seq[r][s]
                       for r, s in zip(cell.route, cell.seg)]
    cols = ["route", "seg", "from_stop", "to_stop", "daytype", "band", "n", "pax_per_day", "trips",
            "trips_gtfs", "veh_per_day", "no_schedule", "load_raw", "load", "seats", "load_factor", "level", "label", "confidence"]
    info = {"scale": scale_info, "days_per_daytype": ndays, "n_cells": len(cell),
            "cells_no_schedule": int(cell.no_schedule.sum()),
            "level_share": cell.label.value_counts(normalize=True).round(3).to_dict(),
            "confidence_share": cell.confidence.value_counts(normalize=True).round(3).to_dict()}
    return cell[cols], info
