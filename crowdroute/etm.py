"""Ticket loader: cleans the ONDC e-ticketing CSV and positions each ticket on a GTFS route.

Output of `prepare_tickets()` is one row per ticket with the route and the
(start, end) stop positions on that route's canonical stop sequence.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from . import config
from .gtfs import GTFS, normalize_name


def _daytype(ts: pd.Series) -> pd.Series:
    d = ts.dt.strftime("%Y-%m-%d")
    dow = ts.dt.dayofweek
    out = np.where(dow == 6, "sunday", np.where(dow == 5, "saturday", "weekday"))
    out = np.where(d.isin(config.HOLIDAYS), "sunday", out)
    return pd.Series(out, index=ts.index)


def _route_candidates(name: str) -> list[str]:
    """Variants to try when a ticket route string is not a GTFS route name."""
    cands = [name]
    base = re.sub(r"(UP|DOWN)\d+$", lambda m: m.group(1), name)   # 522BDOWN2 -> 522BDOWN
    cands.append(base)
    cands.append(re.sub(r"(L?STL)(UP|DOWN)$", r"\2", base))         # 567BSTLUP -> 567BUP
    cands.append(re.sub(r"^ML", "", base))                          # ML96UP -> 96UP
    return list(dict.fromkeys(cands))


class StopMatcher:
    """Matches free-text stop names to stop_ids on a given route (cached)."""

    def __init__(self, g: GTFS):
        self.g = g
        self.cache: dict[tuple[str, str], int | None] = {}
        self.stats = {"exact": 0, "fuzzy": 0, "miss": 0}

    GENERIC = {"hospital", "station", "metro", "crossing", "depot", "village", "road", "nagar",
               "pur", "block", "market", "terminal", "isbt", "chowk", "colony", "sector", "phase",
               "extension", "extn", "border", "government", "govt", "hostel", "model", "public", "bus", "stand", "gate", "school", "college", "mandi",
               "mor", "marg", "red", "light", "flyover", "west", "east", "north", "south", "new", "old",
               "main", "the", "of", "and", "t", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
               "n", "o", "p", "q", "r", "s", "u", "v", "w", "x", "y", "z"}

    @classmethod
    def _distinctive_ok(cls, query: str, cand: str) -> bool:
        """Require that a distinctive (non-generic) token of the query appears in the candidate."""
        q_tokens = [t for t in query.split() if t not in cls.GENERIC and not t.isdigit()]
        if not q_tokens:  # only generic words: demand a near-exact match
            return fuzz.token_sort_ratio(query, cand) >= 90
        c_tokens = cand.split()
        cand_joined = cand.replace(" ", "")
        matched = sum(any(fuzz.ratio(q, c) >= 85 for c in c_tokens) or (len(q) >= 4 and q in cand_joined)
                      for q in q_tokens)
        return matched / len(q_tokens) >= 0.5

    def match(self, route: str, name: str) -> int | None:
        key = (route, name)
        if key in self.cache:
            return self.cache[key]
        norm = normalize_name(name)
        names = self.g.route_norm_names[route]
        sid = names.get(norm)
        if sid is not None:
            self.stats["exact"] += 1
        else:
            sid = None
            hits = process.extract(norm, list(names), scorer=fuzz.token_set_ratio, score_cutoff=config.FUZZY_MIN_SCORE, limit=5)
            for cand, score, _ in hits:
                if self._distinctive_ok(norm, cand):
                    sid = names[cand]
                    break
            if sid is not None:
                self.stats["fuzzy"] += 1
            else:
                self.stats["miss"] += 1
        self.cache[key] = sid
        return sid


def load_raw(path=config.RAW_ETM) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NULL"])
    df["ts"] = pd.to_datetime(df.created_at, utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
    df = df.drop(columns=["created_at", "ticket_type_id"])
    df["date"] = df.ts.dt.strftime("%Y-%m-%d")
    df["daytype"] = _daytype(df.ts)
    df["band"] = (df.ts.dt.hour * 60 + df.ts.dt.minute) // config.TIMEBAND_MIN
    df["week"] = df.ts.dt.isocalendar().week.astype(int)
    return df


def prepare_tickets(df: pd.DataFrame, g: GTFS) -> tuple[pd.DataFrame, dict]:
    """Attach route + stop positions. Returns (clean_df, report)."""
    report: dict = {"n_raw": len(df)}

    # 1. Resolve route names
    route_map = {}
    for r in df.route.unique():
        for c in _route_candidates(r):
            if c in g.route_seq:
                route_map[r] = c
                break
    df["route_g"] = df.route.map(route_map)
    report["route_unmatched_rows"] = int(df.route_g.isna().sum())
    df = df.dropna(subset=["route_g"]).copy()

    # 2. Match stops on the route (and on the opposite direction, for direction fixes)
    m = StopMatcher(g)
    pairs = df[["route_g", "start_stop_name", "end_stop_name"]].drop_duplicates()
    rows = []
    for route, s_name, e_name in pairs.itertuples(index=False):
        s_id, e_id = m.match(route, s_name), m.match(route, e_name)
        pos = g.route_pos[route]
        s_pos = pos.get(s_id, -1) if s_id is not None else -1
        e_pos = pos.get(e_id, -1) if e_id is not None else -1
        use_route, flipped = route, False
        if s_pos >= 0 and e_pos >= 0 and s_pos > e_pos:
            # Ticket direction contradicts route direction: try the opposite route.
            opp = g.opposite_route(route)
            if opp:
                s2, e2 = m.match(opp, s_name), m.match(opp, e_name)
                p2 = g.route_pos[opp]
                if s2 is not None and e2 is not None and p2[s2] < p2[e2]:
                    use_route, s_pos, e_pos, flipped = opp, p2[s2], p2[e2], True
        rows.append((route, s_name, e_name, use_route, s_pos, e_pos, flipped))
    res = pd.DataFrame(rows, columns=["route_g", "start_stop_name", "end_stop_name", "route_use", "s_pos", "e_pos", "flipped"])
    df = df.merge(res, on=["route_g", "start_stop_name", "end_stop_name"], how="left")
    report["stop_match_stats"] = m.stats

    ok = (df.s_pos >= 0) & (df.e_pos >= 0) & (df.s_pos < df.e_pos)
    report["rows_start_unmatched"] = int((df.s_pos < 0).sum())
    report["rows_end_unmatched"] = int((df.e_pos < 0).sum())
    report["rows_same_or_reversed"] = int(((df.s_pos >= 0) & (df.e_pos >= 0) & (df.s_pos >= df.e_pos)).sum())
    report["rows_direction_fixed"] = int(df.flipped.sum())
    clean = df[ok].copy()
    clean["route"] = clean.route_use
    clean = clean[["ts", "date", "daytype", "band", "week", "route", "s_pos", "e_pos",
                   "passenger_count", "amount", "vehicle_number", "is_ac"]]
    report["n_clean"] = len(clean)
    report["clean_fraction"] = round(len(clean) / report["n_raw"], 4)
    return clean, report
