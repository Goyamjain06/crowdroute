"""Unit tests on small synthetic inputs (no raw data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crowdroute.gtfs import normalize_name, GTFS
from crowdroute.etm import StopMatcher, _route_candidates
from crowdroute.loads import crowd_level, confidence, expand_tickets


def test_normalize_name():
    assert normalize_name("Laxmi Nagar X-ing") == normalize_name("Laxmi Nagar Xing")
    assert normalize_name("S.J. Hospital") == "sj hospital"
    assert normalize_name("Azad Pur Metro Stn") == "azad pur metro station"


def test_route_candidates():
    assert "522BDOWN" in _route_candidates("522BDOWN2")
    assert "567BUP" in _route_candidates("567BSTLUP")
    assert "96UP" in _route_candidates("ML96UP")


def _tiny_gtfs():
    g = GTFS(routes=pd.DataFrame(), stops=pd.DataFrame(), trips=pd.DataFrame(), stop_times=pd.DataFrame())
    g.route_seq = {"1UP": [1, 2, 3, 4], "1DOWN": [4, 3, 2, 1]}
    g.route_pos = {r: {s: i for i, s in enumerate(seq)} for r, seq in g.route_seq.items()}
    g.stop_name = {1: "Alpha Terminal", 2: "Beta Crossing", 3: "Gamma Hospital", 4: "Delta Depot"}
    g.route_norm_names = {r: {normalize_name(g.stop_name[s]): s for s in seq} for r, seq in g.route_seq.items()}
    return g


def test_stop_matcher_rejects_wrong_distinctive_token():
    m = StopMatcher(_tiny_gtfs())
    assert m.match("1UP", "Gamma Hosp.") == 3
    assert m.match("1UP", "Beta X-ing") == 2
    assert m.match("1UP", "Omega Hospital") is None   # 'hospital' alone must not match Gamma Hospital


def test_crowd_levels_and_confidence():
    codes, labels = crowd_level(np.array([0.5, 1.0, 1.7, 3.0]))
    assert list(codes) == [1, 2, 3, 4]
    assert labels[0] == "Seats available" and labels[3] == "Packed"
    assert list(confidence(np.array([2, 10, 100]))) == ["low", "medium", "high"]


def test_expand_tickets_shifts_band_along_route():
    t = pd.DataFrame({
        "ts": pd.to_datetime(["2026-10-05 08:55"]).tz_localize("Asia/Kolkata"),
        "date": ["2026-10-05"], "daytype": ["weekday"], "band": [17], "week": [41],
        "route": ["1UP"], "s_pos": [0], "e_pos": [3], "passenger_count": [2], "is_ac": [True],
    })
    offsets = {"1UP": np.array([0, 300, 900, 1500.0])}   # 0, 5, 15, 25 minutes
    ex = expand_tickets(t, offsets)
    assert len(ex) == 3 and list(ex.seg) == [0, 1, 2]
    assert list(ex.band) == [17, 18, 18]                   # crosses into the 09:00 band after 5 min
    assert ex.pax.sum() == 6
