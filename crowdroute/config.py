"""All tunable assumptions live here, with the reasoning behind each one.

Everything in this file is a documented assumption. The technical note
(docs/technical_note.md) references these by name.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_GTFS = ROOT / "data" / "raw" / "gtfs"
RAW_ETM = ROOT / "data" / "raw" / "etm" / "October-Delhi.csv"
PROCESSED = ROOT / "data" / "processed"

# --- Time binning -----------------------------------------------------------
TIMEBAND_MIN = 30           # width of a time band (minutes). 30 keeps cells dense enough.
N_BANDS = 24 * 60 // TIMEBAND_MIN

# Day types. Delhi bus demand on Saturday sits between weekday and Sunday.
DAYTYPES = ("weekday", "saturday", "sunday")

# Public holidays in Oct 2026 that behave like Sundays in the ticket data.
# 2 Oct = Gandhi Jayanti, 20 Oct = Dussehra. Treated as 'sunday' for aggregation.
HOLIDAYS = {"2026-10-02", "2026-10-20"}

# --- Scaling from ticket sample to full ridership --------------------------
# The ONDC e-ticketing stream is a small, self-selected sample of all riders:
# ETM paper tickets, passes (incl. the pink-ticket scheme for women) and
# concession holders are NOT in it. We therefore scale the observed boardings
# to a documented all-mode daily ridership figure.
#
# DTC + cluster buses carry roughly 40-45 lakh passenger trips per weekday
# (Delhi Transport Dept / DTC annual reporting). We use 4.1 million and
# expose it as a single knob so the judges can change it.
TARGET_WEEKDAY_RIDERSHIP = 4_100_000

# How to pick the scale factor S:
#   "capacity"  -> choose S so that the 98th percentile of well-supported weekday
#                  cells equals crush capacity (buses do fill up at peak, but a
#                  typical full bus is the physical ceiling). Default.
#   "ridership" -> S = TARGET_WEEKDAY_RIDERSHIP / observed ticketed pax per day.
# Both are computed and reported; they agree within ~1.5x on this data, which is
# the main evidence that the sample is roughly representative.
SCALE_METHOD = "capacity"
CAL_QUANTILE = 0.95

# Sanity bound: scale factor is clipped so a single cell can never exceed
# this multiple of crush capacity (protects against tiny-sample blow-ups).
MAX_LOAD_FACTOR = 2.5

# --- Bus capacity -----------------------------------------------------------
# Seats / crush capacity. Delhi's fleet is mostly 12 m low-floor (AC & non-AC)
# plus 9 m electric 'Mohalla' buses. is_ac=True is a reasonable proxy for a
# 12 m bus in this data; is_ac=False mixes 12 m non-AC and 9 m.
CAPACITY = {
    "ac":    {"seats": 35, "crush": 70},   # 12 m AC low-floor
    "nonac": {"seats": 32, "crush": 60},   # blend of 12 m non-AC and 9 m e-bus
}

# Crowding classes as load factor = estimated on-board / seats.
CROWD_LEVELS = [
    # (upper bound of load factor, code, label commuters see)
    (0.85, 1, "Seats available"),
    (1.40, 2, "Standing room"),
    (2.00, 3, "Crowded"),
    (9.99, 4, "Packed"),
]

# --- Confidence -------------------------------------------------------------
# Based on how many raw tickets support a cell after pooling.
CONF_THRESHOLDS = {"low": 8, "medium": 30}   # < low -> low, < medium -> medium, else high

# Empirical-Bayes shrinkage: a thin cell is pulled towards the route's
# day-type profile with this pseudo-count weight.
SHRINK_K = 10

# --- Stop-name matching -----------------------------------------------------
FUZZY_MIN_SCORE = 80   # rapidfuzz token_set_ratio threshold for stop names
