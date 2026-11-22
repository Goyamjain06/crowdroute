"""End-to-end: raw ticket CSV + GTFS -> data/processed/{tickets_clean.parquet, crowding.parquet, model_info.json}

Run:  python scripts/build_model.py [--etm path/to/October-Delhi.csv] [--gtfs path/to/gtfs_dir]
Takes ~2.5 minutes and ~4 GB RAM on the full October file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crowdroute import config                      # noqa: E402
from crowdroute.gtfs import load_gtfs              # noqa: E402
from crowdroute.etm import load_raw, prepare_tickets  # noqa: E402
from crowdroute.loads import build_crowding        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etm", default=str(config.RAW_ETM))
    ap.add_argument("--gtfs", default=str(config.RAW_GTFS))
    args = ap.parse_args()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    t = time.time()
    g = load_gtfs(Path(args.gtfs))
    print(f"[1/3] GTFS loaded: {len(g.route_seq)} routes, {len(g.stop_name)} stops  ({time.time()-t:.0f}s)")

    raw = load_raw(Path(args.etm))
    clean, report = prepare_tickets(raw, g)
    clean.to_parquet(config.PROCESSED / "tickets_clean.parquet")
    print(f"[2/3] Tickets placed on routes: {report['n_clean']:,} of {report['n_raw']:,} ({report['clean_fraction']:.1%})  ({time.time()-t:.0f}s)")

    cell, info = build_crowding(clean, raw, g)
    cell.to_parquet(config.PROCESSED / "crowding.parquet")
    info["ticket_report"] = report
    json.dump(info, open(config.PROCESSED / "model_info.json", "w"), indent=1, default=str)
    print(f"[3/3] Crowding cells: {len(cell):,}; scale factor S={info['scale']['scale_factor']} "
          f"(implied e-ticket share {info['scale']['implied_sampling_rate']:.1%})  ({time.time()-t:.0f}s)")
    print("level share:", info["level_share"])
    print("confidence share:", info["confidence_share"])


if __name__ == "__main__":
    main()
