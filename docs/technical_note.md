# CrowdRoute — technical note

*Transport Stack Hackathon 2026, Problem 2: crowdedness-aware journey planner. Phase 1.*

## 1. What we built

A reproducible pipeline that converts raw e-ticket records into an estimate of on-board load for every Delhi bus route, on every stretch between consecutive stops, for every 30-minute band of a weekday, Saturday or Sunday/holiday; a journey planner on static GTFS that shows this crowding beside each option; and an operator view of the same model. Everything runs from two inputs: `October-Delhi.csv` (2,559,899 tickets, 1–31 Oct 2026) and the Delhi static GTFS feed (2,574 route-directions, 6,862 stops, 80,357 scheduled trips).

## 2. Data preparation

**Route resolution.** Ticket route strings are GTFS `route_short_name`s with a direction suffix (`8UP`, `717ADOWN`). 98.7 % match directly; variants such as `522BDOWN2`, `567BSTLUP` and `ML96UP` are reduced to their base route. 33 k tickets (1.3 %) on routes absent from GTFS are dropped.

**Stop matching.** Ticket stop names are free text and only 57 % match GTFS names exactly ("Aiims" vs "AIIMS", "Pul Prehladpur ( Badarpur)" vs "Pul Pehladpur ( BadarPur)"). We normalise names (case, punctuation, Xing/X-ing/Crossing, Stn/Station, collapsed initials) and then fuzzy-match **only against the stops of the ticket's own route** (rapidfuzz `token_set_ratio ≥ 80`), with a guard that at least half of the *distinctive* tokens (not "hospital", "metro", "nagar"…) must appear in the candidate. This stops "S.J. Hospital" matching "T.B. Hospital". Result: 43 k distinct (route, name) pairs matched exactly, 17 k fuzzily, 5.7 k unmatched.

**Direction.** If start comes after end on the route, we try the opposite direction (`UP`↔`DOWN`); 2.6 k tickets are fixed this way, 6.7 k remain inconsistent and are dropped.

**Outcome.** 2,149,131 tickets (84.0 %) are placed with a valid start < end position. The remaining 16 % are mostly tickets whose stop name is not on the GTFS version of the route (route changes since the feed) — they are dropped rather than guessed.

**Day types.** Weekday (20 days), Saturday (5), Sunday (4) + the two public holidays, 2 Oct and 20 Oct, which carry 45 % and 55 % of weekday volume and are folded into Sunday (6 days).

## 3. The crowding model

Let a ticket board at position *s* and alight at *e* on route *r* at time *t*, carrying *p* passengers (`passenger_count`, 1–3).

1. **Expansion.** The ticket contributes *p* riders to every stretch *k ∈ [s, e)*. The time at which the rider is on stretch *k* is *t + (τ<sub>k</sub> − τ<sub>s</sub>)*, where τ is the median scheduled run time from the route's first stop (GTFS `stop_times`, interpolated where a trip skips stops). A 40-minute ride is therefore counted in the bands it actually spans. 2.15 M tickets expand to 27.3 M (ticket, stretch) rows.

2. **Aggregation.** Riders are summed per cell *c = (route, stretch, day-type, band)* and divided by the number of days of that type → *sampled on-board riders per day in that band*, `pax_per_day`. The raw ticket count `n` is kept as the cell's evidence.

3. **Buses per band.** `trips = max(GTFS scheduled trips crossing the stretch in that band, N̂)`, where N̂ is an occupancy estimate from the tickets themselves: within a cell-day, *k* tickets carry a vehicle registration and *v* distinct vehicles appear; if *N* buses were running, E[v] = N(1 − (1 − 1/N)<sup>k</sup>), solved for *N* and averaged over days. This matters because the GTFS feed is thin on some routes (median 18 trips/day/direction) while the tickets prove more buses are running; 28 % of cells use N̂.

4. **Scaling.** `load_sampled = pax_per_day / trips` is the per-bus load *among e-ticket holders only*. E-tickets exclude ETM paper tickets, passes (including the women's pink-ticket scheme) and concessions, so the sampled load is multiplied by a scale factor **S**. We compute S two ways:
   - *Capacity calibration (used):* choose S so that the 95th percentile of `load_sampled` over well-supported weekday cells (`n ≥ 8`) equals crush capacity. Buses do fill up at peak; a full bus is the physical ceiling for a typical busy cell. **S = 48.1**, implied e-ticket share 2.1 %.
   - *Ridership calibration (cross-check):* S = 4.1 M weekday passenger trips (DTC + cluster, published order of magnitude) / 102,303 ticketed passengers per weekday → **S = 40.1**, 2.5 %.
   - The hackathon brief for Problem 3 states e-ticketing is ≈3–4 % of bus tickets. Three independent numbers agree to within a factor of ~1.5, which is the evidence that treating the sample as roughly representative is reasonable.

5. **Shrinkage.** A cell with few tickets is pulled towards the route's own profile: `load = (n·load_raw + K·prior) / (n + K)` with K = 10 and prior = ticket-weighted mean load of that route, day-type and band across all its stretches. This removes single-ticket spikes without flattening real hotspots.

6. **Capacity and levels.** Seats are 35 (AC, a proxy for 12 m low-floor) or 32 (non-AC, a blend of 12 m and 9 m e-buses), chosen by the majority `is_ac` of the cell's tickets; crush capacity 70 / 60. `load_factor = load / seats`. Levels: **Seats available** ≤ 0.85 · **Standing room** ≤ 1.4 · **Crowded** ≤ 2.0 · **Packed** above. Loads are clipped at 2.5 × crush (0.05 % of cells hit the clip).

7. **Confidence.** `low` < 8 tickets, `medium` < 30, `high` ≥ 30. Of 2.88 M cells, 7 % are high, 20 % medium, 73 % low — the low ones are overwhelmingly off-peak bands on minor routes. The planner always reports the confidence of the stretches a journey actually rides, and falls back to the route profile (flagged low) when a cell is missing.

## 4. The planner

Deliberately simple, as the brief suggests. Origin and destination are stops (searchable by name) or coordinates; stops within 600 m are candidates. We enumerate direct journeys and one-transfer journeys (shared stop between two routes), take the next scheduled GTFS trip after the requested time for each leg, and drop options with more than 60 minutes' wait. Each leg is annotated with the level of every stretch it rides *in the band the bus reaches that stretch*, the worst stretch, the mean load factor and a confidence. The journey's level is its worst leg's worst stretch — that is what the commuter will experience. We also re-plan at +15/+30/+45 minutes and surface departures that are calmer than the best immediate option ("Leave a little later?"). Planning takes ~10 ms.

## 5. Checks against known patterns

Full report with figures: [validation.md](validation.md).

- **Peaks.** Weekday network load is 2.3× midday at 08–10 h and 1.6× at 18–20 h; 49 % of well-supported stretches are crowded/packed at 08–10 h versus 10 % at 13–15 h. Sunday is flat; Saturday sits between.
- **Weekends and holidays.** Saturday carries 79 % of weekday volume, Sunday 49 %; Gandhi Jayanti and Dussehra 45–55 %.
- **Tidal flow.** For 384 UP/DOWN pairs with ≥ 500 tickets each way, 64 % peak at opposite ends of the day (e.g. 227A: UP AM/PM = 0.21, DOWN = 8.1).
- **Busy corridors.** The top routes by riders (717A, 511A, 729, 623, 764, 522) are Delhi's known trunk corridors.
- **Hold-out.** A model built on 1–21 Oct versus one built on 22–31 Oct, compared over 232 k weekday cells that are well-supported in both: Spearman 0.76, exact level match 66 %, within one level 94 %. The ranking of stretches transfers; the absolute level at a boundary (e.g. 0.85) is where the two halves disagree.

## 6. Where the estimates are least reliable

1. **Routes with atypical e-ticket adoption.** S is global. A route where conductors push QR tickets harder than average will read as more crowded than it is; a handful of Rohini feeder routes (114C, 112) hit the ceiling at 09:00. The fix is per-route calibration against ETM paper-ticket counts or AVL passenger counts, which we do not have in Phase 1. We flag clipped cells.
2. **Low-confidence cells** (73 % of cells, mostly off-peak on minor routes). The level shown there is largely the route profile; the UI says so.
3. **Intended vs actual alighting.** Tickets give the planned destination. Early alighting inflates the tail of long tickets; late alighting (riding past a paid stage) is invisible. Net effect unknown; likely a few per cent.
4. **Pass holders and paper tickets** have a different spatial pattern (passes skew to long daily commutes) that a single scale factor cannot capture.
5. **Bus size.** `is_ac` is a coarse proxy. A 9 m e-bus with 60 riders is packed; a 12 m with 60 is merely crowded.
6. **Schedule drift.** GTFS run times are used for the time shift and the planner; real buses are slower at peak, so the band assignment is slightly early on long routes.
7. **Thin schedule in GTFS.** 22 % of cells have tickets in bands where GTFS lists no trip; frequency there comes only from the vehicle-number estimate.

## 7. Reproducibility

`python scripts/build_model.py` rebuilds everything from the two raw inputs in ~2.5 minutes (4 GB RAM). All assumptions are named constants in `crowdroute/config.py` with the reasoning next to them; changing `SCALE_METHOD`, `CAL_QUANTILE`, capacities, level thresholds or shrinkage strength and rebuilding is the intended way to probe the model. `scripts/validate.py` regenerates every number in this note. Unit tests (`pytest tests`) cover name normalisation, matching guards, level mapping and the time-shifted expansion.

## 8. What we would do in Phase 2

Per-route calibration using the GTFS-RT vehicle positions (Problem 1 data) to count actual trips per band and to replace scheduled run times with observed ones; a fleet roster for capacity; an explicit pass-holder layer; and crowding-aware route choice inside the planner (penalise crowded stretches in the ranking rather than only displaying them).
