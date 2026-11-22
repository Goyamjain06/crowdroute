.PHONY: setup data model serve validate test
setup:        ## install python deps
	pip install -r requirements.txt
data:         ## unpack the raw data you downloaded into data/raw/
	mkdir -p data/raw/gtfs data/raw/etm
	unzip -o -q data/raw/delhi_buses_static_gtfs_v1.zip -d data/raw/gtfs
	unzip -o -q data/raw/October-Delhi.csv.zip -d data/raw/etm
model:        ## tickets + GTFS -> crowding model (~2.5 min)
	python scripts/build_model.py
serve:        ## API + web UI on http://localhost:8000
	uvicorn crowdroute.api:app --port 8000
validate:     ## pattern checks + hold-out -> docs/validation.md
	python scripts/validate.py
test:
	python -m pytest -q tests
