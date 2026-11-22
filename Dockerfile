FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Expects data/raw/gtfs/*.txt and data/raw/etm/October-Delhi.csv to be mounted or copied in.
# Build the model once (python scripts/build_model.py), then serve.
EXPOSE 8000
CMD ["uvicorn", "crowdroute.api:app", "--host", "0.0.0.0", "--port", "8000"]
