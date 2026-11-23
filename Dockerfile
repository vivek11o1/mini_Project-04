FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py helper.py ./
COPY src ./src
# The repo does not commit models/*.pkl; helper.py rebuilds them from this file.
COPY data/processed/rfm_segmented.csv ./data/processed/rfm_segmented.csv

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
