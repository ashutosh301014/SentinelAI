FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production image contents only: backend, frontend, trained artifacts,
# demo traces, and dependency-graph configuration. Raw HDFS data,
# training scripts, reports, and tests are excluded on purpose.
COPY src ./src
COPY app ./app
COPY models ./models
COPY data/demo ./data/demo
COPY data/config ./data/config

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
