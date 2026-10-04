FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RX_PROVIDER=rules RX_DB_PATH=/data/rx_intake.db

COPY pyproject.toml ./
COPY rx_intake ./rx_intake
COPY data ./data
RUN pip install --no-cache-dir . && mkdir -p /data && useradd --create-home app && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rx_intake.api:app", "--host", "0.0.0.0", "--port", "8000"]
