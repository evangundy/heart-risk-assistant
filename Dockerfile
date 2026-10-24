# Heart Risk Assistant: Streamlit app serving the best MLflow model.
# Build AFTER training so models/best_model.joblib exists:
#   python -m src.train && python -m src.compare_runs
#   docker build -t heart-risk-assistant .
#   docker run -p 8501:8501 --env-file .env heart-risk-assistant
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY models/ models/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
