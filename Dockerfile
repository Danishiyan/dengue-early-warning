# Image for the dengue forecast API (FastAPI + trained model).
FROM python:3.12-slim

# Don't write .pyc files; print logs straight away (no buffering).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Small cloud containers report the host's CPU count, not their real share.
# One thread stops scikit-learn starting many threads that fight over a tiny CPU.
ENV OMP_NUM_THREADS=1 \
    LOKY_MAX_CPU_COUNT=1

WORKDIR /app

# Install runtime packages first, so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user that owns /app (so it can write data/ and models/) and runs the server.
RUN useradd --create-home --uid 1000 appuser \
    && chown appuser:appuser /app

COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser src/ src/
USER appuser

# Build-time pipeline: download data -> build features -> train model.
# reports/ is excluded by .dockerignore, but train.py writes its CV results there.
RUN mkdir -p reports \
    && python -m src.download_data \
    && python -m src.features \
    && python -m src.train

# Cloud platforms (Render, Azure Container Apps) tell the app which port to use through
# the PORT variable; 8000 is the default for running locally.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
