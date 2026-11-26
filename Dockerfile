# Image for the dengue forecast API (FastAPI + trained model).
FROM python:3.12-slim

# Don't write .pyc files; print logs straight away (no buffering).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
