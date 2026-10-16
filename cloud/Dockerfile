# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────
# ffmpeg  – required for Phase 2 (FFmpeg subprocess pipe in pipeline.py)
# libglib2.0-0 – required by OpenCV (used internally by scenedetect)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY . .

# Make startup wrapper executable
RUN chmod +x ./wait_for_quadrant.sh

# Expose FastAPI port
EXPOSE 8000

# Run the application (wait_for_quadrant.sh blocks until Qdrant is healthy)
CMD ["./wait_for_quadrant.sh", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
