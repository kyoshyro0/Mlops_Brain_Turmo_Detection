# ==========================================
# STAGE 1: Builder (Backend build environment)
# ==========================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS backend-builder

# Install Python 3.12, virtualenv, and build-essential for any compilation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for extremely fast package resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install python dependencies into a virtual environment (/app/.venv)
# PyTorch with CUDA 11.8 will be fetched automatically via uv configured index
RUN uv sync --frozen --no-dev

# Remove pycache and other temporary files to reduce image size
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /app/.venv -type f -name "*.pyc" -delete \
    && find /app/.venv -type f -name "*.pyo" -delete

# ==========================================
# STAGE 2: Backend Runtime (Production GPU backend)
# ==========================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS backend

# Install ONLY runtime Python 3.12 and GUI libraries required by OpenCV / YOLO
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Map Windows absolute paths (registered in host MLflow DB) to Linux paths
RUN mkdir -p /D:/Workspaces && ln -s /app /D:/Workspaces/Mlops_Brain_Turmo

WORKDIR /app

# Copy the virtual environment directly from the builder stage
COPY --from=backend-builder /app/.venv /app/.venv

# Copy application source and configs
COPY src/ ./src/
COPY configs/ ./configs/

# Create model directories
RUN mkdir -p models/train/weights

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Run API directly using the virtualenv python (no uv dependency in final image)
CMD ["python", "src/servering/api.py"]

# ==========================================
# STAGE 3: Frontend (Optimized CPU-only frontend)
# ==========================================
FROM python:3.12-slim AS frontend

# Install minimal system dependencies (no OpenGL/GLib needed for frontend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast minimal package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install ONLY the necessary UI packages (skip PyTorch, YOLO, MLflow, etc.)
# This reduces the frontend image size from ~6GB to ~300MB
RUN uv pip install --no-cache-dir --system \
    streamlit>=1.32.0 \
    requests>=2.32.5 \
    pillow>=11.0.0

# Copy only the frontend app
COPY src/servering/app.py ./src/servering/app.py

# Clean up pycache files
RUN find /usr/local -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local -type f -name "*.pyc" -delete

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "src/servering/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
