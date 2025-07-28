# Dockerfile

# Small, CPU-only base
FROM python:3.11-slim

# System deps for scikit-learn & PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch wheel (with add_safe_globals) then all other deps
RUN pip install --no-cache-dir \
      --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
      torch==2.3.1+cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy your app (script + data folders)
COPY challenge1b.py ./challenge1b.py
COPY nltk_data ./nltk_data
# Optional local models (if you have them)
# COPY models ./models

# Reasonable CPU/offline defaults
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Keep PyTorch to one thread inside container
ENV TORCH_NUM_THREADS=1

# Default command
CMD ["python", "challenge1b.py"]
