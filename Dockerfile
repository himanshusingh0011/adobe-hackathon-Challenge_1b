# Small, CPU-only base
FROM python:3.11-slim

# System deps for scikit-learn & PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install Python deps (CPU only). Internet is used at build-time; runtime will be offline.
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app (script + data folders)
# (We will often mount the host repo in at runtime, but copying lets the image run standalone too.)
COPY challenge1b.py ./challenge1b.py
COPY nltk_data ./nltk_data
# Optional local models (if you have them)
# COPY models ./models

# Reasonable CPU/offline defaults (your script also sets these)
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Keep PyTorch to one thread inside container
ENV TORCH_NUM_THREADS=1

# Default command scans /app for *Collection */challenge1b_input.json and writes outputs there
CMD ["python", "challenge1b.py"]
