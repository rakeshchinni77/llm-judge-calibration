# ---------------------------------------------------------------------------
# Dockerfile — LLM-as-a-Judge Calibration Pipeline
# ---------------------------------------------------------------------------
# Pinned to Python 3.11-slim for reproducibility.
# Build context: repository root.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System deps (needed for numpy/scikit-learn native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt ./

# Install Python dependencies (no cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository into the container
COPY . .

# Default command: run the frozen test evaluation
CMD ["python", "evaluate.py", "--split", "test"]
