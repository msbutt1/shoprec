# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt* pyproject.toml* poetry.lock* ./

# Install dependencies
# Try Poetry first if pyproject.toml exists, otherwise use pip
RUN if [ -f pyproject.toml ]; then \
        pip install --no-cache-dir poetry==1.7.1 && \
        poetry config virtualenvs.create false && \
        poetry install --no-dev --no-root || \
        (echo "Poetry install failed, trying pip..." && \
         pip install --no-cache-dir -r requirements.txt); \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy project files
COPY . .

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Run the FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
