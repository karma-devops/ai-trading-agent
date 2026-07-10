# ============================================================================
# AI Trading Agent - HyperLiquid (Private)
# Dockerfile for EasyPanel / zip upload / standalone Docker builds
# ============================================================================
#
# HOW TO USE THIS FILE
# ---------------------
# Option A - EasyPanel zip upload:
#   1. Download the repo as a ZIP.
#   2. In EasyPanel, create a new project from "Upload" or "Dockerfile".
#   3. Upload the ZIP.
#   4. Set the environment variables listed in .env_example.
#   5. Expose port 5000.
#
# Option B - Manual docker build:
#   docker build -t ai-trading-agent-hl .
#   docker run -p 5000:5000 --env-file .env ai-trading-agent-hl
#
# Option C - docker-compose:
#   docker compose up -d
# ============================================================================

FROM python:3.12-slim

# Install system dependencies needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    llvm \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Do not write .pyc files; flush stdout/stderr immediately for logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies first so Docker layer caching works
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure directories expected by the code exist at runtime
RUN mkdir -p /app/src/data /app/temp_data /app/logs

# The dashboard listens on this port by default
EXPOSE 5000

# Use Gunicorn for production (configured in start.sh)
CMD ["./start.sh"]
