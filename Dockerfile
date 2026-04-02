FROM python:3.12-slim

WORKDIR /app

# System deps: gcc for psycopg2-binary build wheels, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.5

# Copy dependency manifest (no lockfile in repo — poetry resolves at build time)
COPY pyproject.toml ./

# Install main deps only (no dev groups, no optional extras)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Copy source
COPY src/ ./src/

# Data directory for SQLite caches and audit log
RUN mkdir -p data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
