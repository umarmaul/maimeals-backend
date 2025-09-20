# Builder
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS builder
WORKDIR /app
COPY pyproject.toml ./
COPY . .
RUN uv build

# Runtime
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# psycopg runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy and install the built wheel as root
COPY --from=builder /app/dist/*.whl /tmp/wheels/
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Create non-root user and drop privileges
RUN useradd -r -s /usr/sbin/nologin -m appuser
USER appuser

EXPOSE 7800
CMD ["maichat-server"]