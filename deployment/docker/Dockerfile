# deployment/docker/Dockerfile
# Phase 6: Production Docker Configuration
# Multi-stage build for optimized production images

# Stage 1: Base image with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r scie && useradd -r -g scie scie

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Set ownership
RUN chown -R scie:scie /app

# Switch to non-root user
USER scie

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Development command
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 3: Testing image
FROM development as testing

# Install additional testing dependencies
RUN pip install --no-cache-dir \
    pytest-asyncio \
    pytest-mock \
    pytest-xdist \
    coverage

# Copy test files
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Set environment for testing
ENV TESTING=1 \
    PYTHONPATH=/app

# Test command
CMD ["pytest", "tests/", "-v", "--cov=PY_Files", "--cov-report=html"]

# Stage 4: Production image
FROM base as production

# Install production dependencies only
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn

# Copy production configuration
COPY configs/production.yaml ./configs/
COPY deployment/docker/gunicorn.conf.py ./gunicorn.conf.py

# Copy source code
COPY . .

# Remove development and test files
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -delete && \
    find . -name "tests" -type d -exec rm -rf {} + && \
    find . -name "scripts" -type d -exec rm -rf {} + && \
    find . -name "*.md" -delete && \
    find . -name ".git" -type d -exec rm -rf {} +

# Set ownership
RUN chown -R scie:scie /app

# Switch to non-root user
USER scie

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Production command
CMD ["gunicorn", "--config", "gunicorn.conf.py", "main:app"]

# Stage 5: Security scanning
FROM production as security-scan

# Install security scanning tools
RUN pip install --no-cache-dir \
    bandit \
    safety

# Run security scans
RUN bandit -r PY_Files/ -f json -o bandit-report.json && \
    safety check --json --output safety-report.json

# Stage 6: Final production image
FROM production as final

# Copy security scan results if needed
# COPY --from=security-scan /app/bandit-report.json ./reports/
# COPY --from=security-scan /app/safety-report.json ./reports/

# Set production environment
ENV ENVIRONMENT=production \
    PYTHONPATH=/app

# Final production command
CMD ["gunicorn", "--config", "gunicorn.conf.py", "main:app"]
