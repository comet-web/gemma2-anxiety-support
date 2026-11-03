FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the NeMo Guardrails embedding model (intent matching) into the image.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY guardrails ./guardrails
ENV RAILS_CONFIG_PATH=/app/guardrails

RUN useradd --create-home appuser && chown -R appuser /app /opt/fastembed
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s CMD curl -fsS http://localhost:8080/healthz || exit 1
CMD ["uvicorn", "anxiety_support.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
