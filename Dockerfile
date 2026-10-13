FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (imajı küçük tutar); GPU için bu satırı değiştirin
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[embed,demo]"

COPY config ./config
COPY scripts ./scripts
COPY demo ./demo

EXPOSE 8000 8501
CMD ["uvicorn", "anatomy_ks.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
