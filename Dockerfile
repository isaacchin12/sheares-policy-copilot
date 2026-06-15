FROM python:3.11-slim

WORKDIR /app

# Install system deps for unstructured + presidio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    python -m spacy download en_core_web_sm

COPY . .

# Pre-build the index from the committed sample corpus
RUN python ingestion/build_index.py || echo "Index build skipped (no Qdrant at build time)"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
