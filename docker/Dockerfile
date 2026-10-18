FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy requirements from project root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy source code directory
COPY src/ ./src/

# 3. Copy API endpoint script to app root
COPY src/apis/predict_api.py ./app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]