FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt 2>/dev/null || pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY sql/ ./sql/

ENV PYTHONPATH=/app

EXPOSE 8010

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
