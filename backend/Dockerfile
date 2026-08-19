FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Mount host data at /data (e.g. data.zip and/or Books.csv).
# Extraction looks for /data/data.zip → /data/Books.csv (not /data.zip at /).
ENV PYTHONPATH=/app
ENV DATA_DIR=/data
ENV BOOKS_CSV_PATH=/data/Books.csv
ENV DATA_ZIP_PATH=/data/data.zip

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
