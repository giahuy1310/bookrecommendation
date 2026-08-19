FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Books.csv / data.zip are expected at /data when mounted in compose.
ENV PYTHONPATH=/app
ENV BOOKS_CSV_PATH=/data/Books.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
