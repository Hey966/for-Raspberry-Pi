FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
ENV OMP_NUM_THREADS=1
CMD ["python", "app.py"]
