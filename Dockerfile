FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY wildlife_tracker ./wildlife_tracker
ENV PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN mkdir -p /tmp/ultralytics
CMD ["python", "-m", "wildlife_tracker", "run"]
