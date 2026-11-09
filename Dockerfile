FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY courtvision ./courtvision
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY tools ./tools
COPY evaluation ./evaluation

# Pre-download the default weights so the container runs offline
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt'); YOLO('yolov8s.pt')"

ENTRYPOINT ["courtvision"]
CMD ["--help"]
