FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/config && \
    curl -L -o /app/config/pubtables1m_detection_detr_r18.pth "https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth?download=true" & \
    curl -L -o /app/config/pubtables1m_structure_detr_r18.pth "https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_structure_detr_r18.pth?download=true" & \
    wait

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]