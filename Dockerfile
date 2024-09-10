FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /app

# Install OpenJDK 21 and Maven
RUN apt-get update && \
    apt-get install -y openjdk-21-jdk maven g++ gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV HF_HOME=/app/src/beir_cache/huggingface