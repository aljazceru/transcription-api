# Use NVIDIA CUDA base image with PyTorch pre-installed for faster builds
# This image includes CUDA, cuDNN, and PyTorch - saving significant build time
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support first (if not using pre-built image)
# This is much faster than installing via requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install remaining dependencies
COPY requirements.txt .
# Remove torch from requirements since we already installed it
RUN grep -v "^torch" requirements.txt > requirements_no_torch.txt && \
    pip install -r requirements_no_torch.txt

# Install grpcio-tools for protobuf generation
RUN pip install grpcio-tools==1.60.0

# Copy proto files and generate gRPC code
COPY proto/ ./proto/
RUN mkdir -p ./src && \
    python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./src \
    --grpc_python_out=./src \
    ./proto/transcription.proto
# Note: Don't modify imports - they work as-is when sys.path is set correctly

# Copy application code
COPY src/ ./src/
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh

# Environment variables
ENV MODEL_PATH=large-v3 \
    GRPC_PORT=50051 \
    WEBSOCKET_PORT=8765 \
    ENABLE_WEBSOCKET=true \
    CACHE_DIR=/app/models \
    TORCH_HOME=/app/models \
    HF_HOME=/app/models

# Create model cache directory
RUN mkdir -p /app/models

# Volume for model cache
VOLUME ["/app/models"]

# Expose ports
EXPOSE 50051 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.channel_ready()" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]