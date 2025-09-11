#!/bin/bash
# Build script with options for different configurations

set -e

# Default values
DOCKERFILE="Dockerfile"
USE_CACHE=true
PLATFORM="linux/amd64"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pytorch)
            DOCKERFILE="Dockerfile.pytorch"
            echo "Using PyTorch base image (faster build)"
            shift
            ;;
        --cuda)
            DOCKERFILE="Dockerfile"
            echo "Using NVIDIA CUDA base image"
            shift
            ;;
        --no-cache)
            USE_CACHE=false
            echo "Building without cache"
            shift
            ;;
        --platform)
            PLATFORM="$2"
            echo "Building for platform: $PLATFORM"
            shift 2
            ;;
        --help)
            echo "Usage: ./build.sh [options]"
            echo "Options:"
            echo "  --pytorch    Use PyTorch base image (fastest)"
            echo "  --cuda       Use NVIDIA CUDA base image (default)"
            echo "  --no-cache   Build without using cache"
            echo "  --platform   Target platform (default: linux/amd64)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
BUILD_CMD="docker build"

if [ "$USE_CACHE" = false ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

BUILD_CMD="$BUILD_CMD --platform $PLATFORM -f $DOCKERFILE -t transcription-api:latest ."

echo "Building transcription-api..."
echo "Command: $BUILD_CMD"

# Execute build
eval $BUILD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "To run the service:"
    echo "  docker compose up -d"
    echo ""
    echo "Or with GPU support:"
    echo "  docker compose --profile gpu up -d"
else
    echo "Build failed!"
    exit 1
fi