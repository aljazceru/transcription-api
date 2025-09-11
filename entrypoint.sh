#!/bin/bash
set -e

# Download model if not already cached
# Whisper stores models as .pt files in the root of the cache directory
if [ ! -z "$MODEL_PATH" ]; then
    MODEL_FILE="/app/models/$MODEL_PATH.pt"
    
    # Check if model file exists
    if [ ! -f "$MODEL_FILE" ]; then
        echo "Model $MODEL_PATH not found at $MODEL_FILE, downloading..."
        python -c "
import whisper
import os
# Set all cache paths to use the shared volume
os.environ['TORCH_HOME'] = '/app/models'
os.environ['HF_HOME'] = '/app/models'
os.environ['TRANSFORMERS_CACHE'] = '/app/models'
os.environ['XDG_CACHE_HOME'] = '/app/models'
model_name = '$MODEL_PATH'
print(f'Downloading model {model_name}...')
model = whisper.load_model(model_name, download_root='/app/models')
print(f'Model {model_name} downloaded and cached successfully')
"
    else
        echo "Model $MODEL_PATH already cached at $MODEL_FILE"
        # Just verify it loads properly
        python -c "
import whisper
import os
os.environ['TORCH_HOME'] = '/app/models'
os.environ['XDG_CACHE_HOME'] = '/app/models'
model = whisper.load_model('$MODEL_PATH', download_root='/app/models')
print(f'Model $MODEL_PATH loaded successfully from cache')
"
    fi
fi

# Generate gRPC code if not already generated
if [ ! -f "/app/src/transcription_pb2.py" ]; then
    echo "Generating gRPC code from proto files..."
    python -m grpc_tools.protoc \
        -I/app/proto \
        --python_out=/app/src \
        --grpc_python_out=/app/src \
        /app/proto/transcription.proto
    
    # Fix imports in generated files (keep absolute import)
    # No need to modify - the generated import should work as-is
fi

# Start the transcription server
echo "Starting Transcription API Server..."
echo "gRPC Port: $GRPC_PORT"
echo "WebSocket Port: $WEBSOCKET_PORT (Enabled: $ENABLE_WEBSOCKET)"
echo "Model: $MODEL_PATH"

cd /app/src
exec python transcription_server.py