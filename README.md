# Transcription API Service

A high-performance, standalone transcription service with **REST API**, **gRPC**, and **WebSocket** support, optimized for real-time speech-to-text applications. Perfect for desktop applications, web services, and IoT devices.

## Features

- **Multiple API Interfaces**: REST API, gRPC, and WebSocket
- **High Performance**: Optimized with TF32, cuDNN, and efficient batching
- **Whisper Models**: Support for all Whisper models (tiny to large-v3)
- **Real-time Streaming**: Bidirectional streaming for live transcription
- **Voice Activity Detection**: Smart VAD to filter silence and noise
- **Anti-hallucination**: Advanced filtering to reduce Whisper hallucinations
- **Docker Ready**: Easy deployment with GPU support
- **Interactive Docs**: Auto-generated API documentation (Swagger/OpenAPI)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
cd transcription-api

# Start the service (uses 'base' model by default)
docker compose up -d

# Check logs
docker compose logs -f

# Stop the service
docker compose down
```

### Configuration

Edit `.env` or `docker-compose.yml` to configure:

```env
# Model Configuration
MODEL_PATH=base          # tiny, base, small, medium, large, large-v3

# Service Ports
GRPC_PORT=50051         # gRPC service port
WEBSOCKET_PORT=8765     # WebSocket service port
REST_PORT=8000          # REST API port

# Feature Flags
ENABLE_WEBSOCKET=true   # Enable WebSocket support
ENABLE_REST=true        # Enable REST API

# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # GPU device ID (if available)
```

## API Endpoints

The service provides three ways to access transcription:

### 1. REST API (Port 8000)

The REST API is perfect for simple HTTP-based integrations.

#### Base URLs
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health

#### Key Endpoints

**Transcribe File**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "language=en" \
  -F "task=transcribe" \
  -F "vad_enabled=true"
```

**Health Check**
```bash
curl http://localhost:8000/health
```

**Get Capabilities**
```bash
curl http://localhost:8000/capabilities
```

**WebSocket Streaming** (via REST API)
```bash
# Connect to WebSocket
ws://localhost:8000/ws/transcribe
```

For detailed API documentation, visit http://localhost:8000/docs after starting the service.

### 2. gRPC (Port 50051)

For high-performance, low-latency applications. See protobuf definitions in `proto/transcription.proto`.

### 3. WebSocket (Port 8765)

Legacy WebSocket endpoint for backward compatibility.


## Usage Examples

### REST API (Python)

```python
import requests

# Transcribe a file
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/transcribe',
        files={'file': f},
        data={
            'language': 'en',
            'task': 'transcribe',
            'vad_enabled': True
        }
    )
    result = response.json()
    print(result['full_text'])
```

### REST API (cURL)

```bash
# Transcribe an audio file
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "language=en"

# Health check
curl http://localhost:8000/health

# Get service capabilities
curl http://localhost:8000/capabilities
```

### WebSocket (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/transcribe');

ws.onopen = () => {
  console.log('Connected');

  // Send audio data (base64-encoded PCM16)
  ws.send(JSON.stringify({
    type: 'audio',
    data: base64AudioData,
    language: 'en',
    vad_enabled: true
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'transcription') {
    console.log('Transcription:', data.text);
  }
};

// Stop transcription
ws.send(JSON.stringify({ type: 'stop' }));
```

## Rust Client Usage

### Build and Run Examples

```bash
cd examples/rust-client

# Build
cargo build --release

# Run live transcription from microphone
cargo run --bin live-transcribe

# Transcribe a file
cargo run --bin file-transcribe -- audio.wav

# Stream a WAV file
cargo run --bin stream-transcribe -- audio.wav --realtime
```

## Performance Optimizations

This service includes several performance optimizations:

1. **Shared Model Instance**: Single model loaded in memory, shared across all connections
2. **TF32 & cuDNN**: Enabled for Ampere GPUs for faster inference
3. **No Gradient Computation**: `torch.no_grad()` context for inference
4. **Optimized Threading**: Dynamic thread pool sizing based on CPU cores
5. **Efficient VAD**: Fast voice activity detection to skip silent audio
6. **Batch Processing**: Processes audio in optimal chunk sizes
7. **gRPC Optimizations**: Keepalive and HTTP/2 settings tuned for performance

## Supported Formats

- **Audio**: WAV, MP3, WebM, OGG, FLAC, M4A, raw PCM16
- **Sample Rate**: 16kHz (automatically resampled)
- **Languages**: Auto-detect or specify (en, es, fr, de, it, pt, ru, zh, ja, ko, etc.)
- **Tasks**: Transcribe or Translate to English

## API Documentation

Full interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Health Monitoring

```bash
# Check service health
curl http://localhost:8000/health

# Response:
{
  "healthy": true,
  "status": "running",
  "model_loaded": "large-v3",
  "uptime_seconds": 3600,
  "active_sessions": 2
}
```
