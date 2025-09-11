# Transcription API Service

A high-performance, standalone transcription service with gRPC and WebSocket support, optimized for real-time speech-to-text applications. Perfect for desktop applications, web services, and IoT devices.

## Features

- **Dual Protocol Support**: Both gRPC (recommended) and WebSocket
- **Real-Time Streaming**: Bidirectional audio streaming with immediate transcription
- **Multiple Models**: Support for all Whisper models (tiny to large-v3)
- **Language Support**: 50+ languages with automatic detection
- **Docker Ready**: Simple deployment with Docker Compose
- **Production Ready**: Health checks, monitoring, and graceful shutdown
- **Rust Client Examples**: Ready-to-use Rust client for desktop applications

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
MODEL_PATH=base          # tiny, base, small, medium, large, large-v3
GRPC_PORT=50051         # gRPC service port
WEBSOCKET_PORT=8765     # WebSocket service port
ENABLE_WEBSOCKET=true   # Enable WebSocket support
CUDA_VISIBLE_DEVICES=0  # GPU device ID (if available)
```

## API Protocols

### gRPC (Recommended for Desktop Apps)

**Why gRPC?**
- Strongly typed with Protocol Buffers
- Excellent performance with HTTP/2
- Built-in streaming support
- Auto-generated client code
- Better error handling

**Proto Definition**: See `proto/transcription.proto`

**Service Methods**:
- `StreamTranscribe`: Bidirectional streaming for real-time transcription
- `TranscribeFile`: Single file transcription
- `GetCapabilities`: Query available models and languages
- `HealthCheck`: Service health status

### WebSocket (Alternative)

**Protocol**:
```javascript
// Connect
ws://localhost:8765

// Send audio
{
  "type": "audio",
  "data": "base64_encoded_pcm16_audio"
}

// Receive transcription
{
  "type": "transcription",
  "text": "Hello world",
  "start_time": 0.0,
  "end_time": 1.5,
  "is_final": true,
  "timestamp": 1234567890
}

// Stop
{
  "type": "stop"
}
```

## Rust Client Usage

### Installation

```toml
# Add to your Cargo.toml
[dependencies]
tonic = "0.10"
tokio = { version = "1.35", features = ["full"] }
# ... see examples/rust-client/Cargo.toml for full list
```

### Live Microphone Transcription

```rust
use transcription_client::TranscriptionClient;

#[tokio::main]
async fn main() -> Result<()> {
    // Connect to service
    let mut client = TranscriptionClient::connect("http://localhost:50051").await?;
    
    // Start streaming from microphone
    let stream = client.stream_from_microphone(
        "auto",       // language
        "transcribe", // task
        "base"        // model
    ).await?;
    
    // Process transcriptions
    while let Some(transcription) = stream.next().await {
        println!("{}", transcription.text);
    }
    
    Ok(())
}
```

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

## Audio Requirements

- **Format**: PCM16 (16-bit signed integer)
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Chunk Size**: Minimum ~500 bytes (flexible for real-time)

## Performance Optimization

### For Real-Time Applications

1. **Use gRPC**: Lower latency than WebSocket
2. **Small Chunks**: Send audio in 0.5-1 second chunks
3. **Model Selection**:
   - `tiny`: Fastest, lowest accuracy (real-time on CPU)
   - `base`: Good balance (near real-time on CPU)
   - `small`: Better accuracy (may lag on CPU)
   - `large-v3`: Best accuracy (requires GPU for real-time)

### GPU Acceleration

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Architecture

```
┌─────────────┐
│ Rust App    │
│ (Desktop)   │
└──────┬──────┘
       │ gRPC/HTTP2
       ▼
┌─────────────┐
│ Transcription│
│   Service    │
│  ┌────────┐ │
│  │Whisper │ │
│  │ Model  │ │
│  └────────┘ │
└─────────────┘
```

### Components

1. **gRPC Server**: Handles streaming audio and returns transcriptions
2. **WebSocket Server**: Alternative protocol for web clients
3. **Transcription Engine**: Whisper/SimulStreaming for speech-to-text
4. **Session Manager**: Handles multiple concurrent streams
5. **Model Cache**: Prevents re-downloading models

## Advanced Configuration

### Using SimulStreaming

For even lower latency, mount SimulStreaming:

```yaml
volumes:
  - ./SimulStreaming:/app/SimulStreaming
environment:
  - SIMULSTREAMING_PATH=/app/SimulStreaming
```

### Custom Models

Mount your own Whisper models:

```yaml
volumes:
  - ./models:/app/models
environment:
  - MODEL_PATH=/app/models/custom-model.pt
```

### Monitoring

The service exposes metrics on `/metrics` (when enabled):

```bash
curl http://localhost:9090/metrics
```

## API Reference

### gRPC Methods

#### StreamTranscribe
```protobuf
rpc StreamTranscribe(stream AudioChunk) returns (stream TranscriptionResult);
```

Bidirectional streaming for real-time transcription. Send audio chunks, receive transcriptions.

#### TranscribeFile
```protobuf
rpc TranscribeFile(AudioFile) returns (TranscriptionResponse);
```

Transcribe a complete audio file in one request.

#### GetCapabilities
```protobuf
rpc GetCapabilities(Empty) returns (Capabilities);
```

Query available models, languages, and features.

#### HealthCheck
```protobuf
rpc HealthCheck(Empty) returns (HealthStatus);
```

Check service health and status.

## Language Support

Supports 50+ languages including:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- And many more...

Use `"auto"` for automatic language detection.

## Troubleshooting

### Service won't start
- Check if ports 50051 and 8765 are available
- Ensure Docker has enough memory (minimum 4GB)
- Check logs: `docker compose logs transcription-api`

### Slow transcription
- Use a smaller model (tiny or base)
- Enable GPU if available
- Reduce audio quality to 16kHz mono
- Send smaller chunks more frequently

### Connection refused
- Check firewall settings
- Ensure service is running: `docker compose ps`
- Verify correct ports in client configuration

### High memory usage
- Models are cached in memory for performance
- Use smaller models for limited memory systems
- Set memory limits in docker-compose.yml

## Development

### Building from Source

```bash
# Install dependencies
pip install -r requirements.txt

# Generate gRPC code
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./src \
    --grpc_python_out=./src \
    ./proto/transcription.proto

# Run the service
python src/transcription_server.py
```

### Running Tests

```bash
# Test gRPC connection
grpcurl -plaintext localhost:50051 list

# Test health check
grpcurl -plaintext localhost:50051 transcription.TranscriptionService/HealthCheck

# Test with example audio
python test_client.py
```

## R&D Project Notice

This is a research and development project for exploring real-time transcription capabilities. It is not production-ready and should be used for experimentation and development purposes only.

### Known Limitations
- Memory usage scales with model size (1.5-6GB for large models)
- Single model instance shared across connections
- No authentication or rate limiting
- Not optimized for high-concurrency production use

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Support

- GitHub Issues: [Report bugs or request features]
- Documentation: [Full API documentation]
- Examples: See `examples/` directory