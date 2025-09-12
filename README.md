# Transcription API Service

A high-performance, standalone transcription service with gRPC and WebSocket support, optimized for real-time speech-to-text applications. Perfect for desktop applications, web services, and IoT devices.


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
