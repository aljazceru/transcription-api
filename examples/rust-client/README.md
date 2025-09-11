# Rust Transcription Client Examples

This directory contains Rust client examples for the Transcription API service.

## Available Clients

### 1. `file-transcribe` - File Transcription
Transcribe audio files either by sending the entire file or streaming in real-time chunks.

```bash
# Send entire file at once (fast, but no real-time feedback)
cargo run --bin file-transcribe -- audio.wav

# Stream file in chunks for real-time transcription (like YouTube)
cargo run --bin file-transcribe -- audio.wav --stream

# With VAD (Voice Activity Detection) to filter silence
cargo run --bin file-transcribe -- audio.wav --stream --vad

# Specify model and language
cargo run --bin file-transcribe -- audio.wav --stream --model large-v3 --language en
```

### 2. `realtime-playback` - Play Audio with Live Transcription
Plays audio through your speakers while showing real-time transcriptions, similar to YouTube's live captions.

```bash
# Basic usage - plays audio and shows transcriptions
cargo run --bin realtime-playback -- audio.wav

# With timestamps for each transcription
cargo run --bin realtime-playback -- audio.wav --timestamps

# With VAD to reduce noise transcriptions
cargo run --bin realtime-playback -- audio.wav --vad

# Using a specific model
cargo run --bin realtime-playback -- audio.wav --model large-v3
```

### 3. `stream-transcribe` - Stream WAV Files
Streams WAV files chunk by chunk for transcription.

```bash
# Stream without delays (fast processing)
cargo run --bin stream-transcribe -- audio.wav

# Simulate real-time streaming with delays
cargo run --bin stream-transcribe -- audio.wav --realtime
```

### 4. `live-transcribe` - Live Microphone Transcription
Captures audio from your microphone and transcribes in real-time.

```bash
# Use default microphone
cargo run --bin live-transcribe

# Specify server and language
cargo run --bin live-transcribe -- --server http://localhost:50051 --language en
```

### 5. `stdin-transcribe` - Transcribe Audio from stdin
Accepts audio data from stdin, perfect for piping from other tools.

```bash
# Pipe audio from parec (PulseAudio/PipeWire)
parec --format=s16le --rate=16000 --channels=1 | cargo run --bin stdin-transcribe

# With options
parec --format=s16le --rate=16000 --channels=1 | \
  cargo run --bin stdin-transcribe -- --language en --no-vad --chunk-seconds 2.5
```

### 6. `system-audio` - System Audio Capture
Attempts to capture system audio using available audio devices.

```bash
# List available audio devices
cargo run --bin system-audio -- --list-devices

# Capture from specific device
cargo run --bin system-audio -- --device pulse
```

## Video Call & System Audio Transcription

### Transcribe Video Calls (Zoom, Teams, Meet, etc.)
Use the provided script to transcribe any video call or system audio:

```bash
# Transcribe system audio (video calls, YouTube, etc.)
./transcribe_video_call.sh

# List available audio sources
./transcribe_video_call.sh --list

# Use microphone instead of system audio
./transcribe_video_call.sh --microphone
```

### Quick YouTube/System Audio Test
```bash
# Test with any playing audio (YouTube, music, etc.)
./test_youtube.sh
```

**Note**: System audio capture requires `pulseaudio-utils` package:
```bash
sudo apt-get install pulseaudio-utils
```

## Building

```bash
# Build all binaries
cargo build --release

# Build specific binary
cargo build --release --bin realtime-playback
```

## Common Options

All clients support these common options:
- `--server <URL>` - gRPC server address (default: http://localhost:50051)
- `--language <code>` - Language code: en, es, fr, de, etc., or "auto" (default: auto)
- `--model <name>` - Model to use: tiny, base, small, medium, large-v3 (default: base)
- `--vad` - Enable Voice Activity Detection to filter silence

## Features

### Real-time Streaming
The `--stream` flag in `file-transcribe` and the `realtime-playback` binary both support real-time streaming, which means:
- Audio is sent in small chunks (0.5 second intervals)
- Transcriptions appear as the audio is being processed
- Similar experience to YouTube's live captions
- Lower latency compared to sending entire file

### Voice Activity Detection (VAD)
When `--vad` is enabled, the service will:
- Filter out silence and background noise
- Reduce false transcriptions (like repeated "Thank you")
- Improve transcription quality for speech-only content

### Audio Playback
The `realtime-playback` binary uses the `rodio` library to:
- Play audio through your system's default audio output
- Synchronize playback with transcription display
- Support multiple audio formats (WAV, MP3, FLAC, etc.)

## Requirements

- Rust 1.70 or later
- The Transcription API server running (usually on localhost:50051)
- For live transcription: A working microphone
- For playback: Audio output device (speakers/headphones)

## System Requirements

### For Video Call Transcription (Ubuntu/Linux)
- PulseAudio utilities: `sudo apt-get install pulseaudio-utils`
- PipeWire or PulseAudio audio server
- The monitor audio source must be available

## Troubleshooting

### "Connection refused" error
Make sure the Transcription API server is running:
```bash
cd ../../
docker compose up
```

### No audio playback
- Check your system's default audio output device
- Ensure the audio file format is supported (WAV, MP3, FLAC)
- Try with a different audio file

### Poor transcription quality
- Use a larger model (e.g., `--model large-v3`)
- For system audio: use `--no-vad` flag to disable voice activity detection
- Ensure audio quality is good (16kHz or higher recommended)
- Use 2.5-3 second chunks for optimal accuracy

### System audio not working
- Install pulseaudio-utils: `sudo apt-get install pulseaudio-utils`
- Check monitor source exists: `./transcribe_video_call.sh --list`
- Make sure audio is playing when you start transcription
- Use headphones to avoid echo/feedback