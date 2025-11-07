#!/usr/bin/env python3
"""
REST API for Transcription Service
Exposes transcription functionality via HTTP endpoints
"""

import os
import sys
import asyncio
import logging
import time
import base64
from typing import Optional, List
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcription_server import (
    ModelManager, TranscriptionEngine, get_global_model_manager,
    SAMPLE_RATE, MAX_AUDIO_LENGTH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transcription API",
    description="Real-time speech-to-text transcription service powered by OpenAI Whisper",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class AudioConfig(BaseModel):
    """Audio configuration for transcription"""
    language: str = Field(default="auto", description="Language code (e.g., 'en', 'es', 'auto')")
    task: str = Field(default="transcribe", description="Task: 'transcribe' or 'translate'")
    vad_enabled: bool = Field(default=True, description="Enable Voice Activity Detection")


class TranscriptionSegment(BaseModel):
    """A single transcription segment"""
    text: str
    start_time: float
    end_time: float
    confidence: float


class TranscriptionResponse(BaseModel):
    """Response for file transcription"""
    segments: List[TranscriptionSegment]
    full_text: str
    detected_language: str
    duration_seconds: float
    processing_time: float


class StreamTranscriptionResult(BaseModel):
    """Result for streaming transcription"""
    text: str
    start_time: float
    end_time: float
    is_final: bool
    confidence: float
    language: str
    timestamp_ms: int


class CapabilitiesResponse(BaseModel):
    """Service capabilities"""
    available_models: List[str]
    supported_languages: List[str]
    supported_formats: List[str]
    max_audio_length_seconds: int
    streaming_supported: bool
    vad_supported: bool


class HealthResponse(BaseModel):
    """Health check response"""
    healthy: bool
    status: str
    model_loaded: str
    uptime_seconds: int
    active_sessions: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


# Global state
start_time = time.time()
active_sessions = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the model manager on startup"""
    logger.info("Starting REST API...")
    model_manager = get_global_model_manager()
    logger.info(f"REST API initialized with model: {model_manager.get_model_name()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down REST API...")
    model_manager = get_global_model_manager()
    model_manager.cleanup()
    logger.info("REST API shutdown complete")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint"""
    return {
        "service": "Transcription API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "capabilities": "/capabilities",
            "transcribe": "/transcribe",
            "stream": "/stream"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        model_manager = get_global_model_manager()
        return HealthResponse(
            healthy=True,
            status="running",
            model_loaded=model_manager.get_model_name(),
            uptime_seconds=int(time.time() - start_time),
            active_sessions=len(active_sessions)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/capabilities", response_model=CapabilitiesResponse, tags=["Info"])
async def get_capabilities():
    """Get service capabilities"""
    return CapabilitiesResponse(
        available_models=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        supported_languages=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        supported_formats=["wav", "mp3", "webm", "ogg", "flac", "m4a", "raw_pcm16"],
        max_audio_length_seconds=MAX_AUDIO_LENGTH,
        streaming_supported=True,
        vad_supported=True
    )


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Transcription"])
async def transcribe_file(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(default="auto", description="Language code or 'auto'"),
    task: str = Form(default="transcribe", description="'transcribe' or 'translate'"),
    vad_enabled: bool = Form(default=True, description="Enable Voice Activity Detection")
):
    """
    Transcribe a complete audio file

    Supported formats: WAV, MP3, WebM, OGG, FLAC, M4A

    Example:
    ```bash
    curl -X POST "http://localhost:8000/transcribe" \
      -F "file=@audio.wav" \
      -F "language=en" \
      -F "task=transcribe"
    ```
    """
    start_processing = time.time()

    try:
        # Read file content
        audio_data = await file.read()

        # Validate file size
        if len(audio_data) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")

        # Get format from filename
        file_format = file.filename.split('.')[-1].lower()
        if file_format not in ["wav", "mp3", "webm", "ogg", "flac", "m4a", "pcm"]:
            file_format = "wav"  # Default to wav

        # Get transcription engine
        model_manager = get_global_model_manager()
        engine = TranscriptionEngine(model_manager)

        # Create config object
        from transcription_pb2 import AudioConfig as ProtoAudioConfig
        config = ProtoAudioConfig(
            language=language,
            task=task,
            vad_enabled=vad_enabled
        )

        # Transcribe
        result = engine.transcribe_file(audio_data, file_format, config)

        processing_time = time.time() - start_processing

        # Convert to response model
        segments = [
            TranscriptionSegment(
                text=seg['text'],
                start_time=seg['start_time'],
                end_time=seg['end_time'],
                confidence=seg['confidence']
            )
            for seg in result['segments']
        ]

        return TranscriptionResponse(
            segments=segments,
            full_text=result['full_text'],
            detected_language=result['detected_language'],
            duration_seconds=result['duration_seconds'],
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe/stream", tags=["Transcription"])
async def transcribe_stream(
    file: UploadFile = File(..., description="Audio file to stream and transcribe"),
    language: str = Form(default="auto", description="Language code or 'auto'"),
    vad_enabled: bool = Form(default=True, description="Enable Voice Activity Detection")
):
    """
    Stream transcription results as they are generated (Server-Sent Events)

    Returns a stream of JSON objects, one per line.

    Example:
    ```bash
    curl -X POST "http://localhost:8000/transcribe/stream" \
      -F "file=@audio.wav" \
      -F "language=en"
    ```
    """
    try:
        # Read file content
        audio_data = await file.read()

        # Get transcription engine
        model_manager = get_global_model_manager()
        engine = TranscriptionEngine(model_manager)

        async def generate():
            """Generate streaming transcription results"""
            import numpy as np

            try:
                # Convert audio to PCM16
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Process in chunks
                chunk_size = SAMPLE_RATE * 3  # 3 second chunks
                offset = 0

                while offset < len(audio):
                    chunk_end = min(offset + chunk_size, len(audio))
                    chunk = audio[offset:chunk_end]

                    # Convert back to bytes
                    chunk_bytes = (chunk * 32768.0).astype(np.int16).tobytes()

                    # Transcribe chunk
                    result = engine.transcribe_chunk(chunk_bytes, language=language, vad_enabled=vad_enabled)

                    if result:
                        yield f"data: {{\n"
                        yield f'  "text": "{result["text"]}",\n'
                        yield f'  "start_time": {result["start_time"]},\n'
                        yield f'  "end_time": {result["end_time"]},\n'
                        yield f'  "is_final": {str(result["is_final"]).lower()},\n'
                        yield f'  "confidence": {result["confidence"]},\n'
                        yield f'  "language": "{result.get("language", language)}",\n'
                        yield f'  "timestamp_ms": {int(time.time() * 1000)}\n'
                        yield "}\n\n"

                    offset = chunk_end

                    # Small delay to simulate streaming
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f'data: {{"error": "{str(e)}"}}\n\n'

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Stream setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Stream setup failed: {str(e)}")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming

    Protocol:
    1. Client connects
    2. Server sends: {"type": "connected", "session_id": "..."}
    3. Client sends: {"type": "audio", "data": "<base64-encoded-pcm16>"}
    4. Server sends: {"type": "transcription", "text": "...", ...}
    5. Client sends: {"type": "stop"} to end session

    Example (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/transcribe');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data);
    };
    ws.send(JSON.stringify({type: 'audio', data: base64AudioData}));
    ```
    """
    await websocket.accept()
    session_id = str(time.time())
    audio_buffer = bytearray()

    # Get transcription engine
    model_manager = get_global_model_manager()
    engine = TranscriptionEngine(model_manager)

    # Store session
    active_sessions[session_id] = {
        'start_time': time.time(),
        'last_activity': time.time()
    }

    try:
        # Send connection confirmation
        await websocket.send_json({
            'type': 'connected',
            'session_id': session_id
        })

        while True:
            # Receive message
            data = await websocket.receive_json()

            active_sessions[session_id]['last_activity'] = time.time()

            if data['type'] == 'audio':
                # Decode base64 audio
                audio_data = base64.b64decode(data['data'])
                audio_buffer.extend(audio_data)

                # Process when we have enough audio (3 seconds)
                min_bytes = int(SAMPLE_RATE * 3.0 * 2)  # 3 seconds of PCM16

                while len(audio_buffer) >= min_bytes:
                    chunk = bytes(audio_buffer[:min_bytes])
                    audio_buffer = audio_buffer[min_bytes:]

                    # Get config from data
                    language = data.get('language', 'auto')
                    vad_enabled = data.get('vad_enabled', True)

                    result = engine.transcribe_chunk(chunk, language=language, vad_enabled=vad_enabled)

                    if result:
                        await websocket.send_json({
                            'type': 'transcription',
                            'text': result['text'],
                            'start_time': result['start_time'],
                            'end_time': result['end_time'],
                            'is_final': result['is_final'],
                            'confidence': result.get('confidence', 0.9),
                            'language': result.get('language', language),
                            'timestamp_ms': int(time.time() * 1000)
                        })

            elif data['type'] == 'stop':
                # Process remaining audio
                if audio_buffer:
                    language = data.get('language', 'auto')
                    vad_enabled = data.get('vad_enabled', True)
                    result = engine.transcribe_chunk(bytes(audio_buffer), language=language, vad_enabled=vad_enabled)

                    if result:
                        await websocket.send_json({
                            'type': 'transcription',
                            'text': result['text'],
                            'start_time': result['start_time'],
                            'end_time': result['end_time'],
                            'is_final': True,
                            'confidence': result.get('confidence', 0.9),
                            'language': result.get('language', language),
                            'timestamp_ms': int(time.time() * 1000)
                        })

                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            'type': 'error',
            'error': str(e)
        })
    finally:
        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]


# Additional utility endpoints

@app.get("/sessions", tags=["Info"])
async def list_sessions():
    """List active transcription sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "start_time": info['start_time'],
                "last_activity": info['last_activity'],
                "duration": time.time() - info['start_time']
            }
            for sid, info in active_sessions.items()
        ]
    }


@app.post("/test", tags=["Testing"])
async def test_transcription():
    """
    Test endpoint that returns a sample transcription
    Useful for testing without audio files
    """
    return {
        "text": "This is a test transcription.",
        "language": "en",
        "duration": 2.5,
        "timestamp": int(time.time() * 1000)
    }


def main():
    """Run the REST API server"""
    port = int(os.environ.get('REST_PORT', '8000'))
    host = os.environ.get('REST_HOST', '0.0.0.0')

    logger.info(f"Starting REST API server on {host}:{port}")

    uvicorn.run(
        "rest_api:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
