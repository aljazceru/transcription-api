#!/usr/bin/env python3
"""
Standalone Transcription Service with gRPC and WebSocket support
Optimized for real-time streaming transcription
"""

import os
import sys
import asyncio
import logging
import time
import json
import base64
from typing import Optional, AsyncIterator, Dict, List
from dataclasses import dataclass, asdict
from concurrent import futures
import threading
from datetime import datetime
import atexit

# Add current directory to path for generated protobuf imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grpc
import numpy as np
import soundfile
import librosa
import torch

# Add SimulStreaming to path if available
simulstreaming_path = os.environ.get('SIMULSTREAMING_PATH', '/app/SimulStreaming')
if os.path.exists(simulstreaming_path):
    sys.path.insert(0, simulstreaming_path)
    USE_SIMULSTREAMING = True
    try:
        from simulstreaming_whisper import simulwhisper_args, simul_asr_factory
    except ImportError:
        USE_SIMULSTREAMING = False
        import whisper
else:
    USE_SIMULSTREAMING = False
    import whisper

# Import generated protobuf classes (will be generated later)
from transcription_pb2 import (
    AudioChunk, AudioFile, TranscriptionResult, TranscriptionResponse,
    TranscriptionSegment, Capabilities, HealthStatus, Empty, AudioConfig
)
import transcription_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 30 * 60  # 30 minutes


@dataclass
class TranscriptionSession:
    """Manages a single transcription session"""
    session_id: str
    config: AudioConfig
    audio_buffer: bytearray
    start_time: float
    last_activity: float
    transcriptions: List[dict]


class ModelManager:
    """Singleton manager for Whisper model to share across all connections"""
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    _device = None
    _model_name = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, model_name: str = "large-v3"):
        """Initialize the model (only once)"""
        with self._lock:
            if not self._initialized:
                self._model_name = model_name
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._load_model()
                self._initialized = True
                logger.info(f"ModelManager initialized with {model_name} on {self._device}")
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            download_root = os.environ.get('TORCH_HOME', '/app/models')
            self._model = whisper.load_model(
                self._model_name, 
                device=self._device, 
                download_root=download_root
            )
            logger.info(f"Loaded shared Whisper model: {self._model_name} on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def get_model(self):
        """Get the shared model instance"""
        if not self._initialized:
            raise RuntimeError("ModelManager not initialized. Call initialize() first.")
        return self._model
    
    def get_device(self):
        """Get the device being used"""
        return self._device
    
    def get_model_name(self):
        """Get the model name"""
        return self._model_name
    
    def cleanup(self):
        """Cleanup resources (call on shutdown)"""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._initialized = False
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                logger.info("ModelManager cleaned up")


class TranscriptionEngine:
    """Core transcription engine using shared Whisper model"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize with optional model manager (for testing)"""
        self.model_manager = model_manager or ModelManager()
        self.model = None  # Will get from manager
        self.processor = None
        self.online_processor = None
        self.device = self.model_manager.get_device()
        self.model_name = self.model_manager._model_name
    
    def load_model(self):
        """Load the transcription model"""
        # Model is already loaded in ModelManager
        # This method is kept for compatibility
        pass
    
    def get_model(self):
        """Get the shared model instance from ModelManager"""
        return self.model_manager.get_model()
    
    def is_speech(self, audio: np.ndarray, energy_threshold: float = 0.002, zero_crossing_threshold: int = 50) -> bool:
        """
        Simple Voice Activity Detection
        Returns True if the audio chunk likely contains speech
        """
        # Check if audio is too quiet (likely silence)
        energy = np.sqrt(np.mean(audio**2))
        if energy < energy_threshold:
            return False
        
        # Check zero crossing rate (helps distinguish speech from noise)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
        
        # Speech typically has moderate zero crossing rate
        # Pure noise tends to have very high zero crossing rate
        if zero_crossings > len(audio) * zero_crossing_threshold / SAMPLE_RATE:
            return False
            
        return True
    
    def transcribe_chunk(self, audio_data: bytes, language: str = "auto", vad_enabled: bool = True) -> Optional[dict]:
        """Transcribe a single audio chunk"""
        try:
            # Convert bytes to numpy array
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check if audio contains speech (VAD) - only if enabled
            if vad_enabled:
                energy = np.sqrt(np.mean(audio**2))
                if not self.is_speech(audio):
                    logger.info(f"No speech detected in audio chunk (energy: {energy:.4f}), skipping transcription")
                    return None
                else:
                    logger.info(f"Speech detected in chunk (energy: {energy:.4f})")
            
            if USE_SIMULSTREAMING and self.online_processor:
                # Use SimulStreaming for real-time processing
                self.online_processor.insert_audio_chunk(audio)
                result = self.online_processor.process_iter()
                
                if result and result[0] is not None:
                    return {
                        'text': result[2],
                        'start_time': result[0],
                        'end_time': result[1],
                        'is_final': True,
                        'confidence': 0.95  # SimulStreaming doesn't provide confidence
                    }
            else:
                # Use standard Whisper
                model = self.get_model()
                if model:
                    # Pad audio to minimum length if needed
                    if len(audio) < SAMPLE_RATE:
                        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
                    
                    # Use more conservative settings to reduce hallucinations
                    result = model.transcribe(
                        audio,
                        language=None if language == "auto" else language,
                        fp16=self.device == "cuda",
                        temperature=0.0,  # More deterministic, less hallucination
                        no_speech_threshold=0.6,  # Higher threshold for detecting non-speech
                        logprob_threshold=-1.0,  # Filter out low probability results
                        compression_ratio_threshold=2.4  # Filter out repetitive results
                    )
                    
                    if result and result.get('text'):
                        text = result['text'].strip()
                        
                        # Filter out common hallucinations
                        hallucination_phrases = [
                            "thank you", "thanks", "you", "uh", "um", 
                            "thank you for watching", "please subscribe",
                            "bye", "bye-bye", ".", "...", ""
                        ]
                        
                        # Check if the result is just a hallucination
                        text_lower = text.lower().strip()
                        if text_lower in hallucination_phrases:
                            logger.debug(f"Filtered out hallucination: {text}")
                            return None
                        
                        # Check for repetitive text (another sign of hallucination)
                        words = text.lower().split()
                        if len(words) > 1 and len(set(words)) == 1:
                            logger.debug(f"Filtered out repetitive text: {text}")
                            return None
                        
                        return {
                            'text': text,
                            'start_time': 0,
                            'end_time': len(audio) / SAMPLE_RATE,
                            'is_final': True,
                            'confidence': 0.9,
                            'language': result.get('language', language)
                        }
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
        
        return None
    
    def transcribe_file(self, audio_data: bytes, format: str, config: AudioConfig) -> dict:
        """Transcribe a complete audio file"""
        try:
            # Convert audio to numpy array based on format
            if format == "raw_pcm16":
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # Use librosa for other formats
                import io
                audio, _ = librosa.load(io.BytesIO(audio_data), sr=SAMPLE_RATE)
            
            # Transcribe with Whisper
            model = self.get_model()
            if model:
                result = model.transcribe(
                    audio,
                    language=None if config.language == "auto" else config.language,
                    task=config.task or "transcribe",
                    fp16=self.device == "cuda"
                )
                
                segments = []
                for seg in result.get('segments', []):
                    segments.append({
                        'text': seg['text'].strip(),
                        'start_time': seg['start'],
                        'end_time': seg['end'],
                        'confidence': seg.get('avg_logprob', 0) + 1.0  # Convert to 0-1 range
                    })
                
                return {
                    'segments': segments,
                    'full_text': result['text'].strip(),
                    'detected_language': result.get('language', config.language),
                    'duration_seconds': len(audio) / SAMPLE_RATE
                }
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
        
        return {
            'segments': [],
            'full_text': '',
            'detected_language': 'unknown',
            'duration_seconds': 0
        }


class TranscriptionServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    """gRPC service implementation"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.engine = TranscriptionEngine(self.model_manager)
        self.sessions: Dict[str, TranscriptionSession] = {}
        self.start_time = time.time()
    
    async def StreamTranscribe(self, request_iterator: AsyncIterator[AudioChunk],
                              context: grpc.aio.ServicerContext) -> AsyncIterator[TranscriptionResult]:
        """Bidirectional streaming transcription"""
        session_id = None
        config = None
        audio_buffer = bytearray()
        
        try:
            async for chunk in request_iterator:
                # Get session ID and config from first chunk
                if not session_id:
                    session_id = chunk.session_id or str(time.time())
                    config = chunk.config or AudioConfig(
                        language="auto",
                        task="transcribe"
                    )
                
                # Add audio to buffer
                audio_buffer.extend(chunk.audio_data)
                
                # Process when we have enough audio (3 seconds for better accuracy)
                min_bytes = int(SAMPLE_RATE * 3.0 * 2)  # 3 seconds of PCM16
                
                while len(audio_buffer) >= min_bytes:
                    # Extract chunk to process
                    audio_chunk = bytes(audio_buffer[:min_bytes])
                    audio_buffer = audio_buffer[min_bytes:]
                    
                    # Transcribe
                    logger.debug(f"Processing audio chunk of {len(audio_chunk)} bytes")
                    result = self.engine.transcribe_chunk(
                        audio_chunk,
                        language=config.language,
                        vad_enabled=config.vad_enabled if config else False
                    )
                    logger.debug(f"Transcription result: {result}")
                    
                    if result:
                        # Send transcription result
                        yield TranscriptionResult(
                            text=result['text'],
                            start_time=result['start_time'],
                            end_time=result['end_time'],
                            is_final=result['is_final'],
                            confidence=result.get('confidence', 0.9),
                            language=result.get('language', config.language),
                            session_id=session_id,
                            timestamp_ms=int(time.time() * 1000)
                        )
            
            # Process remaining audio
            if audio_buffer:
                result = self.engine.transcribe_chunk(
                    bytes(audio_buffer),
                    language=config.language,
                    vad_enabled=config.vad_enabled if config else False
                )
                
                if result:
                    yield TranscriptionResult(
                        text=result['text'],
                        start_time=result['start_time'],
                        end_time=result['end_time'],
                        is_final=True,
                        confidence=result.get('confidence', 0.9),
                        language=result.get('language', config.language),
                        session_id=session_id,
                        timestamp_ms=int(time.time() * 1000)
                    )
                    
        except Exception as e:
            logger.error(f"Error in StreamTranscribe: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def TranscribeFile(self, request: AudioFile, context: grpc.aio.ServicerContext) -> TranscriptionResponse:
        """Transcribe a complete audio file"""
        try:
            result = self.engine.transcribe_file(
                request.audio_data,
                request.format,
                request.config
            )
            
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
                duration_seconds=result['duration_seconds']
            )
            
        except Exception as e:
            logger.error(f"Error in TranscribeFile: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetCapabilities(self, request: Empty, context: grpc.aio.ServicerContext) -> Capabilities:
        """Get service capabilities"""
        return Capabilities(
            available_models=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
            supported_languages=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
            supported_formats=["wav", "mp3", "webm", "raw_pcm16"],
            max_audio_length_seconds=MAX_AUDIO_LENGTH,
            streaming_supported=True,
            vad_supported=False  # Can be implemented later
        )
    
    async def HealthCheck(self, request: Empty, context: grpc.aio.ServicerContext) -> HealthStatus:
        """Health check endpoint"""
        return HealthStatus(
            healthy=True,
            status="running",
            model_loaded=self.engine.model_name,
            uptime_seconds=int(time.time() - self.start_time),
            active_sessions=len(self.sessions)
        )


async def serve_grpc(port: int = 50051):
    """Start the gRPC server"""
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    
    # Use the shared model manager
    model_manager = get_global_model_manager()
    servicer = TranscriptionServicer(model_manager)
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    
    logger.info(f"gRPC server started on port {port}")
    await server.wait_for_termination()


# Global model manager instance (shared across all handlers)
_global_model_manager = None

def get_global_model_manager() -> ModelManager:
    """Get or create the global model manager"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
        model_name = os.environ.get('MODEL_PATH', 'large-v3')
        _global_model_manager.initialize(model_name)
    return _global_model_manager


# WebSocket support for compatibility
async def handle_websocket(websocket, path):
    """Handle WebSocket connections for compatibility"""
    import websockets
    
    # Use the shared model manager instead of creating new engine
    model_manager = get_global_model_manager()
    engine = TranscriptionEngine(model_manager)
    session_id = str(time.time())
    audio_buffer = bytearray()
    
    try:
        # Send connection confirmation
        await websocket.send(json.dumps({
            'type': 'connected',
            'session_id': session_id
        }))
        
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'audio':
                # Decode base64 audio
                audio_data = base64.b64decode(data['data'])
                audio_buffer.extend(audio_data)
                
                # Process when we have enough audio (3 seconds for better accuracy)
                min_bytes = int(SAMPLE_RATE * 3.0 * 2)  # 3 seconds of PCM16
                
                while len(audio_buffer) >= min_bytes:
                    chunk = bytes(audio_buffer[:min_bytes])
                    audio_buffer = audio_buffer[min_bytes:]
                    
                    result = engine.transcribe_chunk(chunk)
                    
                    if result:
                        await websocket.send(json.dumps({
                            'type': 'transcription',
                            'text': result['text'],
                            'start_time': result['start_time'],
                            'end_time': result['end_time'],
                            'is_final': result['is_final'],
                            'timestamp': int(time.time() * 1000)
                        }))
            
            elif data['type'] == 'stop':
                # Process remaining audio
                if audio_buffer:
                    result = engine.transcribe_chunk(bytes(audio_buffer))
                    if result:
                        await websocket.send(json.dumps({
                            'type': 'transcription',
                            'text': result['text'],
                            'is_final': True,
                            'timestamp': int(time.time() * 1000)
                        }))
                break
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


async def serve_websocket(port: int = 8765):
    """Start the WebSocket server"""
    import websockets
    
    logger.info(f"WebSocket server started on port {port}")
    async with websockets.serve(handle_websocket, "0.0.0.0", port):
        await asyncio.Future()  # Run forever


async def main():
    """Main entry point"""
    grpc_port = int(os.environ.get('GRPC_PORT', '50051'))
    ws_port = int(os.environ.get('WEBSOCKET_PORT', '8765'))
    enable_websocket = os.environ.get('ENABLE_WEBSOCKET', 'true').lower() == 'true'
    
    # Initialize the global model manager once at startup
    logger.info("Initializing shared model manager...")
    model_manager = get_global_model_manager()
    logger.info(f"Model manager initialized with model: {model_manager._model_name}")
    
    try:
        tasks = [serve_grpc(grpc_port)]
        
        if enable_websocket:
            tasks.append(serve_websocket(ws_port))
        
        await asyncio.gather(*tasks)
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down, cleaning up model manager...")
        if model_manager:
            model_manager.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())