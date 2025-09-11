//! Live microphone transcription using gRPC streaming
//! 
//! This example shows how to capture audio from the microphone
//! and stream it to the transcription service in real-time.

use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

// Import generated protobuf types
pub mod transcription {
    tonic::include_proto!("transcription");
}

use transcription::{
    transcription_service_client::TranscriptionServiceClient, AudioChunk, AudioConfig,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// gRPC server address
    #[arg(short, long, default_value = "http://localhost:50051")]
    server: String,

    /// Language code (e.g., "en", "es", "auto")
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Task: transcribe or translate
    #[arg(short, long, default_value = "transcribe")]
    task: String,

    /// Model to use
    #[arg(short, long, default_value = "base")]
    model: String,

    /// Session ID
    #[arg(long)]
    session_id: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Connecting to transcription service at {}", args.server);
    let mut client = TranscriptionServiceClient::connect(args.server).await?;

    // Create channel for audio data
    let (audio_tx, audio_rx) = tokio_mpsc::channel::<AudioChunk>(100);

    // Start audio capture in a separate thread
    let audio_tx_clone = audio_tx.clone();
    std::thread::spawn(move || {
        if let Err(e) = capture_audio(audio_tx_clone) {
            error!("Audio capture error: {}", e);
        }
    });

    // Send initial configuration
    let session_id = args.session_id.unwrap_or_else(|| {
        format!("rust-client-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs())
    });

    info!("Starting transcription session: {}", session_id);

    // Create the first chunk with configuration
    let config = AudioConfig {
        language: args.language.clone(),
        task: args.task.clone(),
        model: args.model.clone(),
        sample_rate: 16000,
        vad_enabled: false,
    };

    // Send a configuration chunk first
    let config_chunk = AudioChunk {
        audio_data: vec![],
        session_id: session_id.clone(),
        config: Some(config),
    };
    
    audio_tx.send(config_chunk).await?;

    // Create stream from receiver
    let audio_stream = ReceiverStream::new(audio_rx);

    // Start bidirectional streaming
    let response = client.stream_transcribe(audio_stream).await?;
    let mut stream = response.into_inner();

    info!("Listening... Press Ctrl+C to stop");

    // Process transcription results
    while let Some(result) = stream.next().await {
        match result {
            Ok(transcription) => {
                if transcription.is_final {
                    println!("\n[FINAL] {}", transcription.text);
                } else {
                    print!("\r[PARTIAL] {}       ", transcription.text);
                    use std::io::{self, Write};
                    io::stdout().flush()?;
                }
            }
            Err(e) => {
                error!("Transcription error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// Capture audio from the default microphone
fn capture_audio(tx: tokio_mpsc::Sender<AudioChunk>) -> Result<()> {
    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;
    
    info!("Using audio device: {}", device.name()?);

    // Configure audio capture for 16kHz mono PCM16
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    // Buffer to accumulate audio samples
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = buffer.clone();

    // Create audio stream
    let stream = device.build_input_stream(
        &config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            let mut buf = buffer_clone.lock().unwrap();
            buf.extend_from_slice(data);
            
            // Send chunks of ~3 seconds (48000 samples at 16kHz) for better accuracy
            while buf.len() >= 48000 {
                let chunk: Vec<i16> = buf.drain(..48000).collect();
                
                // Convert i16 to bytes
                let bytes: Vec<u8> = chunk.iter()
                    .flat_map(|&sample| sample.to_le_bytes())
                    .collect();
                
                // Send audio chunk
                let audio_chunk = AudioChunk {
                    audio_data: bytes,
                    session_id: String::new(),  // Already set in config chunk
                    config: None,
                };
                
                // Use blocking send since we're in a non-async context
                if let Err(e) = tx.blocking_send(audio_chunk) {
                    error!("Failed to send audio chunk: {}", e);
                }
            }
        },
        move |err| {
            error!("Audio stream error: {}", err);
        },
        None,
    )?;

    stream.play()?;

    // Keep the stream alive
    std::thread::park();

    Ok(())
}