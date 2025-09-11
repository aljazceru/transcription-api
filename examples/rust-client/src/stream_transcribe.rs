//! Stream WAV file for real-time transcription
//! 
//! This example shows how to stream a WAV file chunk by chunk
//! to simulate real-time transcription.

use anyhow::Result;
use clap::Parser;
use futures_util::StreamExt;
use hound::WavReader;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

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
    /// WAV file path
    file: String,

    /// gRPC server address
    #[arg(short, long, default_value = "http://localhost:50051")]
    server: String,

    /// Language code (e.g., "en", "es", "auto")
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Simulate real-time by adding delays
    #[arg(short, long)]
    realtime: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Reading WAV file: {}", args.file);
    let mut reader = WavReader::open(&args.file)?;
    let spec = reader.spec();
    
    info!("WAV specs: {} Hz, {} channels, {} bits", 
          spec.sample_rate, spec.channels, spec.bits_per_sample);

    // Collect samples
    let samples: Vec<i16> = reader.samples::<i16>()
        .collect::<hound::Result<Vec<_>>>()?;

    info!("Connecting to transcription service at {}", args.server);
    let mut client = TranscriptionServiceClient::connect(args.server).await?;

    // Create channel for audio chunks
    let (tx, rx) = mpsc::channel::<AudioChunk>(100);

    // Spawn task to send audio chunks
    let tx_clone = tx.clone();
    let realtime = args.realtime;
    tokio::spawn(async move {
        // Send configuration first
        let config = AudioConfig {
            language: args.language.clone(),
            task: "transcribe".to_string(),
            model: "base".to_string(),
            sample_rate: 16000,
            vad_enabled: false,
        };

        let config_chunk = AudioChunk {
            audio_data: vec![],
            session_id: "stream-test".to_string(),
            config: Some(config),
        };
        
        if tx_clone.send(config_chunk).await.is_err() {
            return;
        }

        // Send audio in chunks of 3 seconds for better accuracy (48000 samples at 16kHz)
        let chunk_size = 48000;
        for chunk in samples.chunks(chunk_size) {
            // Convert samples to bytes
            let bytes: Vec<u8> = chunk.iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect();

            let audio_chunk = AudioChunk {
                audio_data: bytes,
                session_id: String::new(),
                config: None,
            };

            if tx_clone.send(audio_chunk).await.is_err() {
                break;
            }

            // Simulate real-time streaming
            if realtime {
                time::sleep(Duration::from_secs(3)).await;
            }
        }
    });

    // Create stream and start transcription
    let stream = ReceiverStream::new(rx);
    let response = client.stream_transcribe(stream).await?;
    let mut result_stream = response.into_inner();

    info!("Streaming audio and receiving transcriptions...");

    // Process results
    while let Some(result) = result_stream.next().await {
        match result {
            Ok(transcription) => {
                println!("[{:.2}s - {:.2}s] {}", 
                         transcription.start_time,
                         transcription.end_time,
                         transcription.text);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }

    Ok(())
}