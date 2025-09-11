//! File transcription using gRPC
//! 
//! This example shows how to transcribe an audio file.
//! Use --stream flag for real-time streaming instead of sending the entire file.

use anyhow::Result;
use clap::Parser;
use std::fs;
use tonic::transport::Channel;
use tracing::{info, debug};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::time::Duration;
use tokio::time;

// Import generated protobuf types
pub mod transcription {
    tonic::include_proto!("transcription");
}

use transcription::{
    transcription_service_client::TranscriptionServiceClient, AudioConfig, AudioFile, AudioChunk,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Audio file path
    file: String,

    /// gRPC server address
    #[arg(short, long, default_value = "http://localhost:50051")]
    server: String,

    /// Language code (e.g., "en", "es", "auto")
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Task: transcribe or translate
    #[arg(short, long, default_value = "transcribe")]
    task: String,

    /// Model to use
    #[arg(short, long, default_value = "base")]
    model: String,

    /// Stream the file in chunks for real-time transcription
    #[arg(long)]
    stream: bool,

    /// Enable VAD (Voice Activity Detection)
    #[arg(short, long)]
    vad: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Reading audio file: {}", args.file);
    
    info!("Connecting to transcription service at {}", args.server);
    let mut client = TranscriptionServiceClient::connect(args.server.clone()).await?;

    if args.stream {
        // Stream mode - send file in chunks for real-time transcription
        stream_file(&mut client, &args).await?;
    } else {
        // Normal mode - send entire file at once
        transcribe_entire_file(&mut client, &args).await?;
    }

    Ok(())
}

async fn transcribe_entire_file(
    client: &mut TranscriptionServiceClient<Channel>,
    args: &Args,
) -> Result<()> {
    let audio_data = fs::read(&args.file)?;

    // Determine format from extension
    let format = match args.file.split('.').last() {
        Some("wav") => "wav",
        Some("mp3") => "mp3",
        Some("webm") => "webm",
        _ => "wav",  // Default to WAV
    };

    let config = AudioConfig {
        language: args.language.clone(),
        task: args.task.clone(),
        model: args.model.clone(),
        sample_rate: 16000,
        vad_enabled: args.vad,
    };

    let request = AudioFile {
        audio_data,
        format: format.to_string(),
        config: Some(config),
    };

    info!("Sending entire file for transcription...");
    let response = client.transcribe_file(request).await?;
    let result = response.into_inner();

    println!("\n=== Transcription Results ===");
    println!("Language: {}", result.detected_language);
    println!("Duration: {:.2} seconds", result.duration_seconds);
    println!("\nFull Text:");
    println!("{}", result.full_text);
    
    if !result.segments.is_empty() {
        println!("\n=== Segments ===");
        for (i, segment) in result.segments.iter().enumerate() {
            println!(
                "[{:03}] {:.2}s - {:.2}s (conf: {:.2}): {}",
                i + 1,
                segment.start_time,
                segment.end_time,
                segment.confidence,
                segment.text
            );
        }
    }

    Ok(())
}

async fn stream_file(
    client: &mut TranscriptionServiceClient<Channel>,
    args: &Args,
) -> Result<()> {
    let audio_data = fs::read(&args.file)?;
    
    info!("Streaming file in real-time chunks...");
    
    // Create channel for audio chunks
    let (tx, rx) = mpsc::channel::<AudioChunk>(100);
    
    // Spawn task to send audio chunks
    let tx_clone = tx.clone();
    let language = args.language.clone();
    let task = args.task.clone();
    let model = args.model.clone();
    let vad = args.vad;
    
    tokio::spawn(async move {
        // Send configuration first
        let config = AudioConfig {
            language,
            task,
            model,
            sample_rate: 16000,
            vad_enabled: vad,
        };

        let config_chunk = AudioChunk {
            audio_data: vec![],
            session_id: "file-stream".to_string(),
            config: Some(config),
        };
        
        if tx_clone.send(config_chunk).await.is_err() {
            return;
        }

        // Assuming PCM16 audio at 16kHz
        // Send in 3 second chunks for better accuracy (96000 bytes = 48000 samples = 3 seconds)
        let chunk_size = 96000;
        
        for (idx, chunk) in audio_data.chunks(chunk_size).enumerate() {
            let audio_chunk = AudioChunk {
                audio_data: chunk.to_vec(),
                session_id: String::new(),
                config: None,
            };

            debug!("Sending chunk {} ({} bytes)", idx, chunk.len());
            
            if tx_clone.send(audio_chunk).await.is_err() {
                break;
            }

            // Simulate real-time streaming (3 seconds per chunk)
            time::sleep(Duration::from_secs(3)).await;
        }
        
        info!("Finished streaming all chunks");
    });

    // Create stream and start transcription
    let stream = ReceiverStream::new(rx);
    let response = client.stream_transcribe(stream).await?;
    let mut result_stream = response.into_inner();

    println!("\n=== Real-time Transcription ===");
    println!("Streaming and transcribing...\n");

    let mut full_transcript = String::new();
    
    // Process results
    while let Some(result) = result_stream.next().await {
        match result {
            Ok(transcription) => {
                println!("[{:06.2}s - {:06.2}s] {}", 
                         transcription.start_time,
                         transcription.end_time,
                         transcription.text);
                
                if transcription.is_final {
                    full_transcript.push_str(&transcription.text);
                    full_transcript.push(' ');
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
    
    println!("\n=== Full Transcript ===");
    println!("{}", full_transcript.trim());

    Ok(())
}