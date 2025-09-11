//! Real-time audio playback with synchronized transcription
//! 
//! This example plays an audio file while streaming it for transcription,
//! showing transcriptions in real-time similar to YouTube.

use anyhow::Result;
use clap::Parser;
use futures_util::StreamExt;
use rodio::{Decoder, OutputStream, Source};
use std::fs::File;
use std::io::BufReader;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, debug};
use hound::WavReader;

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
    /// Audio file path (WAV, MP3, FLAC, etc.)
    file: String,

    /// gRPC server address
    #[arg(short, long, default_value = "http://localhost:50051")]
    server: String,

    /// Language code (e.g., "en", "es", "auto")
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Model to use
    #[arg(short, long, default_value = "base")]
    model: String,

    /// Enable VAD (Voice Activity Detection)
    #[arg(short, long)]
    vad: bool,

    /// Show timestamps
    #[arg(short = 't', long)]
    timestamps: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let file_path = args.file.clone();

    info!("Loading audio file: {}", file_path);
    
    // Start audio playback in a separate thread
    let (_stream, stream_handle) = OutputStream::try_default()?;
    let file = BufReader::new(File::open(&file_path)?);
    let source = Decoder::new(file)?;
    let sample_rate = source.sample_rate();
    let channels = source.channels();
    // Convert to f32 samples for playback
    let source = source.convert_samples::<f32>();
    
    info!("Audio format: {} Hz, {} channels", sample_rate, channels);

    // Also open the file for streaming to transcription service
    // We need to read the raw audio data for transcription
    let mut wav_reader = WavReader::open(&file_path)?;
    let _wav_spec = wav_reader.spec();
    
    // Collect all samples for streaming
    let samples: Vec<i16> = wav_reader.samples::<i16>()
        .collect::<hound::Result<Vec<_>>>()?;
    
    info!("Connecting to transcription service at {}", args.server);
    let mut client = TranscriptionServiceClient::connect(args.server.clone()).await?;

    // Create channel for audio chunks
    let (tx, rx) = mpsc::channel::<AudioChunk>(100);
    
    // Calculate chunk duration for synchronization
    let chunk_samples = 16000 * 3; // 3 second chunks at 16kHz for better accuracy
    let chunk_duration = Duration::from_secs(3);
    
    // Start playback
    println!("\n🎵 Starting audio playback with real-time transcription...\n");
    println!("{}", "─".repeat(80));
    
    let start_time = Instant::now();
    
    // Play audio
    stream_handle.play_raw(source)?;
    
    // Spawn task to stream audio chunks to transcription service
    let tx_clone = tx.clone();
    let show_timestamps = args.timestamps;
    tokio::spawn(async move {
        // Send configuration first
        let config = AudioConfig {
            language: args.language.clone(),
            task: "transcribe".to_string(),
            model: args.model.clone(),
            sample_rate: 16000,
            vad_enabled: args.vad,
        };

        let config_chunk = AudioChunk {
            audio_data: vec![],
            session_id: "realtime-playback".to_string(),
            config: Some(config),
        };
        
        if tx_clone.send(config_chunk).await.is_err() {
            return;
        }

        // Stream audio chunks synchronized with playback
        for (chunk_idx, chunk) in samples.chunks(chunk_samples).enumerate() {
            let chunk_start = Instant::now();
            
            // Convert samples to bytes
            let bytes: Vec<u8> = chunk.iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect();

            let audio_chunk = AudioChunk {
                audio_data: bytes,
                session_id: String::new(),
                config: None,
            };

            debug!("Sending chunk {} ({} samples)", chunk_idx, chunk.len());
            
            if tx_clone.send(audio_chunk).await.is_err() {
                break;
            }

            // Synchronize with playback timing
            let elapsed = chunk_start.elapsed();
            if elapsed < chunk_duration {
                time::sleep(chunk_duration - elapsed).await;
            }
        }
        
        info!("Finished streaming audio chunks");
    });

    // Create stream and start transcription
    let stream = ReceiverStream::new(rx);
    let response = client.stream_transcribe(stream).await?;
    let mut result_stream = response.into_inner();

    // Process transcription results
    let mut last_text = String::new();
    let mut current_line = String::new();
    
    while let Some(result) = result_stream.next().await {
        match result {
            Ok(transcription) => {
                let elapsed = start_time.elapsed().as_secs_f32();
                
                // Clear previous line if text has changed significantly
                if !transcription.text.is_empty() && transcription.text != last_text {
                    if show_timestamps {
                        // Show with timestamps
                        println!("[{:06.2}s] {}", 
                                elapsed,
                                transcription.text);
                    } else {
                        // Update current line for continuous display
                        if transcription.is_final {
                            // Final transcription for this segment
                            println!("{}", transcription.text);
                            current_line.clear();
                        } else {
                            // Interim result - update in place
                            print!("\r{:<80}", transcription.text);
                            use std::io::{self, Write};
                            io::stdout().flush()?;
                            current_line = transcription.text.clone();
                        }
                    }
                    
                    last_text = transcription.text.clone();
                }
            }
            Err(e) => {
                eprintln!("\nTranscription error: {}", e);
                break;
            }
        }
    }
    
    // Clear any remaining interim text
    if !current_line.is_empty() {
        println!();
    }
    
    println!("\n{}", "─".repeat(80));
    println!("✅ Playback and transcription complete!");
    
    // Keep the program alive until playback finishes
    time::sleep(Duration::from_secs(2)).await;

    Ok(())
}