use anyhow::Result;
use clap::Parser;
use futures_util::StreamExt;
use std::io::{self, Read};
use tokio::sync::mpsc;
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
#[command(author, version, about = "Transcribe audio from stdin (for piping from parec)", long_about = None)]
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

    /// Show timestamps
    #[arg(short = 'T', long)]
    timestamps: bool,

    /// Chunk size in seconds (for buffering)
    #[arg(short, long, default_value = "3.0")]
    chunk_seconds: f32,

    /// Disable VAD (Voice Activity Detection) - useful for music/system audio
    #[arg(long)]
    no_vad: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    info!("Connecting to transcription service at {}", args.server);
    let mut client = TranscriptionServiceClient::connect(args.server.clone()).await?;

    // Create channel for audio chunks
    let (tx, rx) = mpsc::channel::<AudioChunk>(100);

    // Spawn task to read from stdin and send chunks
    let tx_clone = tx.clone();
    let chunk_seconds = args.chunk_seconds;
    std::thread::spawn(move || {
        if let Err(e) = read_stdin_and_send(tx_clone, chunk_seconds) {
            error!("Error reading stdin: {}", e);
        }
    });

    // Create the first chunk with configuration
    let config = AudioConfig {
        language: args.language.clone(),
        task: args.task.clone(),
        model: args.model.clone(),
        sample_rate: 16000,
        vad_enabled: !args.no_vad,  // Disable VAD if --no-vad flag is used
    };

    // Send a configuration chunk first
    let config_chunk = AudioChunk {
        audio_data: vec![],
        session_id: "stdin-transcribe".to_string(),
        config: Some(config),
    };

    // Create stream from receiver
    let stream = ReceiverStream::new(rx);
    let stream = futures_util::stream::iter(vec![config_chunk]).chain(stream);

    // Start streaming transcription
    let request = tonic::Request::new(stream);
    let mut response = client.stream_transcribe(request).await?.into_inner();

    println!("\n🎧 Transcribing audio from stdin...");
    println!("Press Ctrl+C to stop\n");
    println!("{}", "─".repeat(80));

    let mut current_line = String::new();

    // Process transcription responses
    while let Some(result) = response.message().await? {
        if !result.text.is_empty() {
            if args.timestamps {
                if result.is_final {
                    println!("[{:.1}s] {}", result.start_time, result.text);
                    current_line.clear();
                } else {
                    print!("\r[{:.1}s] {:<80}", result.start_time, result.text);
                    use std::io::{self as stdio, Write};
                    stdio::stdout().flush()?;
                    current_line = result.text.clone();
                }
            } else {
                if result.is_final {
                    println!("{}", result.text);
                    current_line.clear();
                } else {
                    print!("\r{:<80}", result.text);
                    use std::io::{self as stdio, Write};
                    stdio::stdout().flush()?;
                    current_line = result.text.clone();
                }
            }
        }
    }

    // Clear any remaining interim text
    if !current_line.is_empty() {
        println!();
    }

    Ok(())
}

fn read_stdin_and_send(tx: mpsc::Sender<AudioChunk>, chunk_seconds: f32) -> Result<()> {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    
    // Calculate chunk size in bytes (16kHz, 16-bit mono)
    let samples_per_chunk = (16000.0 * chunk_seconds) as usize;
    let bytes_per_chunk = samples_per_chunk * 2; // 16-bit = 2 bytes
    
    let mut buffer = vec![0u8; bytes_per_chunk];
    
    info!("Reading audio from stdin (chunk size: {} bytes, {} seconds)", 
          bytes_per_chunk, chunk_seconds);
    
    loop {
        // Read a chunk from stdin
        let mut total_read = 0;
        while total_read < bytes_per_chunk {
            match handle.read(&mut buffer[total_read..]) {
                Ok(0) => {
                    // EOF reached
                    if total_read > 0 {
                        // Send remaining data
                        let audio_chunk = AudioChunk {
                            audio_data: buffer[..total_read].to_vec(),
                            session_id: String::new(),
                            config: None,
                        };
                        let _ = tx.blocking_send(audio_chunk);
                    }
                    info!("End of stdin reached");
                    return Ok(());
                }
                Ok(n) => {
                    total_read += n;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                    // Retry on interrupt
                    continue;
                }
                Err(e) => {
                    error!("Error reading stdin: {}", e);
                    return Err(e.into());
                }
            }
        }
        
        // Send the chunk
        let audio_chunk = AudioChunk {
            audio_data: buffer.clone(),
            session_id: String::new(),
            config: None,
        };
        
        if tx.blocking_send(audio_chunk).is_err() {
            // Receiver dropped, exit
            break;
        }
    }
    
    Ok(())
}