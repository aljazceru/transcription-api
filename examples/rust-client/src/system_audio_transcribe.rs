use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

// Import generated protobuf types
pub mod transcription {
    tonic::include_proto!("transcription");
}

use transcription::{
    transcription_service_client::TranscriptionServiceClient, AudioChunk, AudioConfig,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Capture and transcribe system audio", long_about = None)]
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

    /// List available audio devices
    #[arg(long)]
    list_devices: bool,

    /// Audio device name or index to use (e.g., "pulse.monitor" for PulseAudio monitor)
    #[arg(short, long)]
    device: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // List devices if requested
    if args.list_devices {
        list_audio_devices()?;
        return Ok(());
    }

    info!("Connecting to transcription service at {}", args.server);
    let mut client = TranscriptionServiceClient::connect(args.server.clone()).await?;

    // Create channel for audio chunks
    let (tx, rx) = tokio_mpsc::channel::<AudioChunk>(100);

    // Start audio capture in a separate thread
    let device_name = args.device.clone();
    std::thread::spawn(move || {
        if let Err(e) = capture_system_audio(tx, device_name) {
            error!("Audio capture error: {}", e);
        }
    });

    // Create the first chunk with configuration
    let config = AudioConfig {
        language: args.language.clone(),
        task: args.task.clone(),
        model: args.model.clone(),
        sample_rate: 16000,
        vad_enabled: true,  // Enable VAD to filter silence
    };

    // Send a configuration chunk first
    let config_chunk = AudioChunk {
        audio_data: vec![],
        session_id: "system-audio".to_string(),
        config: Some(config),
    };

    // Create stream from receiver
    let stream_vec = vec![config_chunk];
    let stream = ReceiverStream::new(rx);
    let stream = futures_util::stream::iter(stream_vec).chain(stream);

    // Start streaming transcription
    let request = tonic::Request::new(stream);
    let mut response = client.stream_transcribe(request).await?.into_inner();

    println!("\n🎧 Capturing system audio for transcription...");
    println!("Press Ctrl+C to stop\n");
    println!("{}", "─".repeat(80));

    // Process transcription responses
    while let Some(result) = response.message().await? {
        if !result.text.is_empty() {
            if result.is_final {
                println!("[FINAL] {}", result.text);
            } else {
                print!("\r[INTERIM] {:<80}", result.text);
                use std::io::{self, Write};
                io::stdout().flush()?;
            }
        }
    }

    Ok(())
}

/// List all available audio devices
fn list_audio_devices() -> Result<()> {
    let host = cpal::default_host();
    
    println!("\n📊 Available Audio Devices:");
    println!("{}", "─".repeat(80));
    
    // List input devices
    println!("\n🎤 Input Devices:");
    for (idx, device) in host.input_devices()?.enumerate() {
        let name = device.name()?;
        let is_monitor = name.contains("monitor") || name.contains("Monitor") || 
                        name.contains("loopback") || name.contains("Loopback") ||
                        name.contains("stereo mix") || name.contains("Stereo Mix");
        
        if is_monitor {
            println!("  [{}] {} 🔊 (System Audio)", idx, name);
        } else {
            println!("  [{}] {}", idx, name);
        }
    }
    
    // Show default device
    if let Some(device) = host.default_input_device() {
        println!("\n⭐ Default Input: {}", device.name()?);
    }
    
    println!("\n💡 Tips for capturing system audio:");
    println!("  Linux: Look for devices with 'monitor' in the name (PulseAudio/PipeWire)");
    println!("  Windows: Install VB-Cable or enable 'Stereo Mix' in sound settings");
    println!("  macOS: Install BlackHole or Loopback for system audio capture");
    
    Ok(())
}

/// Capture audio from system (or specified device)
fn capture_system_audio(tx: tokio_mpsc::Sender<AudioChunk>, device_name: Option<String>) -> Result<()> {
    let host = cpal::default_host();
    
    // Find the appropriate audio device
    let device = if let Some(name) = device_name {
        // Try to find device by name
        let mut found_device = None;
        for input_device in host.input_devices()? {
            if input_device.name()?.contains(&name) {
                found_device = Some(input_device);
                break;
            }
        }
        found_device.ok_or_else(|| anyhow::anyhow!("Device '{}' not found. Use --list-devices to see available devices.", name))?
    } else {
        // Try to find a monitor/loopback device automatically
        let mut monitor_device = None;
        for input_device in host.input_devices()? {
            let name = input_device.name()?;
            if name.contains("monitor") || name.contains("Monitor") || 
               name.contains("loopback") || name.contains("Loopback") ||
               name.contains("stereo mix") || name.contains("Stereo Mix") {
                info!("Found system audio device: {}", name);
                monitor_device = Some(input_device);
                break;
            }
        }
        
        if let Some(device) = monitor_device {
            device
        } else {
            warn!("No system audio device found. Using default input device.");
            warn!("To capture system audio, you may need to:");
            warn!("  - Linux: Enable PulseAudio monitor");
            warn!("  - Windows: Enable Stereo Mix or install VB-Cable");
            warn!("  - macOS: Install BlackHole or Loopback");
            host.default_input_device()
                .ok_or_else(|| anyhow::anyhow!("No input device available"))?
        }
    };
    
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

                let audio_chunk = AudioChunk {
                    audio_data: bytes,
                    session_id: String::new(),
                    config: None,
                };

                // Send chunk (ignore errors if receiver is closed)
                let tx_clone = tx.clone();
                tokio::spawn(async move {
                    let _ = tx_clone.send(audio_chunk).await;
                });
            }
        },
        move |err| {
            error!("Audio stream error: {}", err);
        },
        None
    )?;

    // Start the stream
    stream.play()?;
    info!("Audio capture started");

    // Keep the stream alive
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}