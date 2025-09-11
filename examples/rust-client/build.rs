fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile protobuf files
    tonic_build::configure()
        .build_server(false)  // We only need the client
        .compile(
            &["../../proto/transcription.proto"],
            &["../../proto"],
        )?;
    Ok(())
}