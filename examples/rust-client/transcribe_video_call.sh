#!/bin/bash

# Enhanced script for transcribing video calls on Ubuntu with PipeWire
# Uses parec (PulseAudio compatibility) to capture system audio

set -e

echo "🎥 Video Call Transcription Service"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED} $1 not found.${NC}"
        echo "Please install: sudo apt-get install $2"
        return 1
    fi
    return 0
}

echo "Checking dependencies..."
check_dependency "parec" "pulseaudio-utils" || exit 1
check_dependency "sox" "sox" || echo -e "${YELLOW}  sox not installed (optional but recommended)${NC}"

# Function to find the monitor source for system audio
find_monitor_source() {
    # List all sources and find monitors (what you hear)
    local monitors=$(pactl list sources short 2>/dev/null | grep -i "monitor" | awk '{print $2}')
    
    if [ -z "$monitors" ]; then
        # Try pacmd if pactl doesn't work
        monitors=$(pacmd list-sources 2>/dev/null | grep "name:" | grep "monitor" | sed 's/.*<\(.*\)>.*/\1/')
    fi
    
    if [ -z "$monitors" ]; then
        # Fallback: try to construct monitor name from default sink
        local default_sink=$(pactl info 2>/dev/null | grep "Default Sink" | cut -d: -f2 | xargs)
        if [ -n "$default_sink" ]; then
            monitors="${default_sink}.monitor"
        fi
    fi
    
    echo "$monitors" | head -1
}

# List available sources
if [ "$1" == "--list" ]; then
    echo -e "${GREEN} Available Audio Sources:${NC}"
    echo ""
    pactl list sources short 2>/dev/null || pacmd list-sources 2>/dev/null | grep "name:"
    echo ""
    echo -e "${GREEN} Monitor sources (system audio):${NC}"
    pactl list sources short 2>/dev/null | grep -i "monitor" || echo "No monitor sources found"
    exit 0
fi

# Help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --list             List all available audio sources"
    echo "  --source SOURCE    Use specific audio source"
    echo "  --microphone       Capture microphone instead of system audio"
    echo "  --combined         Capture both microphone and system audio"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Auto-detect and transcribe system audio"
    echo "  $0 --microphone    # Transcribe from microphone"
    echo "  $0 --combined      # Transcribe both mic and system audio"
    echo ""
    exit 0
fi

# Determine what to capture
if [ "$1" == "--microphone" ]; then
    echo -e "${GREEN} Using microphone input${NC}"
    # Run the existing live-transcribe for microphone
    exec cargo run --bin live-transcribe
    exit 0
elif [ "$1" == "--combined" ]; then
    echo -e "${YELLOW}+ Combined audio capture not yet implemented${NC}"
    echo "For now, please run two separate instances:"
    echo "  1. $0 (for system audio)"
    echo "  2. $0 --microphone (for mic)"
    exit 1
elif [ "$1" == "--source" ] && [ -n "$2" ]; then
    SOURCE="$2"
    echo -e "${GREEN} Using specified source: $SOURCE${NC}"
else
    # Auto-detect monitor source
    SOURCE=$(find_monitor_source)
    if [ -z "$SOURCE" ]; then
        echo -e "${RED} Could not find system audio monitor source${NC}"
        echo ""
        echo "This might happen if:"
        echo "  1. No audio is currently playing"
        echo "  2. PipeWire/PulseAudio is not running"
        echo ""
        echo "Try:"
        echo "  1. Play some audio (music/video)"
        echo "  2. Run: $0 --list"
        echo "  3. Use a specific source: $0 --source <source_name>"
        exit 1
    fi
    echo -e "${GREEN} Found system audio source: $SOURCE${NC}"
fi

echo ""
echo -e "${GREEN} Starting video call transcription...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""
echo " Tips for best results:"
echo "  • Join your video call first"
echo "  • Use headphones to avoid echo"
echo "  • Close other audio sources (music, videos)"
echo "  • Speak clearly for better transcription"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Start audio capture and transcription
echo -e "${GREEN}Starting audio capture from: $SOURCE${NC}"
echo -e "${GREEN}Starting transcription service...${NC}"
echo ""

# Use our new stdin-transcribe binary that accepts piped audio
# parec captures system audio and pipes it directly to our transcriber
# --no-vad disables Voice Activity Detection for system audio (YouTube, music, etc.)
parec --format=s16le --rate=16000 --channels=1 --device="$SOURCE" 2>/dev/null | \
cargo run --bin stdin-transcribe -- --language en --chunk-seconds 2.5 --no-vad
