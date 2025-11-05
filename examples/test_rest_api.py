#!/usr/bin/env python3
"""
Test script for the REST API
Demonstrates basic usage of the transcription REST API
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Model: {data['model_loaded']}")
        print(f"Uptime: {data['uptime_seconds']}s")
        print(f"Active Sessions: {data['active_sessions']}")
        print("✓ Health check passed\n")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}\n")
        return False


def test_capabilities():
    """Test capabilities endpoint"""
    print("=" * 60)
    print("Testing Capabilities Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/capabilities", timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()
        print(f"Available Models: {', '.join(data['available_models'])}")
        print(f"Supported Languages: {', '.join(data['supported_languages'][:5])}...")
        print(f"Supported Formats: {', '.join(data['supported_formats'])}")
        print(f"Max Audio Length: {data['max_audio_length_seconds']}s")
        print(f"Streaming Supported: {data['streaming_supported']}")
        print("✓ Capabilities check passed\n")
        return True
    except Exception as e:
        print(f"✗ Capabilities check failed: {e}\n")
        return False


def test_transcribe_file(audio_file: str):
    """Test file transcription endpoint"""
    print("=" * 60)
    print("Testing File Transcription")
    print("=" * 60)

    if not Path(audio_file).exists():
        print(f"✗ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
        print("Example: python test_rest_api.py audio.wav\n")
        return False

    try:
        print(f"Uploading: {audio_file}")

        with open(audio_file, 'rb') as f:
            files = {'file': (Path(audio_file).name, f, 'audio/wav')}
            data = {
                'language': 'auto',
                'task': 'transcribe',
                'vad_enabled': True
            }

            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/transcribe",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            elapsed = time.time() - start_time

        result = response.json()

        print(f"\nTranscription Results:")
        print(f"  Language: {result['detected_language']}")
        print(f"  Duration: {result['duration_seconds']:.2f}s")
        print(f"  Processing Time: {result['processing_time']:.2f}s")
        print(f"  Segments: {len(result['segments'])}")
        print(f"  Request Time: {elapsed:.2f}s")
        print(f"\nFull Text:")
        print(f"  {result['full_text']}")

        if result['segments']:
            print(f"\nFirst Segment:")
            seg = result['segments'][0]
            print(f"  [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]")
            print(f"  Text: {seg['text']}")
            print(f"  Confidence: {seg['confidence']:.2f}")

        print("\n✓ Transcription test passed\n")
        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Transcription test failed: {e}\n")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}\n")
        return False


def test_root():
    """Test root endpoint"""
    print("=" * 60)
    print("Testing Root Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        response.raise_for_status()

        data = response.json()
        print(f"Service: {data['service']}")
        print(f"Version: {data['version']}")
        print(f"Status: {data['status']}")
        print(f"Endpoints: {', '.join(data['endpoints'].keys())}")
        print("✓ Root endpoint test passed\n")
        return True
    except Exception as e:
        print(f"✗ Root endpoint test failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("REST API Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("=" * 60 + "\n")

    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/", timeout=5)
    except Exception as e:
        print("✗ Cannot connect to server")
        print(f"Error: {e}")
        print("\nMake sure the server is running:")
        print("  docker compose up -d")
        print("  # or")
        print("  python src/transcription_server.py")
        sys.exit(1)

    # Run tests
    results = []

    results.append(("Root Endpoint", test_root()))
    results.append(("Health Check", test_health()))
    results.append(("Capabilities", test_capabilities()))

    # Test transcription if audio file is provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        results.append(("File Transcription", test_transcribe_file(audio_file)))
    else:
        print("=" * 60)
        print("Skipping File Transcription Test")
        print("=" * 60)
        print("To test transcription, provide an audio file:")
        print(f"  python {sys.argv[0]} audio.wav\n")

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
