#!/usr/bin/env python3
"""
Generate Python code from protobuf definitions
Run this before starting the service for the first time
"""

import os
import sys
import subprocess

def generate_proto():
    """Generate Python code from proto files"""
    proto_dir = "proto"
    output_dir = "src"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .proto files
    proto_files = [f for f in os.listdir(proto_dir) if f.endswith('.proto')]
    
    if not proto_files:
        print("No .proto files found in proto/ directory")
        return False
    
    for proto_file in proto_files:
        proto_path = os.path.join(proto_dir, proto_file)
        print(f"Generating code for {proto_file}...")
        
        # Generate Python code
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            proto_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Generated {proto_file.replace('.proto', '_pb2.py')} and {proto_file.replace('.proto', '_pb2_grpc.py')}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate code for {proto_file}")
            print(f"  Error: {e.stderr}")
            return False
    
    # Fix imports in generated files
    print("Fixing imports in generated files...")
    grpc_file = os.path.join(output_dir, "transcription_pb2_grpc.py")
    if os.path.exists(grpc_file):
        with open(grpc_file, 'r') as f:
            content = f.read()
        
        # Fix relative import
        content = content.replace(
            "import transcription_pb2",
            "from . import transcription_pb2"
        )
        
        with open(grpc_file, 'w') as f:
            f.write(content)
        
        print("✓ Fixed imports")
    
    print("\nProtobuf generation complete!")
    print(f"Generated files are in {output_dir}/")
    return True

if __name__ == "__main__":
    # Check if grpcio-tools is installed
    try:
        import grpc_tools
    except ImportError:
        print("Error: grpcio-tools not installed")
        print("Run: pip install grpcio-tools")
        sys.exit(1)
    
    if generate_proto():
        print("\nYou can now start the service with:")
        print("  python src/transcription_server.py")
        print("Or with Docker:")
        print("  docker compose up")
    else:
        sys.exit(1)