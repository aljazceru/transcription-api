# Makefile for Transcription API Service

.PHONY: help build run stop clean test proto rust-client

# Variables
DOCKER_IMAGE = transcription-api
DOCKER_TAG = latest
GRPC_PORT = 50051
WEBSOCKET_PORT = 8765
MODEL = base

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

run: ## Run service with docker-compose
	MODEL_PATH=$(MODEL) docker compose up -d
	@echo "Service started!"
	@echo "gRPC endpoint: localhost:$(GRPC_PORT)"
	@echo "WebSocket endpoint: ws://localhost:$(WEBSOCKET_PORT)"

run-gpu: ## Run service with GPU support
	MODEL_PATH=large-v3 CUDA_VISIBLE_DEVICES=0 docker compose up -d
	@echo "Service started with GPU support!"

stop: ## Stop the service
	docker compose down

logs: ## Show service logs
	docker compose logs -f

clean: ## Clean up containers and volumes
	docker compose down -v
	docker system prune -f

proto: ## Generate protobuf code
	python -m grpc_tools.protoc \
		-I./proto \
		--python_out=./src \
		--grpc_python_out=./src \
		./proto/transcription.proto
	@echo "Generated Python protobuf code in src/"

rust-client: ## Build Rust client examples
	cd examples/rust-client && cargo build --release
	@echo "Rust clients built in examples/rust-client/target/release/"

test-grpc: ## Test gRPC connection
	@command -v grpcurl >/dev/null 2>&1 || { echo "grpcurl not installed. Install from https://github.com/fullstorydev/grpcurl"; exit 1; }
	grpcurl -plaintext localhost:$(GRPC_PORT) list
	grpcurl -plaintext localhost:$(GRPC_PORT) transcription.TranscriptionService/HealthCheck

test-websocket: ## Test WebSocket connection
	@echo "Testing WebSocket connection..."
	@python3 -c "import asyncio, websockets, json; \
		async def test(): \
			async with websockets.connect('ws://localhost:$(WEBSOCKET_PORT)') as ws: \
				data = await ws.recv(); \
				print('Connected:', json.loads(data)); \
		asyncio.run(test())"

install-deps: ## Install Python dependencies
	pip install -r requirements.txt

docker-push: ## Push Docker image to registry
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# Model management
download-models: ## Download Whisper models
	@echo "Downloading Whisper models..."
	python -c "import whisper; \
		for model in ['tiny', 'base', 'small']: \
			print(f'Downloading {model}...'); \
			whisper.load_model(model)"

# Development
dev-run: ## Run service locally (without Docker)
	cd src && python transcription_server.py

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install black flake8 pytest pytest-asyncio

format: ## Format Python code
	black src/

lint: ## Lint Python code
	flake8 src/

# Benchmarking
benchmark: ## Run performance benchmark
	@echo "Running transcription benchmark..."
	time curl -X POST \
		-H "Content-Type: application/octet-stream" \
		--data-binary @test_audio.wav \
		http://localhost:$(GRPC_PORT)/benchmark