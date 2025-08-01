#!/bin/bash
# SpectrumAlert Simple Docker Runner (for Raspberry Pi compatibility)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Docker image name
IMAGE_NAME="spectrumalert:latest"

# Common docker run arguments
DOCKER_ARGS="--device=/dev/bus/usb:/dev/bus/usb \
    --privileged \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/config:/app/config \
    -v /dev:/dev \
    --network host"

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Try: sudo systemctl start docker"
        exit 1
    fi
    
    print_success "Docker is available"
}

check_image() {
    if ! docker image inspect $IMAGE_NAME &> /dev/null; then
        print_error "Docker image '$IMAGE_NAME' not found. Run: $0 build"
        exit 1
    fi
}

build_image() {
    print_status "Building SpectrumAlert Docker image..."
    docker build -t $IMAGE_NAME .
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

setup_dirs() {
    mkdir -p data models logs config
    chmod 755 data models logs config
}

case "${1:-help}" in
    "build")
        check_docker
        setup_dirs
        build_image
        ;;
    
    "interactive"|"run")
        check_docker
        check_image
        print_status "Starting SpectrumAlert in interactive mode..."
        docker run -it --rm --name spectrum-alert $DOCKER_ARGS $IMAGE_NAME
        ;;
    
    "service")
        check_docker
        check_image
        
        if [ ! "$(ls -A models/ 2>/dev/null)" ]; then
            print_warning "No models found. Train models first with: $0 interactive"
        fi
        
        print_status "Starting SpectrumAlert service..."
        docker run -d --name spectrum-alert-service --restart unless-stopped \
            -e PYTHONUNBUFFERED=1 \
            -e SPECTRUM_MODE=service \
            -e LOG_LEVEL=INFO \
            -e SERVICE_MODEL=latest \
            -e SERVICE_LITE_MODE=false \
            -e SERVICE_ALERT_THRESHOLD=0.7 \
            $DOCKER_ARGS $IMAGE_NAME python service_mode.py
        
        print_success "Service started in background"
        print_status "View logs: $0 logs"
        print_status "Stop service: $0 stop"
        ;;
    
    "autonomous")
        check_docker
        check_image
        
        print_status "Starting SpectrumAlert autonomous mode..."
        print_status "This will:"
        print_status "  1. Collect RF data for 24 hours"
        print_status "  2. Automatically train ML models"
        print_status "  3. Start continuous anomaly monitoring"
        print_status "  4. Periodically retrain models (weekly)"
        
        docker run -d --name spectrum-alert-autonomous --restart unless-stopped \
            -e PYTHONUNBUFFERED=1 \
            -e SPECTRUM_MODE=autonomous \
            -e LOG_LEVEL=INFO \
            -e COLLECTION_HOURS=24 \
            -e SERVICE_LITE_MODE=false \
            -e SERVICE_ALERT_THRESHOLD=0.7 \
            -e RETRAIN_INTERVAL_HOURS=168 \
            -e MIN_TRAINING_SAMPLES=1000 \
            $DOCKER_ARGS $IMAGE_NAME python autonomous_service.py
        
        print_success "Autonomous service started"
        print_status "View logs: $0 logs spectrum-alert-autonomous"
        print_status "Check status: $0 status"
        print_status "Stop service: $0 stop"
        ;;
    
    "stop")
        print_status "Stopping containers..."
        docker stop spectrum-alert spectrum-alert-service spectrum-alert-autonomous 2>/dev/null || true
        docker rm spectrum-alert spectrum-alert-service spectrum-alert-autonomous 2>/dev/null || true
        print_success "Containers stopped"
        ;;
    
    "logs")
        container="${2:-spectrum-alert-service}"
        print_status "Showing logs for $container..."
        docker logs -f "$container"
        ;;
    
    "status")
        print_status "Container status:"
        docker ps -a --filter "name=spectrum-alert"
        
        if [ -f "logs/service_status.json" ]; then
            print_status "\nService status:"
            cat logs/service_status.json | python3 -m json.tool 2>/dev/null || cat logs/service_status.json
        fi
        
        if [ -f "logs/autonomous_status.json" ]; then
            print_status "\nAutonomous service status:"
            cat logs/autonomous_status.json | python3 -m json.tool 2>/dev/null || cat logs/autonomous_status.json
        fi
        ;;
    
    "shell")
        container="${2:-spectrum-alert-service}"
        if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
            docker exec -it "$container" /bin/bash
        else
            print_error "Container '$container' is not running"
            exit 1
        fi
        ;;
    
    "clean")
        print_status "Cleaning up..."
        docker stop spectrum-alert spectrum-alert-service spectrum-alert-autonomous 2>/dev/null || true
        docker rm spectrum-alert spectrum-alert-service spectrum-alert-autonomous 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    
    "test")
        check_docker
        check_image
        print_status "Testing RTL-SDR access in container..."
        docker run --rm $DOCKER_ARGS $IMAGE_NAME rtl_test -t
        ;;
    
    *)
        echo "SpectrumAlert Simple Docker Runner"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  build       Build Docker image"
        echo "  interactive Run interactive mode (alias: run)"
        echo "  autonomous  Run autonomous mode (24h collection + training + monitoring)"
        echo "  service     Run service mode in background"
        echo "  stop        Stop all containers"
        echo "  logs [name] Show container logs"
        echo "  status      Show container and service status"
        echo "  shell [name] Open shell in container"
        echo "  test        Test RTL-SDR access"
        echo "  clean       Remove containers and image"
        echo ""
        echo "Examples:"
        echo "  $0 build                # Build image"
        echo "  $0 interactive          # Interactive mode"
        echo "  $0 autonomous           # Autonomous mode (24h + training)"
        echo "  $0 service              # Background service"
        echo "  $0 logs                 # Show service logs"
        echo ""
        ;;
esac
