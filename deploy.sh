#!/bin/bash
# SpectrumAlert Docker Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker service:"
        echo "sudo systemctl start docker"
        exit 1
    fi
    
    # Check for docker-compose or docker compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Function to check USB permissions
check_usb_permissions() {
    print_status "Checking USB permissions for RTL-SDR..."
    
    if groups | grep -q plugdev; then
        print_success "User is in plugdev group"
    else
        print_warning "User is not in plugdev group. You may need to run:"
        echo "sudo usermod -a -G plugdev \$USER"
        echo "Then log out and log back in."
    fi
    
    # Check if RTL-SDR device is connected
    if lsusb | grep -i "realtek\|rtl"; then
        print_success "RTL-SDR device detected"
    else
        print_warning "RTL-SDR device not detected. Make sure it's connected."
    fi
}

# Function to create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p data models logs config
    
    # Set proper permissions
    chmod 755 data models logs config
    
    print_success "Directories created"
}

# Function to build Docker image
build_image() {
    print_status "Building SpectrumAlert Docker image..."
    
    docker build -t spectrumalert:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "SpectrumAlert Docker Deployment"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  interactive Run in interactive mode"
    echo "  service     Run in service mode (continuous monitoring)"
    echo "  stop        Stop all containers"
    echo "  logs        Show container logs"
    echo "  status      Show container status"
    echo "  clean       Remove containers and images"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                 # Build the image"
    echo "  $0 interactive           # Run interactive mode"
    echo "  $0 service               # Run service mode"
    echo "  $0 logs spectrum-alert   # Show logs for interactive container"
    echo ""
}

# Main script
case "${1:-help}" in
    "build")
        check_docker
        setup_directories
        build_image
        ;;
    
    "interactive")
        check_docker
        check_usb_permissions
        print_status "Starting SpectrumAlert in interactive mode..."
        
        # Try docker run first as fallback for compose issues
        if ! $COMPOSE_CMD ps &> /dev/null; then
            print_warning "Docker Compose having issues, trying direct docker run..."
            docker run -it --rm \
                --device=/dev/bus/usb:/dev/bus/usb \
                --privileged \
                -v "$(pwd)/data:/app/data" \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/logs:/app/logs" \
                -v "$(pwd)/config:/app/config" \
                -v /dev:/dev \
                --network host \
                spectrumalert:latest
        else
            $COMPOSE_CMD up spectrumalert
        fi
        ;;
    
    "service")
        check_docker
        check_usb_permissions
        print_status "Starting SpectrumAlert in service mode..."
        
        # Check if models exist
        if [ ! "$(ls -A models/)" ]; then
            print_warning "No models found in models/ directory."
            print_warning "You may need to train models first in interactive mode."
        fi
        
        if ! $COMPOSE_CMD ps &> /dev/null; then
            print_warning "Docker Compose having issues, trying direct docker run..."
            docker run -d --name spectrum-alert-service \
                --restart unless-stopped \
                --device=/dev/bus/usb:/dev/bus/usb \
                --privileged \
                -e PYTHONUNBUFFERED=1 \
                -e SPECTRUM_MODE=service \
                -e LOG_LEVEL=INFO \
                -e SERVICE_MODEL=latest \
                -e SERVICE_LITE_MODE=false \
                -e SERVICE_ALERT_THRESHOLD=0.7 \
                -v "$(pwd)/data:/app/data" \
                -v "$(pwd)/models:/app/models" \
                -v "$(pwd)/logs:/app/logs" \
                -v "$(pwd)/config:/app/config" \
                -v /dev:/dev \
                --network host \
                spectrumalert:latest python service_mode.py
        else
            $COMPOSE_CMD --profile service up -d spectrumalert-service
        fi
        print_success "Service started in background"
        print_status "Use '$0 logs spectrum-alert-service' to view logs"
        print_status "Use '$0 stop' to stop the service"
        ;;
    
    "stop")
        print_status "Stopping all SpectrumAlert containers..."
        if command -v $COMPOSE_CMD &> /dev/null && $COMPOSE_CMD ps &> /dev/null; then
            $COMPOSE_CMD down
        fi
        # Also stop direct docker containers
        docker stop spectrum-alert spectrum-alert-service 2>/dev/null || true
        docker rm spectrum-alert spectrum-alert-service 2>/dev/null || true
        print_success "All containers stopped"
        ;;
    
    "logs")
        container_name="${2:-spectrum-alert}"
        print_status "Showing logs for $container_name..."
        docker logs -f "$container_name"
        ;;
    
    "status")
        print_status "Container status:"
        if command -v $COMPOSE_CMD &> /dev/null && $COMPOSE_CMD ps &> /dev/null; then
            $COMPOSE_CMD ps
        else
            docker ps -a --filter "name=spectrum-alert"
        fi
        
        print_status "\nService status (if running):"
        if [ -f "logs/service_status.json" ]; then
            cat logs/service_status.json | python -m json.tool
        else
            echo "No service status file found"
        fi
        ;;
    
    "clean")
        print_status "Cleaning up containers and images..."
        if command -v $COMPOSE_CMD &> /dev/null; then
            $COMPOSE_CMD down --rmi all --volumes --remove-orphans 2>/dev/null || true
        fi
        # Clean up direct docker containers
        docker stop spectrum-alert spectrum-alert-service 2>/dev/null || true
        docker rm spectrum-alert spectrum-alert-service 2>/dev/null || true
        docker rmi spectrumalert:latest 2>/dev/null || true
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    
    "shell")
        container_name="${2:-spectrum-alert}"
        print_status "Opening shell in $container_name..."
        docker exec -it "$container_name" /bin/bash
        ;;
    
    "help"|*)
        show_usage
        ;;
esac
