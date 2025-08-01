#!/bin/bash
# SpectrumAlert Docker Troubleshooting Script for Raspberry Pi

echo "SpectrumAlert Docker Troubleshooting"
echo "===================================="
echo ""

# Check system info
echo "1. System Information:"
echo "   OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "   Kernel: $(uname -r)"
echo "   Architecture: $(uname -m)"
echo ""

# Check Docker installation
echo "2. Docker Installation:"
if command -v docker &> /dev/null; then
    echo "   ✓ Docker installed: $(docker --version)"
    
    if systemctl is-active --quiet docker; then
        echo "   ✓ Docker service is running"
    else
        echo "   ✗ Docker service is not running"
        echo "     Try: sudo systemctl start docker"
    fi
    
    if docker info &> /dev/null; then
        echo "   ✓ Docker daemon accessible"
    else
        echo "   ✗ Cannot access Docker daemon"
        echo "     Try: sudo usermod -aG docker $USER (then log out/in)"
    fi
else
    echo "   ✗ Docker not installed"
    echo "     Install with: curl -fsSL https://get.docker.com | sh"
fi
echo ""

# Check Docker Compose
echo "3. Docker Compose:"
if command -v docker-compose &> /dev/null; then
    echo "   ✓ docker-compose installed: $(docker-compose --version)"
    
    # Test docker-compose
    if cd /tmp && echo "version: '3'" > test-compose.yml && docker-compose -f test-compose.yml config &> /dev/null; then
        echo "   ✓ docker-compose working"
    else
        echo "   ✗ docker-compose has issues"
        echo "     This is common on Raspberry Pi. Use run_docker.sh instead"
    fi
    rm -f /tmp/test-compose.yml
elif docker compose version &> /dev/null; then
    echo "   ✓ docker compose (plugin) installed: $(docker compose version)"
else
    echo "   ⚠ Docker Compose not found (this is OK, run_docker.sh works without it)"
fi
echo ""

# Check RTL-SDR
echo "4. RTL-SDR Hardware:"
if lsusb | grep -i "realtek\|rtl" &> /dev/null; then
    echo "   ✓ RTL-SDR device detected:"
    lsusb | grep -i "realtek\|rtl" | sed 's/^/     /'
else
    echo "   ✗ RTL-SDR device not detected"
    echo "     Check USB connection and try: lsusb"
fi

if groups | grep -q plugdev; then
    echo "   ✓ User in plugdev group"
else
    echo "   ✗ User not in plugdev group"
    echo "     Fix with: sudo usermod -aG plugdev $USER (then log out/in)"
fi

if command -v rtl_test &> /dev/null; then
    echo "   ✓ RTL-SDR tools installed"
else
    echo "   ⚠ RTL-SDR tools not installed (this is OK, Docker includes them)"
fi
echo ""

# Check USB permissions
echo "5. USB Permissions:"
if [ -e /dev/bus/usb ]; then
    echo "   ✓ USB bus accessible: /dev/bus/usb"
    usb_devices=$(find /dev/bus/usb -name "*" -type c 2>/dev/null | wc -l)
    echo "   ✓ USB devices found: $usb_devices"
else
    echo "   ✗ USB bus not accessible"
fi
echo ""

# Check disk space
echo "6. Disk Space:"
df_output=$(df -h . | tail -1)
echo "   Current directory: $df_output"
available=$(echo $df_output | awk '{print $4}' | sed 's/[^0-9]//g')
if [ "$available" -gt 1000 ]; then
    echo "   ✓ Sufficient disk space"
else
    echo "   ⚠ Low disk space (may cause Docker issues)"
fi
echo ""

# Check memory
echo "7. Memory:"
mem_info=$(free -h | grep "Mem:")
echo "   $mem_info"
mem_available=$(free -m | grep "Mem:" | awk '{print $7}')
if [ "$mem_available" -gt 500 ]; then
    echo "   ✓ Sufficient memory available"
else
    echo "   ⚠ Low memory (may affect performance)"
fi
echo ""

# Check if image exists
echo "8. SpectrumAlert Docker Image:"
if docker image inspect spectrumalert:latest &> /dev/null; then
    echo "   ✓ SpectrumAlert image exists"
    image_size=$(docker image inspect spectrumalert:latest --format '{{.Size}}' | numfmt --to=iec)
    echo "   ✓ Image size: $image_size"
else
    echo "   ✗ SpectrumAlert image not built"
    echo "     Build with: ./run_docker.sh build"
fi
echo ""

# Recommendations
echo "9. Recommendations:"
echo ""

if ! command -v docker &> /dev/null; then
    echo "   → Install Docker: curl -fsSL https://get.docker.com | sh"
fi

if ! groups | grep -q docker; then
    echo "   → Add user to docker group: sudo usermod -aG docker $USER"
fi

if ! groups | grep -q plugdev; then
    echo "   → Add user to plugdev group: sudo usermod -aG plugdev $USER"
fi

if ! systemctl is-active --quiet docker; then
    echo "   → Start Docker service: sudo systemctl start docker"
    echo "   → Enable Docker on boot: sudo systemctl enable docker"
fi

if ! docker image inspect spectrumalert:latest &> /dev/null; then
    echo "   → Build SpectrumAlert image: ./run_docker.sh build"
fi

if command -v docker-compose &> /dev/null; then
    if ! cd /tmp && echo "version: '3'" > test-compose.yml && docker-compose -f test-compose.yml config &> /dev/null; then
        echo "   → Docker Compose has issues, use run_docker.sh instead of deploy.sh"
        rm -f /tmp/test-compose.yml
    fi
fi

echo ""
echo "Quick Start (if Docker is working):"
echo "  1. ./run_docker.sh build"
echo "  2. ./run_docker.sh interactive"
echo "  3. (Train models in interactive mode)"
echo "  4. ./run_docker.sh service"
echo ""
