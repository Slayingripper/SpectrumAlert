# SpectrumAlert Docker Image
FROM python:3.10-slim

# Install system dependencies for RTL-SDR
RUN apt-get update && apt-get install -y \
    librtlsdr-dev \
    rtl-sdr \
    libusb-1.0-0-dev \
    pkg-config \
    build-essential \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1000 spectrum && \
    usermod -a -G plugdev spectrum

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/config && \
    chown -R spectrum:spectrum /app

# Copy RTL-SDR udev rules for device access
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666", SYMLINK+="rtl_sdr"' > /etc/udev/rules.d/20-rtlsdr.rules

# Switch to non-root user
USER spectrum

# Expose any ports if needed (for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from src.core.robust_collector import SafeRTLSDR; sdr = SafeRTLSDR(); print('healthy' if sdr.open() and sdr.close() else 'unhealthy')" || exit 1

# Default command (interactive mode)
CMD ["python", "main.py"]
