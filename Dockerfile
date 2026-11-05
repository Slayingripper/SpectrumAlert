# SpectrumAlert Docker Image
FROM python:3.11-slim

# Install system dependencies for RTL-SDR
RUN apt-get update && apt-get install -y \
    librtlsdr-dev \
    rtl-sdr \
    libusb-1.0-0-dev \
    pkg-config \
    build-essential \
    udev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1000 spectrum && \
    usermod -a -G plugdev spectrum

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install the package with dependencies
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/config && \
    chown -R spectrum:spectrum /app

# Copy RTL-SDR udev rules for device access
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666", SYMLINK+="rtl_sdr"' > /etc/udev/rules.d/20-rtlsdr.rules

# Switch to non-root user
USER spectrum

# The dashboard's default port (`spectrum-alert web`), not 8080.
EXPOSE 8000

# Health check: simple CLI invocation that does not require SDR
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD spectrum-alert version || exit 1

# Default entrypoint and command: serve the dashboard. Override CMD for any
# other CLI subcommand, e.g. `docker run ... spectrumalert monitor status`.
ENTRYPOINT ["spectrum-alert"]
CMD ["web", "--host", "0.0.0.0", "--port", "8000"]
