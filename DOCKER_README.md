# SpectrumAlert Docker Deployment Guide

This guide explains how to deploy SpectrumAlert using Docker for easy deployment anywhere.

## Prerequisites

1. **Docker and Docker Compose**
   ```bash
   # Install Docker (Ubuntu/Debian)
   sudo apt update
   sudo apt install docker.io docker-compose
   
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and log back in
   ```

2. **RTL-SDR Device**
   - Connect your RTL-SDR USB device
   - Verify detection: `lsusb | grep -i realtek`

3. **USB Permissions**
   ```bash
   # Add user to plugdev group
   sudo usermod -aG plugdev $USER
   # Log out and log back in
   ```

## Quick Start

### For Raspberry Pi Users
If you encounter Docker Compose issues (common on Raspberry Pi), use the simplified script:
```bash
chmod +x run_docker.sh troubleshoot.sh
./troubleshoot.sh  # Check system compatibility
./run_docker.sh build
./run_docker.sh interactive
```

### For Standard Systems
### 1. Build the Docker Image
```bash
chmod +x deploy.sh
./deploy.sh build
```

### 2. Run Interactive Mode (Training & Setup)
```bash
./deploy.sh interactive
```
Use interactive mode to:
- Collect initial RF data
- Train ML models
- Test system functionality

### 3. Run Service Mode (Continuous Monitoring)
```bash
./deploy.sh service
```
Service mode runs continuously in the background, monitoring spectrum and alerting on anomalies.

## Deployment Modes

### Interactive Mode
Best for initial setup, training, and manual operations:
```bash
docker-compose up spectrumalert
```

Features:
- Full menu interface
- Data collection
- Model training
- Manual monitoring
- System diagnostics

### Service Mode
Best for production deployment with continuous monitoring:
```bash
docker-compose --profile service up -d spectrumalert-service
```

Features:
- Automatic model selection
- Continuous monitoring
- Auto-restart on failures
- Background operation
- Status logging

## Configuration

### Environment Variables
Copy and modify the environment file:
```bash
cp .env.example .env
# Edit .env with your settings
```

Key variables:
- `SERVICE_MODEL`: Model to use (`latest` or specific filename)
- `SERVICE_LITE_MODE`: Use lite mode for resource-constrained systems
- `SERVICE_ALERT_THRESHOLD`: Anomaly detection threshold (0.0-1.0)
- `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### Volume Mounts
Data is persisted in these directories:
- `./data/`: RF data files
- `./models/`: Trained ML models
- `./logs/`: Application logs
- `./config/`: Configuration files

## Management Commands

### Build and Deploy
```bash
./deploy.sh build        # Build Docker image
./deploy.sh interactive  # Run interactive mode
./deploy.sh service      # Run service mode
```

### Monitoring and Control
```bash
./deploy.sh status       # Show container status
./deploy.sh logs         # Show container logs
./deploy.sh stop         # Stop all containers
```

### Maintenance
```bash
./deploy.sh shell        # Open shell in container
./deploy.sh clean        # Remove all containers and images
```

## Alternative Scripts (Raspberry Pi Compatible)

For systems where Docker Compose has issues, use these alternative scripts:

### Simple Docker Runner
```bash
./run_docker.sh build        # Build image
./run_docker.sh interactive  # Interactive mode
./run_docker.sh service      # Service mode
./run_docker.sh status       # Check status
./run_docker.sh logs         # View logs
./run_docker.sh stop         # Stop containers
```

### Troubleshooting Script
```bash
./troubleshoot.sh            # Diagnose Docker/RTL-SDR issues
```

## Docker Compose Profiles

### Default Profile (Interactive)
```bash
docker-compose up
```
Runs interactive mode with full menu interface.

### Service Profile
```bash
docker-compose --profile service up -d
```
Runs continuous monitoring service in background.

### Web Profile (Future)
```bash
docker-compose --profile web up -d
```
Runs web dashboard interface (planned feature).

## Troubleshooting

### Raspberry Pi Docker Compose Issues
If you get `Not supported URL scheme http+docker` error:
1. Run the troubleshooting script: `./troubleshoot.sh`
2. Use the simple runner instead: `./run_docker.sh build && ./run_docker.sh interactive`
3. For service mode: `./run_docker.sh service`

### RTL-SDR Not Detected
1. Check USB connection: `lsusb | grep -i realtek`
2. Check permissions: `groups | grep plugdev`
3. Restart container: `./deploy.sh stop && ./deploy.sh service`

### No Trained Models
1. Run interactive mode first: `./deploy.sh interactive`
2. Collect data (menu option 1)
3. Train models (menu option 2)
4. Exit and start service mode

### Container Won't Start
1. Check logs: `./deploy.sh logs`
2. Verify USB access: `docker run --rm --device=/dev/bus/usb:/dev/bus/usb spectrumalert:latest lsusb`
3. Check disk space: `df -h`

### Service Keeps Restarting
1. Check service logs: `./deploy.sh logs spectrumalert-service`
2. Verify models exist: `ls -la models/`
3. Check RTL-SDR stability: May need USB hub with power supply

## Production Deployment

### 1. System Service (systemd)
Create `/etc/systemd/system/spectrumalert.service`:
```ini
[Unit]
Description=SpectrumAlert RF Monitoring
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/spectrumalert
ExecStart=/opt/spectrumalert/deploy.sh service
ExecStop=/opt/spectrumalert/deploy.sh stop
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable spectrumalert
sudo systemctl start spectrumalert
```

### 2. Auto-restart on Boot
Add to docker-compose.yml:
```yaml
restart: unless-stopped
```

### 3. Log Rotation
Configure in `/etc/logrotate.d/spectrumalert`:
```
/opt/spectrumalert/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 1000 1000
}
```

## Security Considerations

1. **Container Security**
   - Runs as non-root user (spectrum:1000)
   - Minimal base image (python:3.11-slim)
   - No unnecessary packages

2. **Device Access**
   - Uses `--device` mount (preferred over `--privileged`)
   - Limited to USB bus access only

3. **Network Security**
   - Uses host networking for RTL-SDR access
   - No exposed ports by default
   - MQTT credentials in environment file

## Performance Tuning

### Resource Limits
Add to docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '2.0'
    reservations:
      memory: 512M
      cpus: '1.0'
```

### USB Performance
For high sample rates, consider:
1. Dedicated USB 3.0 port
2. Powered USB hub
3. USB buffer size tuning

## Monitoring and Alerts

### Health Checks
Built-in health check verifies RTL-SDR access every 30 seconds.

### Service Status
Check service status file:
```bash
cat logs/service_status.json
```

### Log Monitoring
Monitor logs for alerts:
```bash
tail -f logs/service.log | grep -i "anomaly\|alert"
```

## Backup and Recovery

### Backup Models
```bash
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
```

### Backup Configuration
```bash
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/ .env
```

### Restore
```bash
tar -xzf models-backup-YYYYMMDD.tar.gz
tar -xzf config-backup-YYYYMMDD.tar.gz
```
