# SpectrumAlert Web Dashboard

A cyberpunk-themed web interface for real-time RF spectrum monitoring and anomaly detection.

## Features

- **Real-time Monitoring**: Live spectrum analysis with WebSocket updates
- **Anomaly Detection**: Visual alerts for suspicious RF activity with severity classification
- **System Status**: CPU, memory, temperature monitoring
- **Model Analytics**: ML model performance metrics and training statistics
- **Interactive Charts**: Spectrum visualization, anomaly distribution, model performance
- **Cyberpunk Theme**: Dark interface with red/cyan accents matching the skull logo aesthetic

## Installation

Install the required web dependencies:

```bash
pip install fastapi uvicorn jinja2 python-multipart psutil
```

## Quick Start

### Method 1: Using CLI command

```bash
spectrum-alert web --port 8000 --host 0.0.0.0
```

### Method 2: Direct script execution

```bash
python spectrum_alert/web/run.py
```

### Method 3: Docker

Add web dependencies to your Dockerfile and expose port 8000:

```dockerfile
# In your existing Dockerfile
RUN pip install fastapi uvicorn jinja2 python-multipart psutil
EXPOSE 8000

# Run both monitoring and web interface
CMD ["sh", "-c", "spectrum-alert autonomous-monitoring & python spectrum_alert/web/run.py"]
```

## Dashboard Sections

### System Status
- CPU and memory usage with real-time bars
- System uptime and temperature monitoring
- Status indicators for system health, monitoring, and threats

### Spectrum Analysis
- Real-time spectrum chart with frequency/power visualization
- Peak frequency, average power, and SNR statistics
- Frequency band selection (VHF, UHF, Microwave)

### Threat Detection
- Anomaly counters by severity (Critical, High, Medium, Low)
- Donut chart showing anomaly distribution
- Auto-detection toggle

### Recent Alerts
- Live feed of detected anomalies
- Severity-coded alerts with timestamps
- Frequency and confidence score details

### Model Status
- ML model accuracy and confidence metrics
- Training sample count and last update time
- Model performance chart over time

### Configuration
- Detection threshold slider
- Frequency range configuration
- Scan interval adjustment
- Strict mode toggle

## Real-time Features

The dashboard uses WebSockets for real-time updates:

- **System metrics** update every 5 seconds
- **Anomaly alerts** appear immediately when detected
- **Spectrum data** updates continuously during monitoring
- **Connection status** indicator shows WebSocket health

## API Endpoints

The web interface exposes REST API endpoints:

- `GET /` - Dashboard homepage
- `GET /api/system/status` - System metrics
- `GET /api/anomalies/recent` - Recent anomalies with severity counts
- `GET /api/spectrum/analysis` - Spectrum analysis data
- `GET /api/model/info` - ML model information
- `WS /ws` - WebSocket for real-time updates

## Customization

### Themes
The cyberpunk theme uses CSS custom properties that can be easily modified:

```css
:root {
    --accent-red: #ff0080;
    --accent-cyan: #00ffff;
    --accent-purple: #8b00ff;
    /* ... other colors */
}
```

### Charts
Chart.js is used for visualizations. Charts can be customized by modifying the JavaScript in `static/js/dashboard.js`.

### Notifications
Real-time notifications appear for:
- Critical anomalies (with visual/audio alerts)
- System status changes
- Connection status updates
- Configuration changes

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Use a different port
spectrum-alert web --port 8080
```

### WebSocket Connection Issues
- Ensure firewall allows WebSocket connections
- Check that the port is accessible from client machines
- Verify no proxy is blocking WebSocket upgrades

### Missing Dependencies
```bash
# Install all web dependencies
pip install fastapi uvicorn jinja2 python-multipart psutil

# Or install everything at once
pip install spectrum-alert[web]
```

### Performance Issues
- Reduce update frequency in dashboard.js
- Limit the number of data points in charts
- Use the frequency range selector to focus on specific bands

## Security Considerations

- The web interface binds to `0.0.0.0` by default for Docker compatibility
- For production deployments, consider:
  - Using a reverse proxy (nginx, Apache)
  - Adding authentication/authorization
  - Enabling HTTPS
  - Restricting access by IP address

## Browser Compatibility

The dashboard uses modern web technologies:
- WebSockets for real-time updates
- CSS Grid and Flexbox for layout
- Chart.js for visualizations
- Font Awesome for icons

Recommended browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
