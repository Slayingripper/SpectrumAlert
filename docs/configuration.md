# Environment Configuration

SpectrumAlert supports configuration through environment variables for better deployment flexibility.

## Web Server Configuration

### Port Configuration
- **Environment Variable**: `SPECTRUM_ALERT_PORT`
- **Default**: `8000`
- **Description**: Port number for the web dashboard
- **Example**: 
  ```bash
  export SPECTRUM_ALERT_PORT=9000
  spectrum-alert web
  ```

### Host Configuration
- **Environment Variable**: `SPECTRUM_ALERT_HOST`
- **Default**: `0.0.0.0`
- **Description**: Host address to bind the web server to
- **Example**:
  ```bash
  export SPECTRUM_ALERT_HOST=127.0.0.1
  spectrum-alert web
  ```

### Combined Usage
```bash
# Set environment variables
export SPECTRUM_ALERT_PORT=9000
export SPECTRUM_ALERT_HOST=127.0.0.1

# Start web interface
spectrum-alert web

# Or use CLI options (overrides environment variables)
spectrum-alert web --port 9000 --host 127.0.0.1

# Or use the direct run script
cd spectrum_alert/web
python run.py --port 9000 --host 127.0.0.1
```

## Docker Configuration

When using Docker, you can set these environment variables in your docker-compose.yml:

```yaml
version: '3.8'
services:
  spectrum-alert:
    image: spectrum-alert:latest
    environment:
      - SPECTRUM_ALERT_PORT=8080
      - SPECTRUM_ALERT_HOST=0.0.0.0
    ports:
      - "8080:8080"
```

## Production Deployment

For production deployments, consider:

1. **Port Configuration**: Use a reverse proxy (nginx, Apache) and run SpectrumAlert on a non-standard port
2. **Host Binding**: Bind to `127.0.0.1` if using a reverse proxy, or `0.0.0.0` for direct access
3. **Security**: Use HTTPS termination at the reverse proxy level

### Example nginx configuration:
```nginx
server {
    listen 80;
    server_name spectrum-alert.example.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
