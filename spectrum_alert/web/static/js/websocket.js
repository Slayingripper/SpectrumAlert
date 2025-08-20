// WebSocket connection for real-time updates

class WebSocketManager {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.isConnecting = false;
        
        this.connect();
    }

    connect() {
        if (this.isConnecting || (this.websocket && this.websocket.readyState === WebSocket.CONNECTING)) {
            return;
        }

        this.isConnecting = true;
        
        // Determine WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.websocket = new WebSocket(wsUrl);
            this.setupEventHandlers();
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.handleConnectionError();
        }
    }

    setupEventHandlers() {
        this.websocket.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            
            // Update connection status
            if (window.dashboard) {
                window.dashboard.updateConnectionStatus(true);
                window.dashboard.showNotification('Real-time connection established', 'success');
            }
        };

        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnecting = false;
            
            // Update connection status
            if (window.dashboard) {
                window.dashboard.updateConnectionStatus(false);
            }
            
            // Attempt to reconnect if not a normal closure
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError();
        };
    }

    handleMessage(data) {
        if (!window.dashboard) return;

        switch (data.type) {
            case 'system_update':
                this.handleSystemUpdate(data.data);
                break;
                
            case 'anomaly_detected':
                this.handleAnomalyDetected(data.data);
                break;
                
            case 'spectrum_update':
                this.handleSpectrumUpdate(data.data);
                break;
                
            case 'model_update':
                this.handleModelUpdate(data.data);
                break;
                
            case 'status_update':
                this.handleStatusUpdate(data.data);
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    handleSystemUpdate(data) {
        // Update system metrics in real-time
        if (window.dashboard) {
            window.dashboard.updateSystemMetrics(data);
            window.dashboard.updateStatusIndicators(data);
        }
    }

    handleAnomalyDetected(data) {
        // Show new anomaly notification
        const severity = data.severity || 'medium';
        const message = `${severity.toUpperCase()} anomaly detected at ${data.frequency_mhz} MHz`;
        
        if (window.dashboard) {
            window.dashboard.showNotification(message, this.getSeverityType(severity));
            
            // Trigger anomaly data refresh
            window.dashboard.updateAnomalyData();
        }
        
        // Update threat status indicator
        const threatStatus = document.getElementById('threat-status');
        if (threatStatus) {
            threatStatus.className = 'indicator danger';
        }
        
        // Add visual/audio alert for critical anomalies
        if (severity === 'critical') {
            this.triggerCriticalAlert();
        }
    }

    handleSpectrumUpdate(data) {
        // Update spectrum chart with new data
        if (window.dashboard && window.dashboard.charts.spectrum) {
            const chart = window.dashboard.charts.spectrum;
            
            if (data.frequencies && data.powers) {
                chart.data.labels = data.frequencies;
                chart.data.datasets[0].data = data.powers;
                chart.update('none'); // No animation for real-time updates
            }
            
            // Update spectrum statistics
            if (data.peak_frequency) {
                document.getElementById('peak-frequency').textContent = 
                    `${data.peak_frequency.toFixed(2)} MHz`;
            }
            if (data.avg_power) {
                document.getElementById('avg-power').textContent = 
                    `${data.avg_power.toFixed(1)} dBm`;
            }
            if (data.snr) {
                document.getElementById('snr-value').textContent = 
                    `${data.snr.toFixed(1)} dB`;
            }
        }
    }

    handleModelUpdate(data) {
        // Update model performance metrics
        if (window.dashboard) {
            window.dashboard.updateModelStats(data);
            
            // Show notification for significant model changes
            if (data.retrained) {
                window.dashboard.showNotification('Model retrained successfully', 'success');
            }
        }
    }

    handleStatusUpdate(data) {
        // Update various status indicators
        if (data.monitoring_status !== undefined) {
            const monitorStatus = document.getElementById('monitor-status');
            if (monitorStatus) {
                monitorStatus.className = 'indicator';
                if (!data.monitoring_status) {
                    monitorStatus.classList.add('warning');
                }
            }
        }
        
        if (data.system_health !== undefined) {
            const systemStatus = document.getElementById('system-status');
            if (systemStatus) {
                systemStatus.className = 'indicator';
                if (data.system_health === 'warning') {
                    systemStatus.classList.add('warning');
                } else if (data.system_health === 'critical') {
                    systemStatus.classList.add('danger');
                }
            }
        }
    }

    triggerCriticalAlert() {
        // Flash the page border red
        document.body.style.boxShadow = 'inset 0 0 50px rgba(255, 0, 85, 0.8)';
        setTimeout(() => {
            document.body.style.boxShadow = '';
        }, 1000);
        
        // Try to play an alert sound (if available)
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmcbBSuR2/LNeykkrTzEi'); 
            audio.volume = 0.3;
            audio.play().catch(() => {}); // Ignore errors if audio can't play
        } catch (error) {
            // Audio not supported or blocked
        }
    }

    getSeverityType(severity) {
        switch (severity.toLowerCase()) {
            case 'critical':
                return 'error';
            case 'high':
                return 'warning';
            case 'medium':
                return 'info';
            case 'low':
                return 'success';
            default:
                return 'info';
        }
    }

    handleConnectionError() {
        this.isConnecting = false;
        
        if (window.dashboard) {
            window.dashboard.updateConnectionStatus(false);
            
            if (this.reconnectAttempts === 0) {
                window.dashboard.showNotification('Real-time connection lost', 'error');
            }
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            if (window.dashboard) {
                window.dashboard.showNotification('Unable to establish real-time connection', 'error');
            }
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
        
        console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (this.websocket.readyState === WebSocket.CLOSED) {
                this.connect();
            }
        }, delay);
    }

    disconnect() {
        if (this.websocket) {
            this.websocket.close(1000, 'Client disconnecting');
        }
    }

    // Send message to server (for future use)
    send(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, cannot send message');
        }
    }
}

// Initialize WebSocket connection when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for the dashboard to initialize
    setTimeout(() => {
        window.wsManager = new WebSocketManager();
    }, 1000);
});

// Clean up WebSocket connection when page unloads
window.addEventListener('beforeunload', function() {
    if (window.wsManager) {
        window.wsManager.disconnect();
    }
});

// Handle visibility change to manage connection
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible' && window.wsManager) {
        // Page became visible, ensure connection is active
        if (window.wsManager.websocket.readyState === WebSocket.CLOSED) {
            window.wsManager.connect();
        }
    }
});
