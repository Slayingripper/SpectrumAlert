// Dashboard JavaScript for SpectrumAlert Web Interface

// Global state
let currentMonitoring = false;
let trainingInProgress = false;
let dataStatsInterval = null;
let systemStatsInterval = null;
let mqttConnected = false;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initializing...');
    initializeWebSocket();
    loadDataStats();
    loadDataFiles(); // Load available data files
    loadTrainingStatus();
    loadModelStatus();
    loadAnalytics();
    loadSpectrumAnalysis(); // Load real spectrum analysis
    loadSystemStatus();
    loadMQTTStatus();
    loadRecentAnomalies();
    loadMonitoringStatus(); // Check initial monitoring status
    startDataStatsRefresh();
    startSystemStatsRefresh();
    startModelStatusRefresh();
    startAnalyticsRefresh();
    startSpectrumRefresh(); // Start spectrum analysis refresh
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
    
    // Add slider value updates for training configuration
    setupTrainingConfigSliders();
});

// WebSocket connection management
let websocket = null;

function initializeWebSocket() {
    console.log('Initializing WebSocket connection...');
    
    try {
        // Create WebSocket connection
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = function(event) {
            console.log('WebSocket connected successfully');
            showNotification('Real-time connection established', 'success');
        };
        
        websocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        websocket.onclose = function(event) {
            console.log('WebSocket connection closed');
            showNotification('Real-time connection lost', 'warning');
            
            // Attempt to reconnect after 5 seconds
            setTimeout(initializeWebSocket, 5000);
        };
        
        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
            showNotification('Real-time connection error', 'error');
        };
        
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        // Fallback to polling if WebSocket fails
        setTimeout(initializeWebSocket, 10000);
    }
}

// System Status Functions
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/system/advanced-status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success' && result.data) {
                updateSystemStatus(result.data);
            }
        }
    } catch (error) {
        console.error('Error loading system status:', error);
    }
}

function updateSystemStatus(status) {
    // Update CPU usage
    if (status.performance) {
        updateMetricValue('cpu-usage', status.performance.cpu_usage + '%');
        updateMetricBar('cpu-bar', status.performance.cpu_usage);
        
        updateMetricValue('memory-usage', status.performance.memory_usage + '%');
        updateMetricBar('memory-bar', status.performance.memory_usage);
        
        if (status.performance.temperature) {
            updateMetricValue('temperature', Math.round(status.performance.temperature) + '°C');
        }
    }
    
    // Update monitoring status
    if (status.monitoring) {
        const monitoringStatus = status.monitoring.active ? 'ACTIVE' : 'STOPPED';
        updateStatusIndicator('monitor-status', status.monitoring.active);
        updateElementText('monitoring-status', monitoringStatus);
    }
    
    // Update SDR status
    if (status.hardware) {
        updateElementText('sdr-status', status.hardware.sdr_status);
    }
}

function updateMetricValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function updateMetricBar(id, percentage) {
    const element = document.getElementById(id);
    if (element) {
        element.style.width = percentage + '%';
        // Color coding
        if (percentage > 80) {
            element.style.backgroundColor = '#ff0055';
        } else if (percentage > 60) {
            element.style.backgroundColor = '#ffaa00';
        } else {
            element.style.backgroundColor = '#00ffff';
        }
    }
}

function updateStatusIndicator(id, isActive) {
    const element = document.getElementById(id);
    if (element) {
        const led = element.querySelector('.led');
        if (led) {
            led.className = 'led ' + (isActive ? 'active' : 'inactive');
        }
    }
}

function updateElementText(id, text) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
    }
}

function startSystemStatsRefresh() {
    systemStatsInterval = setInterval(loadSystemStatus, 5000); // Every 5 seconds
}

// Recent Anomalies Functions
async function loadRecentAnomalies() {
    try {
        const response = await fetch('/api/anomalies/recent');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.anomalies) {
                updateRecentAnomalies(result.anomalies);
            }
        }
    } catch (error) {
        console.error('Error loading recent anomalies:', error);
    }
}

function updateRecentAnomalies(anomalies) {
    const anomaliesList = document.getElementById('recent-anomalies-list');
    if (!anomaliesList) return;
    
    // Clear existing anomalies
    anomaliesList.innerHTML = '';
    
    if (anomalies.length === 0) {
        anomaliesList.innerHTML = '<div class="no-anomalies">No recent anomalies detected</div>';
        return;
    }
    
    // Display the most recent anomalies
    anomalies.slice(0, 10).forEach(anomaly => {
        const anomalyElement = document.createElement('div');
        anomalyElement.className = `anomaly-item severity-${anomaly.severity || 'medium'}`;
        
        const timestamp = new Date(anomaly.timestamp).toLocaleString();
        const frequency = (anomaly.frequency_hz / 1e6).toFixed(3);
        const confidence = (anomaly.confidence_score * 100).toFixed(1);
        
        anomalyElement.innerHTML = `
            <div class="anomaly-header">
                <span class="anomaly-time">${timestamp}</span>
                <span class="anomaly-severity">${(anomaly.severity || 'medium').toUpperCase()}</span>
            </div>
            <div class="anomaly-details">
                <div class="anomaly-frequency">${frequency} MHz</div>
                <div class="anomaly-confidence">Confidence: ${confidence}%</div>
                <div class="anomaly-description">${anomaly.description || 'Anomaly detected'}</div>
            </div>
        `;
        
        anomaliesList.appendChild(anomalyElement);
    });
}

// Monitoring Status Functions
async function loadMonitoringStatus() {
    try {
        const response = await fetch('/api/monitoring/status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                updateMonitoringStatus(result.data);
            }
        }
    } catch (error) {
        console.error('Error loading monitoring status:', error);
    }
}

function updateMonitoringStatus(status) {
    // Update monitoring status indicators
    const monitoringActive = status.monitoring_active || false;
    updateStatusIndicator('monitoring-status', monitoringActive);
    updateElementText('monitoring-state', monitoringActive ? 'ACTIVE' : 'STOPPED');
}

// Advanced Monitoring Functions
async function startAdvancedMonitoring() {
    try {
        const config = {
            frequency_start: parseFloat(document.getElementById('frequency-range-config')?.value?.split('-')[0] || 144),
            frequency_end: parseFloat(document.getElementById('frequency-range-config')?.value?.split('-')[1] || 148),
            threshold: parseFloat(document.getElementById('threshold-slider')?.value || 0.8),
            strict_threshold: document.getElementById('strict-mode')?.checked || true,
            dc_exclude_hz: parseInt(document.getElementById('dcExclude')?.value || 8000),
            edge_exclude_hz: parseInt(document.getElementById('edgeExclude')?.value || 20000),
            novelty_filter: document.getElementById('noveltyFilter')?.checked || false,
            continuous_learning: document.getElementById('continuousLearning')?.checked || false
        };

        console.log('Starting advanced monitoring with config:', config);
        
        const response = await fetch('/api/monitoring/advanced', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            const result = await response.json();
            currentMonitoring = true;
            updateMonitoringUI(true);
            console.log('Advanced monitoring started:', result);
            showNotification('Advanced monitoring started successfully', 'success');
        } else {
            const error = await response.text();
            console.error('Failed to start advanced monitoring:', error);
            showNotification('Failed to start advanced monitoring', 'error');
        }
    } catch (error) {
        console.error('Error starting advanced monitoring:', error);
        showNotification('Error starting advanced monitoring', 'error');
    }
}

async function startMultiBandMonitoring() {
    try {
        const bands = [
            document.getElementById('band1')?.value,
            document.getElementById('band2')?.value,
            document.getElementById('band3')?.value
        ].filter(band => band && band.trim());

        if (bands.length === 0) {
            showNotification('Please configure at least one frequency band', 'warning');
            return;
        }

        const config = {
            bands: bands,
            data_minutes: 5,
            monitor_minutes: 15,
            max_cycles_per_band: 1,
            threshold: parseFloat(document.getElementById('threshold')?.value || 0.8),
            strict_threshold: document.getElementById('strictThreshold')?.checked,
            dc_exclude_hz: parseInt(document.getElementById('dcExclude')?.value || 10000),
            edge_exclude_hz: parseInt(document.getElementById('edgeExclude')?.value || 25000),
            continuous_learning: document.getElementById('continuousLearning')?.checked,
            novelty_filter: document.getElementById('noveltyFilter')?.checked
        };

        console.log('Starting multi-band monitoring with config:', config);
        
        const response = await fetch('/api/monitoring/multiband', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            const result = await response.json();
            currentMonitoring = true;
            updateMonitoringUI(true);
            console.log('Multi-band monitoring started:', result);
            showNotification(`Multi-band monitoring started for ${bands.length} bands`, 'success');
        } else {
            const error = await response.text();
            console.error('Failed to start multi-band monitoring:', error);
            showNotification('Failed to start multi-band monitoring', 'error');
        }
    } catch (error) {
        console.error('Error starting multi-band monitoring:', error);
        showNotification('Error starting multi-band monitoring', 'error');
    }
}

// MQTT Integration Functions
async function loadMQTTStatus() {
    try {
        const response = await fetch('/api/mqtt/status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success' && result.data) {
                updateMQTTStatus(result.data);
            }
        }
    } catch (error) {
        console.error('Error loading MQTT status:', error);
    }
}

function updateMQTTStatus(status) {
    mqttConnected = status.connected;
    updateElementText('mqtt-connection-status', status.connected ? 'Connected' : 'Disconnected');
    updateStatusIndicator('mqtt-status', status.connected);
    
    if (status.config) {
        if (document.getElementById('mqttBroker')) {
            document.getElementById('mqttBroker').value = status.config.broker || 'localhost';
        }
        if (document.getElementById('mqttPort')) {
            document.getElementById('mqttPort').value = status.config.port || 1883;
        }
        if (document.getElementById('mqttTopic')) {
            document.getElementById('mqttTopic').value = status.config.topic_prefix || 'spectrum_alert';
        }
    }
}

async function connectMQTT() {
    try {
        const config = {
            broker: document.getElementById('mqttBroker')?.value || 'localhost',
            port: parseInt(document.getElementById('mqttPort')?.value || 1883),
            topic_prefix: document.getElementById('mqttTopic')?.value || 'spectrum_alert'
        };

        console.log('Connecting to MQTT with config:', config);
        
        const response = await fetch('/api/mqtt/connect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            const result = await response.json();
            console.log('MQTT connected:', result);
            showNotification('MQTT connected successfully', 'success');
            await loadMQTTStatus();
        } else {
            const error = await response.text();
            console.error('Failed to connect MQTT:', error);
            showNotification('Failed to connect to MQTT', 'error');
        }
    } catch (error) {
        console.error('Error connecting MQTT:', error);
        showNotification('Error connecting to MQTT', 'error');
    }
}

async function disconnectMQTT() {
    try {
        const response = await fetch('/api/mqtt/disconnect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            console.log('MQTT disconnected:', result);
            showNotification('MQTT disconnected successfully', 'success');
            await loadMQTTStatus();
        } else {
            const error = await response.text();
            console.error('Failed to disconnect MQTT:', error);
            showNotification('Failed to disconnect MQTT', 'error');
        }
    } catch (error) {
        console.error('Error disconnecting MQTT:', error);
        showNotification('Error disconnecting MQTT', 'error');
    }
}

async function testMQTT() {
    try {
        if (!mqttConnected) {
            showNotification('MQTT not connected', 'warning');
            return;
        }

        const response = await fetch('/api/mqtt/test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            console.log('MQTT test sent:', result);
            showNotification('MQTT test message sent successfully', 'success');
        } else {
            const error = await response.text();
            console.error('Failed to send MQTT test:', error);
            showNotification('Failed to send MQTT test', 'error');
        }
    } catch (error) {
        console.error('Error sending MQTT test:', error);
        showNotification('Error sending MQTT test', 'error');
    }
}

// Notification System
function showNotification(message, type = 'info') {
    const container = document.getElementById('notifications');
    if (!container) return;

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;

    container.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Configuration and utility functions
function saveConfig() {
    const config = {
        threshold: parseFloat(document.getElementById('threshold-slider')?.value || 0.8),
        frequency_range: document.getElementById('frequency-range-config')?.value || '88-108',
        scan_interval: parseInt(document.getElementById('scan-interval')?.value || 5),
        strict_mode: document.getElementById('strict-mode')?.checked || false
    };

    localStorage.setItem('spectrumAlertConfig', JSON.stringify(config));
    showNotification('Configuration saved successfully', 'success');
}

function loadConfig() {
    try {
        const saved = localStorage.getItem('spectrumAlertConfig');
        if (saved) {
            const config = JSON.parse(saved);
            
            if (document.getElementById('threshold-slider')) {
                document.getElementById('threshold-slider').value = config.threshold || 0.8;
                updateSliderValue('threshold-slider', 'threshold-value');
            }
            if (document.getElementById('frequency-range-config')) {
                document.getElementById('frequency-range-config').value = config.frequency_range || '88-108';
            }
            if (document.getElementById('scan-interval')) {
                document.getElementById('scan-interval').value = config.scan_interval || 5;
            }
            if (document.getElementById('strict-mode')) {
                document.getElementById('strict-mode').checked = config.strict_mode || false;
            }
        }
    } catch (error) {
        console.error('Error loading configuration:', error);
    }
}

function updateSliderValue(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const value = document.getElementById(valueId);
    if (slider && value) {
        value.textContent = slider.value;
    }
}

function updateTimestamp() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    updateElementText('current-time', timeString);
}

function refreshSystemData() {
    loadSystemStatus();
    showNotification('System data refreshed', 'info');
}

function clearAlerts() {
    const alertsList = document.getElementById('alerts-list');
    if (alertsList) {
        alertsList.innerHTML = '<div class="no-alerts">No recent alerts</div>';
        showNotification('Alerts cleared', 'info');
    }
}

function retrainModel() {
    // Enhanced retrain function with better feedback
    if (trainingInProgress) {
        showNotification('⚠️ Training already in progress', 'warning');
        return;
    }
    
    showNotification('🔄 Starting model retraining...', 'info');
    
    // Update UI to show retraining has started
    const retrainBtn = document.querySelector('.btn-retrain');
    if (retrainBtn) {
        retrainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> RETRAINING';
        retrainBtn.disabled = true;
    }
    
    // Call the main training function with additional retrain feedback
    trainModel().then(() => {
        showNotification('✅ Model retrained successfully!', 'success');
        // Refresh model status to show updated metrics
        setTimeout(() => {
            loadModelStatus();
        }, 2000);
    }).catch((error) => {
        showNotification('❌ Retraining failed: ' + error.message, 'error');
    }).finally(() => {
        // Reset retrain button
        if (retrainBtn) {
            retrainBtn.innerHTML = '<i class="fas fa-sync"></i> RETRAIN';
            retrainBtn.disabled = false;
        }
    });
}

// Monitoring Control Functions
async function startMonitoring() {
    if (currentMonitoring) {
        console.log('Monitoring already in progress');
        showNotification('Monitoring is already active', 'warning');
        return;
    }

    try {
        showNotification('Starting spectrum monitoring...', 'info');
        
        // Collect configuration from UI elements
        const config = {
            frequency_range: document.getElementById('frequency-range-config')?.value || '144-146',
            sample_rate: parseInt(document.getElementById('sample-rate')?.value || 2048000),
            gain: 20,
            threshold: parseFloat(document.getElementById('threshold-slider')?.value || 0.8),
            scan_interval: parseInt(document.getElementById('scan-interval')?.value || 5),
            strict_mode: document.getElementById('strict-mode')?.checked || false
        };
        
        console.log('Starting monitoring with config:', config);
        
        const response = await fetch('/api/monitoring/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            const result = await response.json();
            currentMonitoring = true;
            updateMonitoringUI(true);
            
            // Update status indicators
            updateStatusIndicator('monitor-status', true);
            updateStatusIndicator('system-status', true);
            
            console.log('Monitoring started successfully');
            showNotification('✅ Spectrum monitoring started successfully!', 'success');
            
            // Start polling for monitoring status
            startMonitoringStatusPolling();
            
            // Refresh spectrum analysis with new frequency range
            loadSpectrumAnalysis();
            
        } else {
            console.error('Failed to start monitoring:', response.statusText);
            showNotification('❌ Failed to start monitoring: ' + response.statusText, 'error');
        }
    } catch (error) {
        console.error('Error starting monitoring:', error);
        showNotification('❌ Error starting monitoring: ' + error.message, 'error');
    }
}

async function stopMonitoring() {
    if (!currentMonitoring) {
        console.log('Monitoring not currently active');
        showNotification('Monitoring is not currently active', 'warning');
        return;
    }

    try {
        showNotification('Stopping spectrum monitoring...', 'info');
        
        const response = await fetch('/api/monitoring/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            currentMonitoring = false;
            updateMonitoringUI(false);
            
            // Update status indicators
            updateStatusIndicator('monitor-status', false);
            
            console.log('Monitoring stopped successfully');
            showNotification('🛑 Spectrum monitoring stopped', 'success');
            
            // Stop monitoring status polling
            stopMonitoringStatusPolling();
            
        } else {
            console.error('Failed to stop monitoring:', response.statusText);
            showNotification('❌ Failed to stop monitoring: ' + response.statusText, 'error');
        }
    } catch (error) {
        console.error('Error stopping monitoring:', error);
        showNotification('❌ Error stopping monitoring: ' + error.message, 'error');
    }
}

let monitoringStatusInterval = null;

function startMonitoringStatusPolling() {
    // Clear any existing interval
    if (monitoringStatusInterval) {
        clearInterval(monitoringStatusInterval);
    }
    
    // Poll monitoring status every 3 seconds
    monitoringStatusInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/monitoring/status');
            if (response.ok) {
                const result = await response.json();
                if (result.status === 'ok' && result.data) {
                    const isActive = result.data.active || false;
                    const deviceInfo = result.data.device_info || {};
                    
                    // Update UI state if it doesn't match
                    if (currentMonitoring !== isActive) {
                        currentMonitoring = isActive;
                        updateMonitoringUI(isActive);
                        updateStatusIndicator('monitor-status', isActive);
                        
                        if (isActive) {
                            showNotification('📡 Monitoring detected as active', 'success');
                        }
                    }
                    
                    // Update device info if available
                    if (deviceInfo.frequency_hz) {
                        updateElementText('current-frequency', (deviceInfo.frequency_hz / 1e6).toFixed(2) + ' MHz');
                    }
                    if (deviceInfo.sample_rate) {
                        updateElementText('sample-rate', (deviceInfo.sample_rate / 1e6).toFixed(1) + ' MS/s');
                    }
                    if (deviceInfo.gain !== undefined) {
                        updateElementText('current-gain', deviceInfo.gain.toFixed(1) + ' dB');
                    }
                }
            }
        } catch (error) {
            console.error('Error polling monitoring status:', error);
        }
    }, 3000);
}

function stopMonitoringStatusPolling() {
    if (monitoringStatusInterval) {
        clearInterval(monitoringStatusInterval);
        monitoringStatusInterval = null;
    }
}

// Load initial monitoring status
async function loadMonitoringStatus() {
    try {
        const response = await fetch('/api/monitoring/status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                const isActive = result.data.active || false;
                currentMonitoring = isActive;
                updateMonitoringUI(isActive);
                updateStatusIndicator('monitor-status', isActive);
                updateStatusIndicator('system-status', isActive);
                
                console.log(`Initial monitoring status: ${isActive ? 'ACTIVE' : 'INACTIVE'}`);
                
                if (isActive) {
                    showNotification('📡 Monitoring is currently active', 'success');
                    startMonitoringStatusPolling(); // Start polling if already active
                }
            }
        }
    } catch (error) {
        console.error('Error loading monitoring status:', error);
    }
}

function updateMonitoringUI(isMonitoring) {
    const startBtn = document.getElementById('start-monitoring');
    const stopBtn = document.getElementById('stop-monitoring');
    
    if (startBtn && stopBtn) {
        startBtn.style.display = isMonitoring ? 'none' : 'inline-block';
        stopBtn.style.display = isMonitoring ? 'inline-block' : 'none';
    }
}

// Model Training Functions
async function trainModel() {
    if (trainingInProgress) {
        console.log('Training already in progress');
        showNotification('Training already in progress', 'warning');
        return;
    }

    try {
        // Get model type from the new selector
        const modelType = document.getElementById('model-type-select')?.value || 'isolation_forest';
        
        // Get dataset selection from the new selector
        const datasetSelect = document.getElementById('dataset-select')?.value || 'all';
        
        // Get training configuration
        const epochs = parseInt(document.getElementById('training-epochs')?.value || 100);
        const contamination = parseFloat(document.getElementById('contamination')?.value || 0.1);
        
        // Get configuration values
        const config = {
            model_type: modelType,
            dataset: datasetSelect,
            epochs: epochs,
            contamination: contamination,
            save_model: true
        };

        // Create training details message
        let trainingDetails = `Model Type: ${modelType}\nDataset: ${datasetSelect}\nEpochs: ${epochs}\nContamination: ${contamination}`;
        
        console.log('Starting model training with config:', config);
        showNotification(`🔄 Starting ${modelType} training...\n${trainingDetails}`, 'info');
        
        // Show detailed training info in the UI
        updateTrainingProgressDisplay({
            model_type: modelType,
            dataset: datasetSelect,
            status: 'Initializing...',
            progress: 0,
            epochs: epochs,
            contamination: contamination
        });
        
        const response = await fetch('/api/model/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            const result = await response.json();
            trainingInProgress = true;
            updateTrainingUI(true);
            console.log('Training started:', result);
            
            showNotification(`✅ Training started successfully!\nModel: ${modelType}\nDataset: ${datasetSelect}`, 'success');
            
            // Start polling for training status
            startTrainingStatusPolling();
        } else {
            const error = await response.text();
            console.error('Failed to start training:', error);
            showNotification('❌ Failed to start training: ' + error, 'error');
        }
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('❌ Error starting training: ' + error.message, 'error');
    }
}

function updateTrainingProgressDisplay(details) {
    // Check if we have a training progress display, if not create it
    let progressDisplay = document.getElementById('training-progress-display');
    if (!progressDisplay) {
        const trainingPanel = document.querySelector('.training-panel .panel-content');
        if (trainingPanel) {
            const progressHTML = `
                <div id="training-progress-display" class="training-progress-detailed" style="
                    background: #2a2a2a; border: 1px solid #ffaa00; border-radius: 5px; 
                    padding: 15px; margin-bottom: 15px; display: none;
                ">
                    <div class="progress-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="color: #ffaa00; margin: 0;"><i class="fas fa-brain"></i> Training Progress</h4>
                        <button onclick="hideTrainingProgress()" style="
                            background: transparent; border: 1px solid #888; color: #888; 
                            padding: 2px 6px; border-radius: 3px; cursor: pointer; font-size: 10px;
                        ">Hide</button>
                    </div>
                    
                    <div class="training-details" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-bottom: 15px;">
                        <div class="detail-item">
                            <div style="color: #888; font-size: 11px;">Model Type</div>
                            <div id="training-model-type" style="color: #ffaa00; font-weight: bold;"></div>
                        </div>
                        <div class="detail-item">
                            <div style="color: #888; font-size: 11px;">Dataset</div>
                            <div id="training-dataset" style="color: #00aaff; font-weight: bold;"></div>
                        </div>
                        <div class="detail-item">
                            <div style="color: #888; font-size: 11px;">Status</div>
                            <div id="training-status-text" style="color: #00ff88; font-weight: bold;"></div>
                        </div>
                        <div class="detail-item">
                            <div style="color: #888; font-size: 11px;">Progress</div>
                            <div id="training-progress-text" style="color: #ffffff; font-weight: bold;"></div>
                        </div>
                    </div>
                    
                    <div class="progress-bar-container" style="background: #1a1a1a; border-radius: 10px; height: 20px; overflow: hidden; position: relative;">
                        <div id="training-progress-bar" class="progress-bar-fill" style="
                            background: linear-gradient(90deg, #ffaa00, #ff6600); 
                            height: 100%; width: 0%; transition: width 0.3s ease;
                        "></div>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #000; font-size: 11px; font-weight: bold;">
                            <span id="training-progress-percentage">0%</span>
                        </div>
                    </div>
                    
                    <div id="training-metrics" class="training-metrics" style="
                        margin-top: 10px; padding: 10px; background: #1a1a1a; border-radius: 5px; 
                        font-family: monospace; font-size: 10px; color: #888; display: none;
                    "></div>
                </div>
            `;
            trainingPanel.insertAdjacentHTML('afterbegin', progressHTML);
            progressDisplay = document.getElementById('training-progress-display');
        }
    }
    
    if (progressDisplay) {
        document.getElementById('training-model-type').textContent = details.model_type || 'Unknown';
        document.getElementById('training-dataset').textContent = details.dataset || 'Default';
        document.getElementById('training-status-text').textContent = details.status || 'Ready';
        
        const progress = details.progress || 0;
        document.getElementById('training-progress-text').textContent = `${progress}%`;
        document.getElementById('training-progress-bar').style.width = `${progress}%`;
        document.getElementById('training-progress-percentage').textContent = `${progress}%`;
        
        progressDisplay.style.display = 'block';
        
        // Update metrics if available
        if (details.metrics) {
            const metricsDiv = document.getElementById('training-metrics');
            metricsDiv.innerHTML = `
                <div>Epoch: ${details.metrics.epoch || 'N/A'}</div>
                <div>Loss: ${details.metrics.loss || 'N/A'}</div>
                <div>Accuracy: ${details.metrics.accuracy || 'N/A'}</div>
                <div>ETA: ${details.metrics.eta || 'N/A'}</div>
            `;
            metricsDiv.style.display = 'block';
        }
    }
}

function hideTrainingProgress() {
    const progressDisplay = document.getElementById('training-progress-display');
    if (progressDisplay) {
        progressDisplay.style.display = 'none';
    }
}

async function testModel() {
    try {
        console.log('Starting model testing...');
        
        const response = await fetch('/api/model/test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Model test results:', result);
            displayTestResults(result);
        } else {
            const error = await response.text();
            console.error('Failed to test model:', error);
            alert('Failed to test model: ' + error);
        }
    } catch (error) {
        console.error('Error testing model:', error);
        alert('Error testing model. Check console for details.');
    }
}

async function deployModel() {
    try {
        const confirmDeploy = confirm('Are you sure you want to deploy the current model? This will replace the active model.');
        if (!confirmDeploy) return;

        console.log('Deploying model...');
        
        const response = await fetch('/api/model/deploy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Model deployed successfully:', result);
            alert('Model deployed successfully!');
        } else {
            const error = await response.text();
            console.error('Failed to deploy model:', error);
            alert('Failed to deploy model: ' + error);
        }
    } catch (error) {
        console.error('Error deploying model:', error);
        alert('Error deploying model. Check console for details.');
    }
}

// Training Status Management
async function loadTrainingStatus() {
    try {
        const response = await fetch('/api/model/training/status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success' && result.data) {
                updateTrainingStatus({
                    active: result.data.training_active || false,
                    progress: result.data.training_progress?.progress || 0,
                    status: result.data.training_progress?.status || 'Ready',
                    last_training_time: result.data.last_training || 'Never'
                });
            }
        }
    } catch (error) {
        console.error('Error loading training status:', error);
    }
}

function startTrainingStatusPolling() {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/model/training/status');
            if (response.ok) {
                const result = await response.json();
                if (result.status === 'success' && result.data) {
                    const status = {
                        active: result.data.training_active || false,
                        progress: result.data.training_progress?.progress || 0,
                        status: result.data.training_progress?.status || 'Ready',
                        last_training_time: result.data.last_training || 'Never'
                    };
                    updateTrainingStatus(status);
                    
                    if (!status.active) {
                        clearInterval(pollInterval);
                        trainingInProgress = false;
                        updateTrainingUI(false);
                    }
                }
            }
        } catch (error) {
            console.error('Error polling training status:', error);
        }
    }, 2000); // Poll every 2 seconds
}

function updateTrainingStatus(status) {
    // Update progress bar (original functionality)
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (progressFill && progressText) {
        progressFill.style.width = `${status.progress}%`;
        progressText.textContent = status.status || 'Ready';
    }

    // Update status indicators
    const statusContainer = document.querySelector('.training-status');
    if (statusContainer) {
        statusContainer.className = 'training-status';
        if (status.active) {
            statusContainer.classList.add('training');
        } else if (status.last_training_time && status.last_training_time !== 'Never') {
            statusContainer.classList.add('completed');
        }
    }

    // Update individual status values
    updateStatusValue('training-active', status.active ? 'Yes' : 'No');
    updateStatusValue('training-progress', `${status.progress}%`);
    updateStatusValue('last-training', status.last_training_time || 'Never');
    
    // Update detailed progress display if it exists
    const progressDisplay = document.getElementById('training-progress-display');
    if (progressDisplay && progressDisplay.style.display !== 'none') {
        updateTrainingProgressDisplay({
            model_type: status.model_type || 'Unknown',
            dataset: status.dataset || 'Default',
            status: status.status || 'Ready',
            progress: status.progress || 0,
            metrics: status.metrics
        });
    }
    
    // Show notifications for important status changes
    if (status.active && !trainingInProgress) {
        showNotification(`🔄 Training in progress: ${status.status}`, 'info');
        trainingInProgress = true;
    } else if (!status.active && trainingInProgress) {
        showNotification(`✅ Training completed: ${status.status}`, 'success');
        trainingInProgress = false;
        
        // Auto-refresh model status after training completes
        setTimeout(() => {
            loadModelStatus();
        }, 2000);
    }
}

function updateStatusValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function updateTrainingUI(isTraining) {
    const trainBtn = document.querySelector('.btn-train');
    if (trainBtn) {
        trainBtn.disabled = isTraining;
        trainBtn.textContent = isTraining ? 'Training...' : 'Train Model';
    }
}

function displayTestResults(results) {
    // Update results display
    const resultsContainer = document.querySelector('.training-results');
    if (resultsContainer && results.results && results.results.metrics) {
        updateResultValue('accuracy', (results.results.metrics.accuracy * 100).toFixed(2) + '%');
        updateResultValue('precision', (results.results.metrics.precision * 100).toFixed(2) + '%');
        updateResultValue('recall', (results.results.metrics.recall * 100).toFixed(2) + '%');
        updateResultValue('f1-score', (results.results.metrics.f1_score * 100).toFixed(2) + '%');
    }
}

function updateResultValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

// Data Management Functions
async function loadDataStats() {
    try {
        const response = await fetch('/api/data/stats');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                updateDataStats(result.data);
            } else {
                console.error('Error in data stats response:', result.message);
            }
        }
    } catch (error) {
        console.error('Error loading data stats:', error);
    }
}

function updateDataStats(stats) {
    updateElementContent('data-files', stats.total_files || 0);
    updateElementContent('data-samples', (stats.total_samples || 0).toLocaleString());
    updateElementContent('data-size', (stats.data_size_mb || 0).toFixed(1));
    updateElementContent('data-updated', stats.last_updated || 'Never');
}

function updateElementContent(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function startDataStatsRefresh() {
    // Refresh data stats every 30 seconds
    dataStatsInterval = setInterval(loadDataStats, 30000);
}

async function collectData() {
    try {
        console.log('Starting data collection...');
        
        // Get collection parameters from UI
        const duration = document.getElementById('collect-duration')?.value || 10;
        const frequency = document.getElementById('collect-frequency')?.value || '88-108';
        const sampleRate = document.getElementById('sample-rate')?.value || 2048000;
        
        // Parse frequency range and calculate center frequency
        let centerFreq, startFreq, endFreq;
        if (frequency.includes('-')) {
            const freqParts = frequency.split('-');
            startFreq = parseFloat(freqParts[0]);
            endFreq = parseFloat(freqParts[1]);
            centerFreq = (startFreq + endFreq) / 2; // Use center of range
        } else {
            centerFreq = parseFloat(frequency);
            startFreq = centerFreq;
            endFreq = centerFreq;
        }
        
        const config = {
            duration_minutes: parseInt(duration),
            center_frequency: centerFreq * 1e6, // Convert MHz to Hz
            frequency_start: startFreq * 1e6,
            frequency_end: endFreq * 1e6,
            sample_rate: parseInt(sampleRate)
        };

        // Show estimated completion time with frequency info
        const estimatedCompleteTime = new Date(Date.now() + (parseInt(duration) * 60 * 1000));
        showNotification(`Starting ${duration} minute data collection on ${startFreq}-${endFreq} MHz... Expected completion: ${estimatedCompleteTime.toLocaleTimeString()}`, 'info');

        const response = await fetch('/api/data/collect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Data collection result:', result);
            
            if (result.status === 'success') {
                showNotification(`✅ ${result.message}`, 'success');
                
                // Show collection details if available
                if (result.data) {
                    const durationMins = (result.data.duration_seconds / 60).toFixed(1);
                    const details = `
                        Frequency: ${(result.data.frequency_hz / 1e6).toFixed(2)} MHz
                        Duration: ${durationMins} minutes (${result.data.duration_seconds}s)
                        Samples: ${result.data.sample_count.toLocaleString()}
                        Timestamp: ${new Date(result.data.timestamp).toLocaleString()}
                    `;
                    console.log('Collection details:', details);
                }
                
                // Refresh stats after a short delay
                setTimeout(loadDataStats, 2000);
                // Also refresh file list if it's open
                if (window.dataFilesLoaded) {
                    setTimeout(loadDataFiles, 2000);
                }
            } else {
                showNotification(`Data collection failed: ${result.message}`, 'error');
            }
        } else {
            const error = await response.text();
            console.error('Failed to start data collection:', error);
            showNotification('Failed to start data collection: ' + error, 'error');
        }
    } catch (error) {
        console.error('Error starting data collection:', error);
        showNotification('Error starting data collection. Check console for details.', 'error');
    }
}

async function refreshDataStats() {
    console.log('Refreshing data statistics...');
    await loadDataStats();
    showNotification('Data statistics refreshed', 'success');
}

// Enhanced Data Management Functions
window.dataFilesLoaded = false;

async function loadDataFiles() {
    try {
        const response = await fetch('/api/data/files');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                displayDataFiles(result.data.files);
                window.dataFilesLoaded = true;
            }
        }
    } catch (error) {
        console.error('Error loading data files:', error);
        showNotification('Error loading data files', 'error');
    }
}

function displayDataFiles(files) {
    // Check if we have a data files display area, if not create it
    let dataFilesContainer = document.getElementById('data-files-container');
    if (!dataFilesContainer) {
        // Create and insert data files display area
        const dataPanel = document.querySelector('.data-panel .panel-content');
        if (dataPanel) {
            const dataFilesHTML = `
                <div id="data-files-container" class="data-files-section" style="margin-top: 20px;">
                    <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h4 style="color: #00ff88; margin: 0;"><i class="fas fa-folder-open"></i> Data Files</h4>
                        <div class="data-file-controls">
                            <button class="btn-small" onclick="loadDataFiles()" style="background: #1a1a1a; border: 1px solid #00ff88; color: #00ff88; padding: 5px 10px; margin-right: 5px;">
                                <i class="fas fa-sync-alt"></i> Refresh
                            </button>
                            <button class="btn-small" onclick="showDatasetSelector()" style="background: #1a1a1a; border: 1px solid #00aaff; color: #00aaff; padding: 5px 10px;">
                                <i class="fas fa-database"></i> Datasets
                            </button>
                        </div>
                    </div>
                    <div id="data-files-list" class="data-files-list"></div>
                </div>
            `;
            dataPanel.insertAdjacentHTML('beforeend', dataFilesHTML);
            dataFilesContainer = document.getElementById('data-files-container');
        }
    }
    
    const filesList = document.getElementById('data-files-list');
    if (!filesList) return;
    
    if (files.length === 0) {
        filesList.innerHTML = '<div class="no-files">No data files found. Start monitoring to collect data.</div>';
        return;
    }
    
    let filesHTML = '<div class="files-grid">';
    
    files.slice(0, 10).forEach(file => { // Show only first 10 files
        const fileTypeIcon = file.type === 'csv' ? 'fa-file-csv' : 'fa-file-code';
        const freqInfo = file.frequency_range ? 
            `${(file.frequency_range.min_hz / 1e6).toFixed(1)}-${(file.frequency_range.max_hz / 1e6).toFixed(1)} MHz` : 
            'Unknown freq';
        
        filesHTML += `
            <div class="file-item" onclick="analyzeDataFile('${file.filename}')">
                <div class="file-icon"><i class="fas ${fileTypeIcon}"></i></div>
                <div class="file-info">
                    <div class="file-name">${file.filename}</div>
                    <div class="file-details">
                        <span class="file-size">${file.size_mb} MB</span>
                        <span class="file-samples">${file.samples.toLocaleString()} samples</span>
                        <span class="file-freq">${freqInfo}</span>
                    </div>
                    <div class="file-date">${new Date(file.modified).toLocaleDateString()}</div>
                </div>
            </div>
        `;
    });
    
    if (files.length > 10) {
        filesHTML += `<div class="more-files">... and ${files.length - 10} more files</div>`;
    }
    
    filesHTML += '</div>';
    filesList.innerHTML = filesHTML;
}

async function analyzeDataFile(filename) {
    try {
        showNotification(`Analyzing file: ${filename}`, 'info');
        
        const response = await fetch(`/api/data/file/${encodeURIComponent(filename)}`);
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                showFileAnalysis(result.data);
            } else {
                showNotification(`Analysis failed: ${result.message}`, 'error');
            }
        }
    } catch (error) {
        console.error('Error analyzing file:', error);
        showNotification('Error analyzing file', 'error');
    }
}

function showFileAnalysis(analysis) {
    // Create modal or panel to show file analysis
    const modalHTML = `
        <div class="analysis-modal" id="file-analysis-modal" style="
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: rgba(0,0,0,0.8); z-index: 1000; display: flex; 
            align-items: center; justify-content: center;
        ">
            <div class="analysis-content" style="
                background: #1a1a1a; border: 2px solid #00ff88; border-radius: 10px; 
                max-width: 800px; max-height: 80vh; overflow-y: auto; padding: 20px;
                color: #ffffff;
            ">
                <div class="analysis-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="color: #00ff88; margin: 0;"><i class="fas fa-chart-line"></i> File Analysis: ${analysis.filename}</h3>
                    <button onclick="closeFileAnalysis()" style="background: #ff4444; border: none; color: white; padding: 8px 12px; border-radius: 5px; cursor: pointer;">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="analysis-stats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div class="stat-box" style="background: #2a2a2a; padding: 15px; border-radius: 5px; border-left: 4px solid #00ff88;">
                        <div class="stat-label" style="color: #888; font-size: 12px;">File Size</div>
                        <div class="stat-value" style="color: #00ff88; font-size: 18px; font-weight: bold;">${analysis.size_mb} MB</div>
                    </div>
                    <div class="stat-box" style="background: #2a2a2a; padding: 15px; border-radius: 5px; border-left: 4px solid #00aaff;">
                        <div class="stat-label" style="color: #888; font-size: 12px;">Total Samples</div>
                        <div class="stat-value" style="color: #00aaff; font-size: 18px; font-weight: bold;">${analysis.samples.toLocaleString()}</div>
                    </div>
                    <div class="stat-box" style="background: #2a2a2a; padding: 15px; border-radius: 5px; border-left: 4px solid #ffaa00;">
                        <div class="stat-label" style="color: #888; font-size: 12px;">Columns</div>
                        <div class="stat-value" style="color: #ffaa00; font-size: 18px; font-weight: bold;">${analysis.columns.length}</div>
                    </div>
                </div>
                
                ${analysis.frequency_analysis && Object.keys(analysis.frequency_analysis).length > 0 ? `
                <div class="frequency-analysis" style="margin-bottom: 20px;">
                    <h4 style="color: #00aaff; margin-bottom: 10px;"><i class="fas fa-radio"></i> Frequency Analysis</h4>
                    <div style="background: #2a2a2a; padding: 15px; border-radius: 5px;">
                        <p><strong>Range:</strong> ${(analysis.frequency_analysis.min_hz / 1e6).toFixed(2)} - ${(analysis.frequency_analysis.max_hz / 1e6).toFixed(2)} MHz</p>
                        <p><strong>Center:</strong> ${(analysis.frequency_analysis.mean_hz / 1e6).toFixed(2)} MHz</p>
                        <p><strong>Unique Frequencies:</strong> ${analysis.frequency_analysis.unique_frequencies}</p>
                    </div>
                </div>
                ` : ''}
                
                ${analysis.signal_statistics && Object.keys(analysis.signal_statistics).length > 0 ? `
                <div class="signal-analysis" style="margin-bottom: 20px;">
                    <h4 style="color: #ffaa00; margin-bottom: 10px;"><i class="fas fa-signal"></i> Signal Statistics</h4>
                    <div style="background: #2a2a2a; padding: 15px; border-radius: 5px;">
                        <p><strong>Range:</strong> ${analysis.signal_statistics.min.toFixed(2)} to ${analysis.signal_statistics.max.toFixed(2)} dBm</p>
                        <p><strong>Average:</strong> ${analysis.signal_statistics.mean.toFixed(2)} dBm</p>
                        <p><strong>Std Dev:</strong> ${analysis.signal_statistics.std.toFixed(2)} dBm</p>
                    </div>
                </div>
                ` : ''}
                
                <div class="preview-data" style="margin-bottom: 20px;">
                    <h4 style="color: #ff88aa; margin-bottom: 10px;"><i class="fas fa-table"></i> Data Preview</h4>
                    <div style="background: #2a2a2a; padding: 15px; border-radius: 5px; max-height: 300px; overflow: auto;">
                        <table style="width: 100%; color: #ffffff; font-size: 12px; font-family: monospace;">
                            <thead>
                                <tr style="border-bottom: 1px solid #444;">
                                    ${analysis.columns.map(col => `<th style="padding: 5px; text-align: left; color: #00ff88;">${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${analysis.preview_data.slice(0, 5).map(row => 
                                    `<tr>${analysis.columns.map(col => `<td style="padding: 5px;">${typeof row[col] === 'number' ? row[col].toFixed(3) : row[col]}</td>`).join('')}</tr>`
                                ).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="analysis-actions" style="display: flex; gap: 10px; justify-content: center;">
                    <button onclick="useFileForTraining('${analysis.filename}')" style="
                        background: #00ff88; color: #000; border: none; padding: 10px 20px; 
                        border-radius: 5px; cursor: pointer; font-weight: bold;
                    ">
                        <i class="fas fa-brain"></i> Use for Training
                    </button>
                    <button onclick="exportFileAnalysis('${analysis.filename}')" style="
                        background: #00aaff; color: #fff; border: none; padding: 10px 20px; 
                        border-radius: 5px; cursor: pointer; font-weight: bold;
                    ">
                        <i class="fas fa-download"></i> Export Analysis
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    showNotification(`Analysis complete for ${analysis.filename}`, 'success');
}

function closeFileAnalysis() {
    const modal = document.getElementById('file-analysis-modal');
    if (modal) {
        modal.remove();
    }
}

function useFileForTraining(filename) {
    // Close analysis modal
    closeFileAnalysis();
    
    // Set the filename in training section and show training options
    showNotification(`Selected ${filename} for model training`, 'info');
    
    // Could set a hidden field or variable to remember the selected file
    window.selectedTrainingFile = filename;
    
    // Scroll to training section
    const trainingSection = document.querySelector('.training-panel');
    if (trainingSection) {
        trainingSection.scrollIntoView({ behavior: 'smooth' });
        // Highlight the training section briefly
        trainingSection.style.boxShadow = '0 0 20px #00ff88';
        setTimeout(() => {
            trainingSection.style.boxShadow = '';
        }, 3000);
    }
}

function exportFileAnalysis(filename) {
    // Close analysis modal
    closeFileAnalysis();
    showNotification(`Exporting analysis for ${filename}...`, 'info');
    // Implementation for exporting analysis data
}

async function showDatasetSelector() {
    try {
        const response = await fetch('/api/data/datasets');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                displayDatasetSelector(result.data.datasets);
            }
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
        showNotification('Error loading datasets', 'error');
    }
}

function displayDatasetSelector(datasets) {
    const modalHTML = `
        <div class="dataset-modal" id="dataset-selector-modal" style="
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: rgba(0,0,0,0.8); z-index: 1000; display: flex; 
            align-items: center; justify-content: center;
        ">
            <div class="dataset-content" style="
                background: #1a1a1a; border: 2px solid #00aaff; border-radius: 10px; 
                max-width: 900px; max-height: 80vh; overflow-y: auto; padding: 20px;
                color: #ffffff;
            ">
                <div class="dataset-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="color: #00aaff; margin: 0;"><i class="fas fa-database"></i> Available Datasets</h3>
                    <button onclick="closeDatasetSelector()" style="background: #ff4444; border: none; color: white; padding: 8px 12px; border-radius: 5px; cursor: pointer;">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                
                <div class="datasets-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    ${datasets.map(dataset => `
                        <div class="dataset-item" onclick="selectDataset('${dataset.name}', '${dataset.type}')" style="
                            background: #2a2a2a; border: 1px solid #444; border-radius: 8px; padding: 15px; 
                            cursor: pointer; transition: all 0.3s;
                        " onmouseover="this.style.borderColor='#00aaff'" onmouseout="this.style.borderColor='#444'">
                            <div class="dataset-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h4 style="color: #00aaff; margin: 0; font-size: 14px;">${dataset.name}</h4>
                                <span class="dataset-type" style="
                                    background: ${dataset.type === 'directory' ? '#00ff88' : dataset.type === 'file' ? '#ffaa00' : '#ff88aa'}; 
                                    color: #000; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;
                                ">${dataset.type}</span>
                            </div>
                            <div class="dataset-info" style="font-size: 12px; color: #ccc;">
                                <div style="margin-bottom: 5px;"><i class="fas fa-files-o"></i> ${dataset.file_count} files (${dataset.csv_files} CSV, ${dataset.json_files} JSON)</div>
                                <div style="margin-bottom: 5px;"><i class="fas fa-hdd"></i> ${dataset.size_mb.toFixed(1)} MB</div>
                                <div style="color: #888; font-style: italic;">${dataset.description}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div style="margin-top: 20px; text-align: center; color: #888; font-size: 12px;">
                    Click on a dataset to select it for model training
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function closeDatasetSelector() {
    const modal = document.getElementById('dataset-selector-modal');
    if (modal) {
        modal.remove();
    }
}

function selectDataset(datasetName, datasetType) {
    closeDatasetSelector();
    window.selectedDataset = { name: datasetName, type: datasetType };
    showNotification(`Selected dataset: ${datasetName}`, 'success');
    
    // Update training interface to show selected dataset
    const trainingSection = document.querySelector('.training-panel');
    if (trainingSection) {
        trainingSection.scrollIntoView({ behavior: 'smooth' });
        trainingSection.style.boxShadow = '0 0 20px #00aaff';
        setTimeout(() => {
            trainingSection.style.boxShadow = '';
        }, 3000);
    }
    
    // Update dataset display in training section if it exists
    updateTrainingDatasetDisplay(datasetName, datasetType);
}

function updateTrainingDatasetDisplay(datasetName, datasetType) {
    // Check if there's a dataset display area in the training section
    let datasetDisplay = document.getElementById('selected-dataset-display');
    if (!datasetDisplay) {
        // Create dataset display area in training section
        const trainingPanel = document.querySelector('.training-panel .panel-content');
        if (trainingPanel) {
            const datasetHTML = `
                <div id="selected-dataset-display" class="selected-dataset" style="
                    background: #2a2a2a; border: 1px solid #00aaff; border-radius: 5px; 
                    padding: 10px; margin-bottom: 15px; display: none;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <i class="fas fa-database" style="color: #00aaff;"></i>
                            <span style="color: #00aaff; font-weight: bold; margin-left: 5px;">Selected Dataset:</span>
                            <span id="dataset-name-display" style="color: #ffffff; margin-left: 10px;"></span>
                            <span id="dataset-type-display" style="
                                background: #00aaff; color: #000; padding: 2px 6px; 
                                border-radius: 3px; font-size: 10px; margin-left: 10px;
                            "></span>
                        </div>
                        <button onclick="clearSelectedDataset()" style="
                            background: transparent; border: 1px solid #ff4444; color: #ff4444; 
                            padding: 2px 6px; border-radius: 3px; cursor: pointer; font-size: 10px;
                        ">Clear</button>
                    </div>
                </div>
            `;
            trainingPanel.insertAdjacentHTML('afterbegin', datasetHTML);
            datasetDisplay = document.getElementById('selected-dataset-display');
        }
    }
    
    if (datasetDisplay) {
        document.getElementById('dataset-name-display').textContent = datasetName;
        document.getElementById('dataset-type-display').textContent = datasetType;
        datasetDisplay.style.display = 'block';
    }
}

function clearSelectedDataset() {
    window.selectedDataset = null;
    const datasetDisplay = document.getElementById('selected-dataset-display');
    if (datasetDisplay) {
        datasetDisplay.style.display = 'none';
    }
    showNotification('Dataset selection cleared', 'info');
}

// WebSocket Event Handlers (called from websocket.js)
function handleWebSocketMessage(data) {
    if (data.type === 'training_progress') {
        updateTrainingStatus({
            active: true,
            progress: data.progress,
            status: data.message
        });
    } else if (data.type === 'monitoring_update') {
        // Handle monitoring updates
        console.log('Monitoring update:', data);
    } else if (data.type === 'data_stats') {
        updateDataStats(data.stats);
    } else if (data.type === 'multiband_monitoring_started') {
        showNotification(`Multi-band monitoring started for ${data.data.bands.length} bands`, 'success');
    } else if (data.type === 'advanced_monitoring_started') {
        showNotification('Advanced monitoring started with filtering', 'success');
    } else if (data.type === 'mqtt_connected') {
        showNotification(`MQTT connected to ${data.data.broker}`, 'success');
        loadMQTTStatus();
    } else if (data.type === 'mqtt_disconnected') {
        showNotification('MQTT disconnected', 'info');
        loadMQTTStatus();
    } else if (data.type === 'mqtt_test_sent') {
        showNotification('MQTT test message sent', 'success');
    } else if (data.type === 'anomaly_detected') {
        handleAnomalyDetection(data.data);
    } else if (data.type === 'system_status_update') {
        updateSystemStatus(data.data);
    }
}

function handleAnomalyDetection(anomaly) {
    // Update anomaly counters
    const severity = anomaly.severity || 'medium';
    const countElement = document.getElementById(`${severity}-count`);
    if (countElement) {
        const current = parseInt(countElement.textContent) || 0;
        countElement.textContent = current + 1;
    }
    
    // Add to alerts list
    addAlert(anomaly);
    
    // Update threat status indicator
    updateStatusIndicator('threat-status', true);
    
    // Show notification
    showNotification(`${severity.toUpperCase()} threat detected at ${anomaly.frequency} MHz`, 'warning');
}

function addAlert(anomaly) {
    const alertsList = document.getElementById('alerts-list');
    if (!alertsList) return;
    
    // Remove "no alerts" message if present
    const noAlerts = alertsList.querySelector('.no-alerts');
    if (noAlerts) {
        noAlerts.remove();
    }
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${anomaly.severity || 'medium'}`;
    alertElement.innerHTML = `
        <div class="alert-header">
            <span class="alert-time">${new Date().toLocaleTimeString()}</span>
            <span class="alert-severity">${(anomaly.severity || 'medium').toUpperCase()}</span>
        </div>
        <div class="alert-content">
            <div class="alert-frequency">${anomaly.frequency} MHz</div>
            <div class="alert-confidence">Confidence: ${(anomaly.confidence * 100).toFixed(1)}%</div>
        </div>
    `;
    
    // Insert at the top
    alertsList.insertBefore(alertElement, alertsList.firstChild);
    
    // Keep only the last 10 alerts
    const alerts = alertsList.querySelectorAll('.alert');
    if (alerts.length > 10) {
        alerts[alerts.length - 1].remove();
    }
}

// Spectrum analysis functions
function updateSpectrumData(data) {
    // Update spectrum statistics
    if (data.peak_frequency) {
        updateElementText('peak-frequency', data.peak_frequency + ' MHz');
    }
    if (data.avg_power) {
        updateElementText('avg-power', data.avg_power.toFixed(1) + ' dBm');
    }
    if (data.snr) {
        updateElementText('snr-value', data.snr.toFixed(1) + ' dB');
    }
}

async function loadSpectrumAnalysis() {
    try {
        // Get the current frequency range from the UI if available
        const frequencyRange = document.getElementById('frequency-range-config')?.value || '144-146';
        
        const response = await fetch(`/api/spectrum/analysis?frequency_range=${encodeURIComponent(frequencyRange)}`);
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.analysis) {
                updateSpectrumData(result.analysis);
                updateSpectrumChart(result.analysis);
            }
        }
    } catch (error) {
        console.error('Error loading spectrum analysis:', error);
    }
}

function startSpectrumRefresh() {
    // Refresh spectrum analysis every 5 seconds
    setInterval(loadSpectrumAnalysis, 5000);
}

function updateSpectrumChart(data) {
    const canvas = document.getElementById('spectrum-chart');
    if (!canvas || !data) return;
    
    // Create or update the spectrum chart
    if (window.spectrumChart) {
        window.spectrumChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    // Prepare data for chart
    let frequencies = [];
    let amplitudes = [];
    
    if (data.frequencies && data.amplitudes) {
        frequencies = data.frequencies;
        amplitudes = data.amplitudes;
    } else if (data.frequency_data && data.amplitude_data) {
        frequencies = data.frequency_data;
        amplitudes = data.amplitude_data;
    } else {
        // Generate sample data for visualization
        const centerFreq = data.peak_frequency || 146.0;
        const span = 2.0; // 2 MHz span
        for (let i = 0; i < 100; i++) {
            const freq = centerFreq - span/2 + (i * span / 100);
            frequencies.push(freq);
            // Create a synthetic spectrum with peak at center
            const amplitude = -40 + 20 * Math.exp(-Math.pow((freq - centerFreq) / 0.2, 2)) + Math.random() * 5;
            amplitudes.push(amplitude);
        }
    }
    
    window.spectrumChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: frequencies.map(f => f.toFixed(2)),
            datasets: [{
                label: 'Power Spectrum',
                data: amplitudes,
                borderColor: 'rgba(0, 255, 255, 1)',
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#00ffff'
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Frequency (MHz)',
                        color: '#00ffff'
                    },
                    ticks: {
                        color: '#00ffff',
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: 'rgba(0, 255, 255, 0.2)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Power (dBm)',
                        color: '#00ffff'
                    },
                    ticks: {
                        color: '#00ffff'
                    },
                    grid: {
                        color: 'rgba(0, 255, 255, 0.2)'
                    }
                }
            }
        }
    });
}

// Model performance functions
async function loadModelStatus() {
    try {
        const response = await fetch('/api/model/status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success' && result.data) {
                updateModelInfo(result.data);
                updateModelPerformanceChart(result.data.performance_metrics || {});
            }
        }
    } catch (error) {
        console.error('Error loading model status:', error);
    }
}

function startModelStatusRefresh() {
    setInterval(loadModelStatus, 30000); // Refresh every 30 seconds
}

function updateModelInfo(info) {
    if (info.accuracy !== undefined) {
        updateElementText('model-accuracy', (info.accuracy * 100).toFixed(1) + '%');
    }
    if (info.confidence !== undefined) {
        updateElementText('model-confidence', (info.confidence * 100).toFixed(1) + '%');
    }
    if (info.last_trained) {
        const date = new Date(info.last_trained);
        updateElementText('last-trained', date.toLocaleDateString() + ' ' + date.toLocaleTimeString());
    }
    if (info.sample_count !== undefined) {
        updateElementText('sample-count', info.sample_count.toLocaleString());
    }
    
    // Update model type and deployment status
    if (info.model_type) {
        updateElementText('model-type', info.model_type);
    }
    if (info.is_deployed !== undefined) {
        const deployStatus = info.is_deployed ? 'DEPLOYED' : 'NOT DEPLOYED';
        updateElementText('deployment-status', deployStatus);
    }
}

function updateModelPerformanceChart(metrics) {
    const canvas = document.getElementById('model-performance-chart');
    if (!canvas || !metrics) return;
    
    // Create or update the performance chart
    if (window.modelChart) {
        window.modelChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    window.modelChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Precision', 'Recall', 'F1-Score', 'AUC', 'Accuracy'],
            datasets: [{
                label: 'Model Performance',
                data: [
                    (metrics.precision || 0) * 100,
                    (metrics.recall || 0) * 100,
                    (metrics.f1_score || 0) * 100,
                    (metrics.auc_score || 0) * 100,
                    (metrics.accuracy || 0) * 100
                ],
                backgroundColor: 'rgba(0, 255, 255, 0.2)',
                borderColor: 'rgba(0, 255, 255, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(0, 255, 255, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(0, 255, 255, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#00ffff'
                    }
                }
            },
            scales: {
                r: {
                    angleLines: {
                        color: 'rgba(0, 255, 255, 0.3)'
                    },
                    grid: {
                        color: 'rgba(0, 255, 255, 0.3)'
                    },
                    pointLabels: {
                        color: '#00ffff'
                    },
                    ticks: {
                        color: '#00ffff',
                        backdropColor: 'transparent'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// Initialize dashboard features
function initializeDashboard() {
    console.log('Initializing dashboard features...');
    
    // Load saved configuration
    loadConfig();
    
    // Set up slider event listeners
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const valueId = slider.id + '-value';
        slider.addEventListener('input', () => {
            updateSliderValue(slider.id, valueId);
        });
        // Initialize display
        updateSliderValue(slider.id, valueId);
    });
    
    // Set up auto-save for configuration
    const configInputs = document.querySelectorAll('#threshold-slider, #frequency-range-config, #scan-interval, #strict-mode');
    configInputs.forEach(input => {
        input.addEventListener('change', saveConfig);
    });
    
    // Add specific handler for frequency range changes to refresh spectrum
    const frequencyRangeInput = document.getElementById('frequency-range-config');
    if (frequencyRangeInput) {
        frequencyRangeInput.addEventListener('change', function() {
            console.log('Frequency range changed to:', this.value);
            // Refresh spectrum analysis with new frequency range
            loadSpectrumAnalysis();
        });
    }
    
    console.log('Dashboard initialization complete');
}

// Tab system functions
function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Set active button
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach((btn, index) => {
        if (btn.textContent.toLowerCase().includes(tabId.split('-')[0])) {
            btn.classList.add('active');
        }
    });
}

// WebSocket Message Handler
function handleWebSocketMessage(data) {
    console.log('Received WebSocket message:', data);
    
    switch (data.type) {
        case 'anomaly_detected':
            updateAnomalyCount();
            loadRecentAnomalies();
            showNotification(`Anomaly detected: ${data.description || 'Unknown'}`, 'warning');
            break;
            
        case 'system_status':
            updateSystemStatus(data.data);
            break;
            
        case 'monitoring_update':
            loadMonitoringStatus();
            break;
            
        case 'training_complete':
            showNotification('Model training completed successfully', 'success');
            break;
            
        case 'training_failed':
            showNotification('Model training failed', 'error');
            break;
            
        default:
            console.log('Unknown WebSocket message type:', data.type);
    }
}

// Notification System
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to page
    const container = document.getElementById('notifications') || document.body;
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (dataStatsInterval) {
        clearInterval(dataStatsInterval);
    }
    if (systemStatsInterval) {
        clearInterval(systemStatsInterval);
    }
    if (websocket) {
        websocket.close();
    }
});

// Real-time Analytics Functions
async function loadAnalytics() {
    try {
        const timeframe = document.getElementById('analytics-timeframe')?.value || '24h';
        const response = await fetch(`/api/analytics/realtime?timeframe=${timeframe}`);
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success' && result.data) {
                updateAnalyticsMetrics(result.data);
                updateAnalyticsCharts(result.data);
            }
        }
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

function startAnalyticsRefresh() {
    setInterval(loadAnalytics, 15000); // Refresh every 15 seconds for real-time feel
}

function refreshAnalytics() {
    loadAnalytics();
}

function updateAnalyticsMetrics(data) {
    // Update detection metrics
    if (data.detections !== undefined) {
        updateElementText('detections-count', data.detections.toLocaleString());
        const change = data.detections_change || 0;
        updateChangeIndicator('detections-change', change);
    }
    
    // Update anomaly metrics
    if (data.anomalies !== undefined) {
        updateElementText('anomalies-count', data.anomalies.toLocaleString());
        const change = data.anomalies_change || 0;
        updateChangeIndicator('anomalies-change', change);
    }
    
    // Update signal metrics
    if (data.avg_signal !== undefined) {
        updateElementText('avg-signal', data.avg_signal.toFixed(1) + ' dBm');
        const change = data.signal_change || 0;
        updateChangeIndicator('signal-change', change);
    }
    
    // Update accuracy metrics
    if (data.realtime_accuracy !== undefined) {
        updateElementText('realtime-accuracy', (data.realtime_accuracy * 100).toFixed(1) + '%');
        const change = data.accuracy_change || 0;
        updateChangeIndicator('accuracy-change', change);
    }
}

function updateChangeIndicator(elementId, change) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const absChange = Math.abs(change);
    const sign = change >= 0 ? '+' : '-';
    const color = change >= 0 ? '#00ff00' : '#ff0044';
    const icon = change >= 0 ? '↑' : '↓';
    
    element.textContent = `${icon} ${sign}${absChange.toFixed(1)}%`;
    element.style.color = color;
}

function updateAnalyticsCharts(data) {
    updateDetectionsTimelineChart(data.timeline || []);
    updateFrequencyDistributionChart(data.frequency_distribution || []);
}

function updateDetectionsTimelineChart(timelineData) {
    const canvas = document.getElementById('detections-timeline-chart');
    if (!canvas) return;
    
    if (window.detectionsChart) {
        window.detectionsChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    window.detectionsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timelineData.map(d => new Date(d.timestamp).toLocaleTimeString()),
            datasets: [{
                label: 'Detections',
                data: timelineData.map(d => d.detections),
                borderColor: 'rgba(0, 255, 255, 1)',
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                borderWidth: 2,
                fill: true
            }, {
                label: 'Anomalies',
                data: timelineData.map(d => d.anomalies),
                borderColor: 'rgba(255, 0, 68, 1)',
                backgroundColor: 'rgba(255, 0, 68, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#00ffff' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#00ffff' },
                    grid: { color: 'rgba(0, 255, 255, 0.3)' }
                },
                y: {
                    ticks: { color: '#00ffff' },
                    grid: { color: 'rgba(0, 255, 255, 0.3)' }
                }
            }
        }
    });
}

function updateFrequencyDistributionChart(frequencyData) {
    const canvas = document.getElementById('frequency-distribution-chart');
    if (!canvas) return;
    
    if (window.frequencyChart) {
        window.frequencyChart.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    window.frequencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: frequencyData.map(d => `${(d.frequency / 1e6).toFixed(1)} MHz`),
            datasets: [{
                label: 'Signal Strength',
                data: frequencyData.map(d => d.power),
                backgroundColor: 'rgba(0, 255, 255, 0.6)',
                borderColor: 'rgba(0, 255, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#00ffff' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#00ffff' },
                    grid: { color: 'rgba(0, 255, 255, 0.3)' }
                },
                y: {
                    ticks: { color: '#00ffff' },
                    grid: { color: 'rgba(0, 255, 255, 0.3)' }
                }
            }
        }
    });
}
