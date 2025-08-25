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
    loadTrainingStatus();
    loadModelStatus();
    loadAnalytics();
    loadSystemStatus();
    loadMQTTStatus();
    loadRecentAnomalies();
    startDataStatsRefresh();
    startSystemStatsRefresh();
    startModelStatusRefresh();
    startAnalyticsRefresh();
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
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
    // This will use the existing trainModel function
    trainModel();
}

// Monitoring Control Functions
async function startMonitoring() {
    if (currentMonitoring) {
        console.log('Monitoring already in progress');
        return;
    }

    try {
        const response = await fetch('/api/monitoring/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            currentMonitoring = true;
            updateMonitoringUI(true);
            console.log('Monitoring started successfully');
        } else {
            console.error('Failed to start monitoring:', response.statusText);
            alert('Failed to start monitoring. Check console for details.');
        }
    } catch (error) {
        console.error('Error starting monitoring:', error);
        alert('Error starting monitoring. Check console for details.');
    }
}

async function stopMonitoring() {
    if (!currentMonitoring) {
        console.log('Monitoring not currently active');
        return;
    }

    try {
        const response = await fetch('/api/monitoring/stop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            currentMonitoring = false;
            updateMonitoringUI(false);
            console.log('Monitoring stopped successfully');
        } else {
            console.error('Failed to stop monitoring:', response.statusText);
            alert('Failed to stop monitoring. Check console for details.');
        }
    } catch (error) {
        console.error('Error stopping monitoring:', error);
        alert('Error stopping monitoring. Check console for details.');
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
        return;
    }

    try {
        // Get configuration values
        const config = {
            epochs: parseInt(document.getElementById('epochs')?.value || 100),
            batch_size: parseInt(document.getElementById('batchSize')?.value || 32),
            learning_rate: parseFloat(document.getElementById('learningRate')?.value || 0.001),
            validation_split: parseFloat(document.getElementById('validationSplit')?.value || 0.2),
            save_model: true
        };

        console.log('Starting model training with config:', config);
        
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
            
            // Start polling for training status
            startTrainingStatusPolling();
        } else {
            const error = await response.text();
            console.error('Failed to start training:', error);
            alert('Failed to start training: ' + error);
        }
    } catch (error) {
        console.error('Error starting training:', error);
        alert('Error starting training. Check console for details.');
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
    // Update progress bar
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
            if (result.status === 'success' && result.data) {
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
    updateStatValue('total-samples', stats.total_samples || 0);
    updateStatValue('training-samples', stats.total_files || 0);
    updateStatValue('test-samples', Math.floor((stats.total_samples || 0) * 0.2));
    updateStatValue('anomaly-samples', Math.floor((stats.total_samples || 0) * 0.1));
}

function updateStatValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value.toLocaleString();
    }
}

function startDataStatsRefresh() {
    // Refresh data stats every 30 seconds
    dataStatsInterval = setInterval(loadDataStats, 30000);
}

async function collectData() {
    try {
        console.log('Starting data collection...');
        showNotification('Starting data collection...', 'info');
        
        const response = await fetch('/api/data/collect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Data collection result:', result);
            
            if (result.status === 'success') {
                showNotification(`Data collection completed: ${result.message}`, 'success');
                
                // Show collection details if available
                if (result.data) {
                    const details = `
                        Frequency: ${(result.data.frequency_hz / 1e6).toFixed(2)} MHz
                        Duration: ${result.data.duration_seconds}s
                        Samples: ${result.data.sample_count}
                        Timestamp: ${new Date(result.data.timestamp).toLocaleString()}
                    `;
                    console.log('Collection details:', details);
                }
                
                // Refresh stats after a short delay
                setTimeout(loadDataStats, 2000);
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
