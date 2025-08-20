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
    loadSystemStatus();
    loadMQTTStatus();
    startDataStatsRefresh();
    startSystemStatsRefresh();
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
});

// WebSocket connection management
function initializeWebSocket() {
    console.log('Initializing WebSocket connection...');
    // WebSocket initialization is handled in websocket.js
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
    const startBtn = document.getElementById('startMonitoring');
    const stopBtn = document.getElementById('stopMonitoring');
    
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
        
        const response = await fetch('/api/data/collect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Data collection started:', result);
            alert('Data collection started successfully!');
            
            // Refresh stats after a short delay
            setTimeout(loadDataStats, 2000);
        } else {
            const error = await response.text();
            console.error('Failed to start data collection:', error);
            alert('Failed to start data collection: ' + error);
        }
    } catch (error) {
        console.error('Error starting data collection:', error);
        alert('Error starting data collection. Check console for details.');
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
function updateModelInfo(info) {
    if (info.accuracy) {
        updateElementText('model-accuracy', (info.accuracy * 100).toFixed(1) + '%');
    }
    if (info.confidence) {
        updateElementText('model-confidence', (info.confidence * 100).toFixed(1) + '%');
    }
    if (info.last_trained) {
        updateElementText('last-trained', new Date(info.last_trained).toLocaleDateString());
    }
    if (info.sample_count) {
        updateElementText('sample-count', info.sample_count.toLocaleString());
    }
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

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (dataStatsInterval) {
        clearInterval(dataStatsInterval);
    }
    if (systemStatsInterval) {
        clearInterval(systemStatsInterval);
    }
});
