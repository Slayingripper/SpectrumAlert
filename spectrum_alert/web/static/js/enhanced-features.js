// Enhanced Features for SpectrumAlert Dashboard
// This file contains additional functionality for data management, visualization, and training

// Training Configuration Setup
function setupTrainingConfigSliders() {
    // Setup slider value display updates
    const sliders = [
        { id: 'contamination', valueId: 'contamination-value' },
        { id: 'validation-split', valueId: 'validation-split-value' }
    ];
    
    sliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);
        
        if (element && valueElement) {
            // Set initial value
            valueElement.textContent = element.value;
            
            // Update value on change
            element.addEventListener('input', function() {
                valueElement.textContent = this.value;
            });
        }
    });
}

// Real-time Spectrum Visualization
let spectrumChart = null;
let spectrumUpdateInterval = null;

function startSpectrumVisualization() {
    if (spectrumUpdateInterval) {
        stopSpectrumVisualization();
    }
    
    // Create spectrum visualization if it doesn't exist
    createSpectrumVisualization();
    
    // Start polling for spectrum data
    spectrumUpdateInterval = setInterval(updateSpectrumVisualization, 1000); // Update every second
    showNotification('📊 Real-time spectrum visualization started', 'info');
    
    // Update button states
    const startBtn = document.getElementById('start-spectrum-viz');
    const stopBtn = document.getElementById('stop-spectrum-viz');
    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'inline-block';
}

function stopSpectrumVisualization() {
    if (spectrumUpdateInterval) {
        clearInterval(spectrumUpdateInterval);
        spectrumUpdateInterval = null;
        showNotification('📊 Real-time spectrum visualization stopped', 'info');
    }
    
    // Update button states
    const startBtn = document.getElementById('start-spectrum-viz');
    const stopBtn = document.getElementById('stop-spectrum-viz');
    if (startBtn) startBtn.style.display = 'inline-block';
    if (stopBtn) stopBtn.style.display = 'none';
}

function createSpectrumVisualization() {
    // Check if we need to create the spectrum visualization panel
    let spectrumPanel = document.getElementById('spectrum-visualization-panel');
    if (!spectrumPanel) {
        // Create the spectrum panel after the monitoring panel
        const monitoringPanel = document.querySelector('.monitoring-panel');
        if (monitoringPanel) {
            const spectrumHTML = `
                <section class="panel spectrum-panel" id="spectrum-visualization-panel">
                    <div class="panel-header">
                        <h2><i class="fas fa-chart-line"></i> REAL-TIME SPECTRUM</h2>
                        <div class="panel-controls">
                            <button class="btn-start" onclick="startSpectrumVisualization()" id="start-spectrum-viz">
                                <i class="fas fa-play"></i>
                                START VIZ
                            </button>
                            <button class="btn-stop" onclick="stopSpectrumVisualization()" id="stop-spectrum-viz" style="display: none;">
                                <i class="fas fa-stop"></i>
                                STOP VIZ
                            </button>
                            <select class="control-select" id="spectrum-frequency">
                                <option value="433">433 MHz</option>
                                <option value="868">868 MHz</option>
                                <option value="915">915 MHz</option>
                                <option value="2400">2.4 GHz</option>
                            </select>
                        </div>
                    </div>
                    <div class="panel-content">
                        <div class="spectrum-display">
                            <canvas id="spectrum-chart" width="800" height="300"></canvas>
                        </div>
                        <div class="spectrum-stats" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                            <div class="stat-item">
                                <div class="stat-label">Peak Power</div>
                                <div class="stat-value" id="spectrum-peak">-- dBm</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Avg Power</div>
                                <div class="stat-value" id="spectrum-avg">-- dBm</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Peak Freq</div>
                                <div class="stat-value" id="spectrum-freq">-- MHz</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Bandwidth</div>
                                <div class="stat-value" id="spectrum-bw">-- MHz</div>
                            </div>
                        </div>
                    </div>
                </section>
            `;
            monitoringPanel.insertAdjacentHTML('afterend', spectrumHTML);
        }
    }
    
    // Initialize the chart if Chart.js is available
    const canvas = document.getElementById('spectrum-chart');
    if (canvas && !spectrumChart && typeof Chart !== 'undefined') {
        const ctx = canvas.getContext('2d');
        spectrumChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Frequency labels
                datasets: [{
                    label: 'Signal Power (dBm)',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#00ff88' }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Frequency (MHz)',
                            color: '#00ff88'
                        },
                        ticks: { color: '#00ffff' },
                        grid: { color: 'rgba(0, 255, 255, 0.3)' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Power (dBm)',
                            color: '#00ff88'
                        },
                        ticks: { color: '#00ffff' },
                        grid: { color: 'rgba(0, 255, 255, 0.3)' }
                    }
                },
                animation: {
                    duration: 200
                }
            }
        });
    }
}

async function updateSpectrumVisualization() {
    if (!spectrumChart) return;
    
    try {
        // Get current spectrum data
        const response = await fetch('/api/spectrum/current');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                updateSpectrumChart(result.data);
                return;
            }
        }
        
        // Fall back to mock data if API not available
        generateMockSpectrumData();
        
    } catch (error) {
        console.error('Error updating spectrum visualization:', error);
        // Generate mock data if API fails
        generateMockSpectrumData();
    }
}

function generateMockSpectrumData() {
    if (!spectrumChart) return;
    
    const centerFreq = parseFloat(document.getElementById('spectrum-frequency')?.value || 433);
    const span = 10; // 10 MHz span
    const points = 100;
    
    const frequencies = [];
    const powers = [];
    
    for (let i = 0; i < points; i++) {
        const freq = centerFreq - span/2 + (span * i / points);
        frequencies.push(freq.toFixed(2));
        
        // Generate realistic spectrum with noise floor and some signals
        let power = -80 + Math.random() * 10; // Noise floor around -80 dBm
        
        // Add some signals at specific frequencies
        const signals = [centerFreq - 2, centerFreq + 1.5, centerFreq + 3];
        signals.forEach(sigFreq => {
            const dist = Math.abs(freq - sigFreq);
            if (dist < 0.5) {
                power += 30 * Math.exp(-dist * 10); // Strong signal
            }
        });
        
        powers.push(power);
    }
    
    // Update chart data
    spectrumChart.data.labels = frequencies;
    spectrumChart.data.datasets[0].data = powers;
    spectrumChart.update('none');
    
    // Update statistics
    const peakPower = Math.max(...powers);
    const avgPower = powers.reduce((a, b) => a + b, 0) / powers.length;
    const peakIndex = powers.indexOf(peakPower);
    const peakFreq = frequencies[peakIndex];
    
    if (typeof updateElementContent === 'function') {
        updateElementContent('spectrum-peak', `${peakPower.toFixed(1)} dBm`);
        updateElementContent('spectrum-avg', `${avgPower.toFixed(1)} dBm`);
        updateElementContent('spectrum-freq', `${peakFreq} MHz`);
        updateElementContent('spectrum-bw', `${span} MHz`);
    }
}

function updateSpectrumChart(spectrumData) {
    if (!spectrumChart || !spectrumData.frequencies || !spectrumData.powers) return;
    
    spectrumChart.data.labels = spectrumData.frequencies.map(f => (f / 1e6).toFixed(2));
    spectrumChart.data.datasets[0].data = spectrumData.powers;
    spectrumChart.update('none');
    
    // Update statistics
    const peakPower = Math.max(...spectrumData.powers);
    const avgPower = spectrumData.powers.reduce((a, b) => a + b, 0) / spectrumData.powers.length;
    const peakIndex = spectrumData.powers.indexOf(peakPower);
    const peakFreq = spectrumData.frequencies[peakIndex] / 1e6;
    
    if (typeof updateElementContent === 'function') {
        updateElementContent('spectrum-peak', `${peakPower.toFixed(1)} dBm`);
        updateElementContent('spectrum-avg', `${avgPower.toFixed(1)} dBm`);
        updateElementContent('spectrum-freq', `${peakFreq.toFixed(2)} MHz`);
        updateElementContent('spectrum-bw', `${((Math.max(...spectrumData.frequencies) - Math.min(...spectrumData.frequencies)) / 1e6).toFixed(1)} MHz`);
    }
}

// Enhanced monitoring status display
function displayMonitoringDetails(details) {
    // Create enhanced monitoring display if it doesn't exist
    let monitoringDetails = document.getElementById('monitoring-details-display');
    if (!monitoringDetails) {
        const monitoringPanel = document.querySelector('.monitoring-panel .panel-content');
        if (monitoringPanel) {
            const detailsHTML = `
                <div id="monitoring-details-display" class="monitoring-details" style="
                    background: #2a2a2a; border: 1px solid #00ff88; border-radius: 5px; 
                    padding: 15px; margin-top: 15px; display: none;
                ">
                    <div class="details-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="color: #00ff88; margin: 0;"><i class="fas fa-info-circle"></i> Monitoring Details</h4>
                        <button onclick="hideMonitoringDetails()" style="
                            background: transparent; border: 1px solid #888; color: #888; 
                            padding: 2px 6px; border-radius: 3px; cursor: pointer; font-size: 10px;
                        ">Hide</button>
                    </div>
                    
                    <div class="monitoring-metrics" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        <div class="metric-item">
                            <div style="color: #888; font-size: 11px;">Active Time</div>
                            <div id="monitoring-active-time" style="color: #00ff88; font-weight: bold;">--</div>
                        </div>
                        <div class="metric-item">
                            <div style="color: #888; font-size: 11px;">Samples Collected</div>
                            <div id="monitoring-samples" style="color: #00aaff; font-weight: bold;">--</div>
                        </div>
                        <div class="metric-item">
                            <div style="color: #888; font-size: 11px;">Data Rate</div>
                            <div id="monitoring-data-rate" style="color: #ffaa00; font-weight: bold;">--</div>
                        </div>
                        <div class="metric-item">
                            <div style="color: #888; font-size: 11px;">Frequency</div>
                            <div id="monitoring-frequency" style="color: #ff88aa; font-weight: bold;">--</div>
                        </div>
                    </div>
                </div>
            `;
            monitoringPanel.insertAdjacentHTML('beforeend', detailsHTML);
            monitoringDetails = document.getElementById('monitoring-details-display');
        }
    }
    
    if (monitoringDetails && details) {
        if (typeof updateElementContent === 'function') {
            updateElementContent('monitoring-active-time', details.active_time || '--');
            updateElementContent('monitoring-samples', details.samples_collected || '--');
            updateElementContent('monitoring-data-rate', details.data_rate || '--');
            updateElementContent('monitoring-frequency', details.frequency || '--');
        }
        monitoringDetails.style.display = 'block';
    }
}

function hideMonitoringDetails() {
    const monitoringDetails = document.getElementById('monitoring-details-display');
    if (monitoringDetails) {
        monitoringDetails.style.display = 'none';
    }
}

// Enhanced notification system with different types
function showEnhancedNotification(message, type = 'info', duration = 5000, actions = null) {
    const container = document.getElementById('notifications') || document.body;
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const icons = {
        'info': 'fas fa-info-circle',
        'success': 'fas fa-check-circle',
        'warning': 'fas fa-exclamation-triangle',
        'error': 'fas fa-times-circle',
        'spectrum': 'fas fa-radio'
    };
    
    let actionsHTML = '';
    if (actions) {
        actionsHTML = `
            <div class="notification-actions" style="margin-top: 10px;">
                ${actions.map(action => `
                    <button onclick="${action.callback}" style="
                        background: ${action.color || '#00ff88'}; color: #000; border: none; 
                        padding: 5px 10px; margin-right: 5px; border-radius: 3px; cursor: pointer; font-size: 11px;
                    ">${action.label}</button>
                `).join('')}
            </div>
        `;
    }
    
    notification.innerHTML = `
        <div style="display: flex; align-items: flex-start; gap: 10px;">
            <i class="${icons[type] || icons.info}" style="color: ${type === 'error' ? '#ff4444' : type === 'warning' ? '#ffaa00' : type === 'success' ? '#00ff88' : '#00aaff'}; margin-top: 2px;"></i>
            <div style="flex: 1;">
                <div style="white-space: pre-line;">${message}</div>
                ${actionsHTML}
            </div>
            <button onclick="this.parentElement.parentElement.remove()" style="
                background: transparent; border: none; color: #888; cursor: pointer; font-size: 16px;
            ">&times;</button>
        </div>
    `;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: `${20 + container.children.length * 70}px`,
        right: '20px',
        background: '#1a1a1a',
        border: `2px solid ${type === 'error' ? '#ff4444' : type === 'warning' ? '#ffaa00' : type === 'success' ? '#00ff88' : '#00aaff'}`,
        borderRadius: '8px',
        padding: '15px',
        maxWidth: '400px',
        color: '#ffffff',
        fontSize: '12px',
        zIndex: '10000',
        boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
        animation: 'slideIn 0.3s ease-out'
    });
    
    container.appendChild(notification);
    
    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => notification.remove(), 300);
            }
        }, duration);
    }
}

// Add CSS animations for notifications
if (!document.getElementById('enhanced-styles')) {
    const style = document.createElement('style');
    style.id = 'enhanced-styles';
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        
        .files-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .file-item {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .file-item:hover {
            border-color: #00ff88;
            background: #3a3a3a;
        }
        
        .file-icon {
            color: #00aaff;
            font-size: 18px;
            width: 20px;
            text-align: center;
        }
        
        .file-info {
            flex: 1;
        }
        
        .file-name {
            color: #ffffff;
            font-weight: bold;
            font-size: 12px;
            margin-bottom: 3px;
        }
        
        .file-details {
            display: flex;
            gap: 10px;
            font-size: 10px;
            color: #888;
        }
        
        .file-date {
            font-size: 10px;
            color: #666;
        }
        
        .no-files {
            text-align: center;
            color: #888;
            padding: 20px;
            font-style: italic;
        }
        
        .more-files {
            text-align: center;
            color: #00aaff;
            padding: 10px;
            font-size: 11px;
            border: 1px dashed #444;
            border-radius: 5px;
        }
    `;
    document.head.appendChild(style);
}
