// Dashboard JavaScript for SpectrumAlert Web Interface

// Global state
let currentMonitoring = false;

// Chart.js bakes colours in at construction time, so canvas colours are read
// from the active theme instead of hard-coded. refreshChartColors() re-reads
// them and the charts are rebuilt when analytics next loads.
const chartColors = {
    accent: '#5b8dff',
    accentFill: 'rgba(91, 141, 255, 0.12)',
    accentBar: 'rgba(91, 141, 255, 0.7)',
    danger: '#ec6969',
    dangerFill: 'rgba(236, 105, 105, 0.12)',
    text: '#8391a5',
    grid: 'rgba(131, 145, 165, 0.16)'
};

/** Turn a hex colour into rgba so fills follow the active accent exactly. */
function withAlpha(hex, alpha) {
    const value = String(hex || '').trim();
    const m = value.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i);
    if (!m) return value || `rgba(91, 141, 255, ${alpha})`;
    return `rgba(${parseInt(m[1], 16)}, ${parseInt(m[2], 16)}, ${parseInt(m[3], 16)}, ${alpha})`;
}

function refreshChartColors() {
    const s = getComputedStyle(document.documentElement);
    const v = (name, fallback) => (s.getPropertyValue(name) || '').trim() || fallback;
    chartColors.accent = v('--accent', '#5b8dff');
    chartColors.danger = v('--danger', '#ec6969');
    chartColors.text = v('--text-muted', '#8391a5');
    chartColors.grid = v('--border-color', 'rgba(131, 145, 165, 0.16)');
    chartColors.surface = v('--bg-secondary', '#111821');
    chartColors.border = v('--border-color', '#243040');
    chartColors.textBright = v('--text-primary', '#e9eef6');
    chartColors.accentFill = withAlpha(chartColors.accent, 0.12);
    chartColors.accentBar = withAlpha(chartColors.accent, 0.7);
    chartColors.dangerFill = withAlpha(chartColors.danger, 0.12);

    // Global defaults keep tooltips, legends and axis labels on the same
    // typeface and size as the rest of the interface.
    if (window.Chart) {
        Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
        Chart.defaults.font.size = 11;
        Chart.defaults.color = chartColors.text;
        Chart.defaults.borderColor = chartColors.grid;
    }
}
refreshChartColors();

/** One tooltip treatment for every chart. */
function chartTooltip() {
    return {
        backgroundColor: chartColors.surface,
        titleColor: chartColors.textBright,
        bodyColor: chartColors.text,
        borderColor: chartColors.border,
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        boxPadding: 4
    };
}

/** Legend labels as short rules rather than filled boxes. */
function chartLegendLabels(color) {
    return {
        color: color || chartColors.text,
        usePointStyle: true,
        pointStyle: 'rectRounded',
        boxWidth: 9,
        boxHeight: 9,
        padding: 16
    };
}

/** Sentence-case a word for display. Severity values are lowercase in the data. */
function cap(word) {
    return String(word || '').charAt(0).toUpperCase() + String(word || '').slice(1);
}
let trainingInProgress = false;
let dataStatsInterval = null;
let systemStatsInterval = null;
let mqttConnected = false;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initializing...');
    setupTuningControls();
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
});

/**
 * Notifications go through the shared SAUI notifier when it is available, and
 * fall back to the console when it is not. Keeping the wrapper here means this
 * file does not depend on a global exported by another script.
 */
function notify(message, options) {
    if (window.SAUI && SAUI.notify) {
        return SAUI.notify.show(message, options);
    }
    console.log('[notify]', message, options);
    return null;
}

/**
 * Show a loading / empty / error state under the spectrum chart. Defined here
 * rather than in dashboard-extras because updateSpectrumChart is defined here.
 */
function setSpectrumState(kind, title, message, onRetry) {
    const node = document.getElementById('spectrum-state');
    if (!node) return;
    node.className = 'sa-state sa-state--' + (kind === 'ready' ? 'ready' : kind);
    node.hidden = kind === 'ready';
    if (kind === 'ready') { node.innerHTML = ''; return; }
    node.innerHTML = '';
    const titleNode = document.createElement('div');
    titleNode.className = 'sa-state-title';
    titleNode.textContent = title || kind;
    node.appendChild(titleNode);
    if (message) {
        const detail = document.createElement('div');
        detail.className = 'sa-state-detail';
        detail.textContent = message;
        node.appendChild(detail);
    }
    if (kind === 'error' && onRetry) {
        const retry = document.createElement('button');
        retry.className = 'sa-retry';
        retry.textContent = 'Retry';
        retry.addEventListener('click', onRetry);
        node.appendChild(retry);
    }
}

// WebSocket connection management
//
// There is exactly one socket. Previously dashboard.js and websocket.js each
// opened one, so two connections raced for the same endpoint and the second
// one's handlers all bailed out on a `window.dashboard` object that was never
// created. websocket.js is no longer loaded (see dashboard.html).
let websocket = null;
let wsRetryCount = 0;
let wsRetryTimer = null;
let connectionPill = null;

function initializeWebSocket() {
    if (websocket && (websocket.readyState === WebSocket.OPEN ||
                      websocket.readyState === WebSocket.CONNECTING)) {
        return; // already open or opening
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

    setConnectionState('connecting');

    try {
        websocket = new WebSocket(wsUrl);

        websocket.onopen = function() {
            wsRetryCount = 0;
            setConnectionState('live');
        };

        websocket.onmessage = function(event) {
            try {
                handleWebSocketMessage(JSON.parse(event.data));
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        websocket.onclose = function() {
            setConnectionState('down');
            // Exponential backoff with a cap. The previous version retried
            // every 5 s forever with no backoff and no cap, which hammered the
            // server while it was down.
            const delay = Math.min(30000, 1000 * Math.pow(2, wsRetryCount));
            wsRetryCount += 1;
            clearTimeout(wsRetryTimer);
            wsRetryTimer = setTimeout(initializeWebSocket, delay);
        };

        websocket.onerror = function() {
            // onclose always follows, which is where the retry is scheduled.
            setConnectionState('down');
        };
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        setConnectionState('down');
        clearTimeout(wsRetryTimer);
        wsRetryTimer = setTimeout(initializeWebSocket, 10000);
    }
}

function setConnectionState(next) {
    // The pill is created here on first use. It used to be created only in
    // dashboard-extras.js, which assigns a `connectionPill` of its own -- a
    // different binding. This module's writes therefore all went to a null
    // variable and were discarded, and the footer indicator sat on whatever
    // state the workspace controller last set. SAUI.connectionPill caches on
    // the element, so both controllers share one pill rather than fighting over
    // two.
    if (!connectionPill && window.SAUI) {
        var target = document.getElementById('sa-conn');
        if (target) connectionPill = window.SAUI.connectionPill(target);
    }
    if (connectionPill) {
        connectionPill.set(next);
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
    // Two shapes arrive here: the advanced-status fetch nests values under
    // `performance`, while the 5 s WebSocket broadcast sends them flat. The
    // flat form used to be ignored entirely, so pushed updates never moved
    // the cards. Normalise once.
    if (!status.performance && typeof status.cpu_usage === 'number') {
        status = Object.assign({}, status, {
            performance: {
                cpu_usage: status.cpu_usage,
                memory_usage: status.memory_usage,
                disk_usage: status.disk_usage,
                temperature: status.temperature,
                uptime: status.uptime
            }
        });
    }

    // Update CPU usage
    if (status.performance) {
        updateMetricValue('cpu-usage', status.performance.cpu_usage + '%');
        updateMetricBar('cpu-bar', status.performance.cpu_usage);

        updateMetricValue('memory-usage', status.performance.memory_usage + '%');
        updateMetricBar('memory-bar', status.performance.memory_usage);

        // Uptime is part of the same payload; the card sat at "--" forever
        // because nothing ever wrote it.
        if (status.performance.uptime) {
            updateMetricValue('uptime', status.performance.uptime);
        }

        // No sensor is a legitimate answer, especially on VMs. Say so instead
        // of leaving a bare "°C" or a stale default.
        const tempEl = document.getElementById('temperature');
        if (tempEl) {
            if (typeof status.performance.temperature === 'number' &&
                isFinite(status.performance.temperature)) {
                tempEl.textContent = Math.round(status.performance.temperature) + '°C';
                tempEl.title = '';
            } else {
                tempEl.textContent = 'N/A';
                tempEl.title = 'No temperature sensor on this platform';
            }
        }
    }

    // Update monitoring status
    if (status.monitoring) {
        const monitoringStatus = status.monitoring.active ? 'ACTIVE' : 'STOPPED';
        updateStatusIndicator('monitor-status', status.monitoring.active);
        updateElementText('monitoring-status', monitoringStatus);
    }

    // SDR status: colour by probe result and carry the full detail, so
    // "No RTL-SDR device detected" is visibly different from "Available".
    if (status.hardware) {
        updateElementText('sdr-status', status.hardware.sdr_status);
        const el = document.getElementById('sdr-status');
        if (el) {
            const ok = status.hardware.sdr_available === true;
            // --ok/--warn do not exist, so the old fallbacks were the only
            // colours that ever applied and #f59e0b is unreadable on the
            // light theme's white panel. Use the palette's own status tokens.
            el.style.color = ok ? 'var(--sa-ok)' : 'var(--sa-medium)';
            if (status.hardware.sdr_probe_status === 'no-device') {
                el.title = 'Attach a dongle or configure a remote rtl_tcp receiver on the Geolocation page';
            } else {
                el.title = '';
            }
        }
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
            element.style.backgroundColor = '#ef5350';
        } else if (percentage > 60) {
            element.style.backgroundColor = '#d9a441';
        } else {
            element.style.backgroundColor = '#4c8dff';
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
        let url = '/api/anomalies/recent';
        const win = window.currentMonitoredWindowMhz;
        if (Array.isArray(win) && win.length === 2) {
            url += '?min_hz=' + encodeURIComponent(win[0] * 1e6) +
                   '&max_hz=' + encodeURIComponent(win[1] * 1e6);
        }
        const response = await fetch(url);
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

/**
 * Render persisted anomalies into the alert list.
 *
 * This previously wrote into #recent-anomalies-list, an element that has never
 * existed, so it returned early every time and the Recent Alerts panel stayed
 * permanently empty. It now feeds the same alert buffer the live WebSocket path
 * uses, so history and live updates appear in one place and export together.
 */
function updateRecentAnomalies(anomalies) {
    if (!Array.isArray(anomalies)) return;

    const seen = new Set(
        (window.dashboardExtras ? window.dashboardExtras.alerts() : [])
            .map(a => a.id)
    );

    // Newest first, and skip anything already shown by the live path.
    anomalies.forEach((anomaly) => {
        const key = anomaly.id || (anomaly.timestamp + anomaly.frequency_hz);
        if (seen.has(key)) return;
        if (typeof window.recordAnomaly === 'function') {
            window.recordAnomaly({
                id: key,
                severity: anomaly.severity,
                frequency_mhz: anomaly.frequency_hz
                    ? anomaly.frequency_hz / 1e6 : undefined,
                frequency_hz: anomaly.frequency_hz,
                anomaly_score: anomaly.confidence_score,
                anomaly_type: anomaly.anomaly_type,
                description: anomaly.description,
                at: anomaly.timestamp ? new Date(anomaly.timestamp) : undefined
            });
        }
    });

    if (typeof window.dashboardExtras !== 'undefined') {
        window.dashboardExtras.renderAlerts();
    }
}

// Monitoring Status Functions
async function loadMonitoringStatus() {
    try {
        const response = await fetch('/api/monitoring/status');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.data) {
                // The endpoint sets monitoring_active. Reading .active here
                // made every freshly loaded page claim monitoring was off
                // while the backend kept scanning, so the stop button was
                // missing and the shortcut started a second run.
                const isActive = result.data.monitoring_active === true;
                currentMonitoring = isActive;
                updateMonitoringUI(isActive);
                updateMonitoringStatus(result.data);
                updateStatusIndicator('monitor-status', isActive);
                updateStatusIndicator('system-status', isActive);
                console.log(`Initial monitoring status: ${isActive ? 'ACTIVE' : 'INACTIVE'}`);
                if (isActive) {
                    showNotification('📡 Monitoring is currently active', 'success');
                    // Started by an earlier page or another client: follow it
                    // so the guide, the badge and the stop button stay true.
                    const winHz = result.data.monitored_window_hz;
                    if (Array.isArray(winHz) && winHz.length === 2) {
                        window.currentMonitoredWindowMhz = [winHz[0] / 1e6, winHz[1] / 1e6];
                        if (window.SAAlertsSetWindow) {
                            window.SAAlertsSetWindow(winHz[0], winHz[1]);
                        }
                        updateMonitorWindowReadout();
                    }
                    startMonitoringStatusPolling();
                }
            }
        }
    } catch (error) {
        console.error('Error loading monitoring status:', error);
    }
}

function updateMonitoringStatus(status) {
    // Update monitoring status indicators. Only #monitor-status exists in the
    // template; the old 'monitoring-status' / 'monitoring-state' ids were
    // dropped with the header redesign and silently did nothing.
    const monitoringActive = status.monitoring_active || false;
    updateStatusIndicator('monitor-status', monitoringActive);
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
            // Recording a configuration starts no capture, so the monitoring
            // state is left alone: only /api/monitoring/start may claim a run.
            console.log('Advanced configuration recorded:', result);
            showNotification('Advanced settings recorded. Capture is started from Spectrum analysis.', 'success');
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
            // Recording a configuration starts no capture; see above.
            console.log('Multi-band configuration recorded:', result);
            showNotification(`Multi-band settings recorded for ${bands.length} band(s). Capture is started from Spectrum analysis.`, 'success');
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
    // The backend stores the broker settings and raises a flag; it never opens
    // a socket or publishes. Saying "Connected" here would promise a link that
    // does not exist, so the label states what actually happened.
    updateElementText('mqtt-connection-status',
        status.connected ? 'Saved (no live link)' : 'Not saved');
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
            console.log('MQTT connect accepted:', result);
            showNotification('Broker settings saved. This build does not open a connection to the broker.',
                'success');
            await loadMQTTStatus();
        } else {
            const error = await response.text();
            console.error('Failed to save MQTT settings:', error);
            showNotification('Could not save broker settings', 'error');
        }
    } catch (error) {
        console.error('Error saving MQTT settings:', error);
        showNotification('Error saving broker settings', 'error');
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
            console.log('MQTT settings cleared:', result);
            showNotification('Broker settings cleared', 'success');
            await loadMQTTStatus();
        } else {
            const error = await response.text();
            console.error('Failed to clear MQTT settings:', error);
            showNotification('Could not clear broker settings', 'error');
        }
    } catch (error) {
        console.error('Error clearing MQTT settings:', error);
        showNotification('Error clearing broker settings', 'error');
    }
}

async function testMQTT() {
    try {
        if (!mqttConnected) {
            showNotification('Save the broker settings first', 'warning');
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
            console.log('MQTT test accepted:', result);
            showNotification('Test event recorded by the server. Nothing is published to a broker yet.',
                'success');
        } else {
            const error = await response.text();
            console.error('Failed to record MQTT test:', error);
            showNotification('Could not record the test event', 'error');
        }
    } catch (error) {
        console.error('Error recording MQTT test:', error);
        showNotification('Error recording the test event', 'error');
    }
}

// Notification System
//
// There were three definitions of this function. The last one won, created its
// own DOM with a flat 5 s dismiss, ignored the mute preference and bypassed the
// severity filter -- so an error toast could be silently suppressed by the same
// preference that suppressed informational ones. Everything now routes through
// SAUI.notify, which owns the DOM, timers, filtering, mute state and history.
function showNotification(message, type = 'info') {
    const severity = { error: 'error', warning: 'warning', success: 'success', info: 'info' }[type]
        || 'info';
    return notify(message, { severity: severity });
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
    showNotification('Settings saved in this browser. They are not sent to the server.', 'success');
}

function loadConfig() {
    try {
        const saved = localStorage.getItem('spectrumAlertConfig');
        if (saved) {
            const config = JSON.parse(saved);
            
            if (document.getElementById('threshold-slider')) {
                document.getElementById('threshold-slider').value = config.threshold || 0.8;
                updateSliderValue('threshold-slider', 'threshold-slider-value');
            }
            if (document.getElementById('frequency-range-config')) {
                document.getElementById('frequency-range-config').value = config.frequency_range || '88-108';
            }
            if (document.getElementById('scan-interval')) {
                document.getElementById('scan-interval').value = config.scan_interval || 5;
            }
            // Strict mode is compiled into the detector and always on; the box is
            // a read-only statement of that, so a stale saved value must not
            // leave it looking switchable.
            if (document.getElementById('strict-mode')) {
                document.getElementById('strict-mode').checked = true;
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
    // The visible list is rendered by dashboard-extras from its own buffer, so
    // confirmation comes first and the buffer is what actually gets emptied.
    const shown = document.querySelectorAll('#alerts-list .sa-alert, #alerts-list .alert').length;
    const message = shown
        ? 'Clear ' + shown + ' alert(s) from this list? The detection history ' +
          'on the server is not deleted.'
        : 'Clear the alert list?';
    if (!window.confirm(message)) return;

    if (typeof window.SAAlertsClear === 'function') {
        window.SAAlertsClear();
    } else {
        const alertsList = document.getElementById('alerts-list');
        if (alertsList) alertsList.innerHTML = '<div class="no-alerts">No recent alerts</div>';
    }
    showNotification('Alert list cleared', 'info');
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

// ---------------------------------------------------------------- tuning ---
//
// The receiver captures one contiguous window, `sample_rate` wide, centred on a
// single frequency. These helpers keep the UI's notion of "the range" in step
// with what the hardware can actually see, instead of claiming a whole band.
const BAND_PRESETS = { vhf: 165, uhf: 650, microwave: 3500 };

function currentCenterMhz() {
    const raw = parseFloat(document.getElementById('center-freq')?.value);
    return isFinite(raw) ? raw : 145;
}

function currentSampleRateHz() {
    return parseInt(document.getElementById('sample-rate')?.value || 2048000, 10);
}

function monitoredWindowMhz() {
    const center = currentCenterMhz();
    const half = currentSampleRateHz() / 2 / 1e6;
    return [center - half, center + half];
}

function setMonitorWindowLabel(startMhz, endMhz, extra) {
    const el = document.getElementById('monitor-window-value');
    if (!el) return;
    if (!isFinite(startMhz) || !isFinite(endMhz)) {
        el.textContent = 'idle — start a scan';
        return;
    }
    el.textContent = startMhz.toFixed(3) + ' – ' + endMhz.toFixed(3) + ' MHz (' +
        (endMhz - startMhz).toFixed(3) + ' MHz wide)' + (extra ? ' ' + extra : '');
}

function updateMonitorWindowReadout() {
    const live = window.currentMonitoredWindowMhz;
    if (currentMonitoring && live) {
        setMonitorWindowLabel(live[0], live[1], '· live');
        return;
    }
    const [a, b] = monitoredWindowMhz();
    setMonitorWindowLabel(a, b, currentMonitoring ? '· next scan' : '');
}

function applyBandPreset() {
    const sel = document.getElementById('frequency-range');
    const center = sel ? BAND_PRESETS[sel.value] : null;
    if (center !== null && center !== undefined) {
        const input = document.getElementById('center-freq');
        if (input) input.value = center;
    }
    if (currentMonitoring) {
        retuneMonitoring();
    } else {
        updateMonitorWindowReadout();
    }
}

async function retuneMonitoring() {
    const [start, end] = monitoredWindowMhz();
    setMonitorWindowLabel(start, end, '· retuning');
    try {
        const resp = await fetch('/api/monitoring/retune', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                center_freq: currentCenterMhz(),
                sample_rate: currentSampleRateHz(),
                frequency_range: start.toFixed(3) + '-' + end.toFixed(3)
            })
        });
        if (resp.ok) {
            const r = await resp.json();
            if (r.status === 'success') {
                showNotification('Re-tuned to ' + currentCenterMhz().toFixed(3) + ' MHz', 'info');
            }
        }
    } catch (error) {
        console.error('Retune failed:', error);
    }
}

function setupTuningControls() {
    const preset = document.getElementById('frequency-range');
    if (preset) preset.addEventListener('change', applyBandPreset);
    const center = document.getElementById('center-freq');
    if (center) {
        center.addEventListener('change', () => {
            const sel = document.getElementById('frequency-range');
            if (sel) sel.value = 'custom';
            if (currentMonitoring) {
                retuneMonitoring();
            } else {
                updateMonitorWindowReadout();
            }
        });
    }
    const rate = document.getElementById('sample-rate');
    if (rate) rate.addEventListener('change', updateMonitorWindowReadout);
    updateMonitorWindowReadout();
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
        
        // Collect configuration from UI elements.
        //
        // Tuning is a centre frequency plus a sample rate; the receiver sees
        // sample_rate / 2 either side. A start-end pair is still sent so older
        // code paths collapse back to the same centre. There is deliberately no
        // "all bands" option: the dongle cannot capture everything at once.
        const centerMhz = currentCenterMhz();
        const sampleRate = currentSampleRateHz();
        const [windowStart, windowEnd] = monitoredWindowMhz();

        const config = {
            center_freq: centerMhz,
            frequency_range: windowStart.toFixed(3) + '-' + windowEnd.toFixed(3),
            sample_rate: sampleRate,
            gain: 20,
            threshold: parseFloat(document.getElementById('threshold-slider')?.value || 0.8),
            scan_interval: parseInt(document.getElementById('scan-interval')?.value || 5),
            // || false would coerce a deliberate false into false here, but
            // startAdvancedMonitoring used `|| true` which made Strict Mode
            // impossible to turn off. Compare explicitly instead.
            strict_mode: document.getElementById('strict-mode')?.checked === true
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
            await response.json();
            currentMonitoring = true;
            updateMonitoringUI(true);

            // Update status indicators
            updateStatusIndicator('monitor-status', true);
            updateStatusIndicator('system-status', true);

            showNotification('✅ Spectrum monitoring started successfully!', 'success');

            // Start polling for monitoring status
            startMonitoringStatusPolling();

            // Scope the alert list to the window actually being monitored, so
            // persisted detections from earlier scans at other frequencies do
            // not read as alerts "on a frequency not being monitored".
            const [startMhz, endMhz] = monitoredWindowMhz();
            window.currentMonitoredWindowMhz = [startMhz, endMhz];
            if (window.SAAlertsSetWindow) {
                window.SAAlertsSetWindow(startMhz * 1e6, endMhz * 1e6);
            }
            updateMonitorWindowReadout();

        } else {
            // Surface the server's reason (missing dongle, busy device, ...),
            // not just the HTTP status text.
            let detail = response.statusText;
            try {
                const body = await response.json();
                if (body && body.message) detail = body.message;
            } catch (parseError) {
                // keep statusText
            }
            console.error('Failed to start monitoring:', detail);
            showNotification('❌ Failed to start monitoring: ' + detail, 'error');
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

            // Back to idle: drop the window scope and show today's history.
            window.currentMonitoredWindowMhz = null;
            if (window.SAAlertsSetWindow) {
                window.SAAlertsSetWindow(null, null);
            }
            updateMonitorWindowReadout();

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
                    // monitoring_active is what the endpoint returns; reading
                    // .active (which never exists) flipped a running session
                    // back to "stopped" in this page a tick after it started.
                    const isActive = result.data.monitoring_active === true;
                    const deviceInfo = result.data.device_info || {};

                    // Keep the window scope in step with the server's real
                    // capture window, even for a session this page did not start.
                    const winHz = result.data.monitored_window_hz;
                    if (isActive && Array.isArray(winHz) && winHz.length === 2) {
                        window.currentMonitoredWindowMhz = [winHz[0] / 1e6, winHz[1] / 1e6];
                        if (window.SAAlertsSetWindow) {
                            window.SAAlertsSetWindow(winHz[0], winHz[1]);
                        }
                        updateMonitorWindowReadout();
                    }

                    // Badge and header LED follow the server every tick, even
                    // when the local flag already agrees with it.
                    updateMonitoringStatus(result.data);

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
                        // Deliberately not 'sample-rate': that id belongs to a
                        // <select> in Data Management. Writing text into it
                        // destroyed its options and made collectData send NaN.
                        // The live rate belongs in its own read-only field.
                        updateElementText('live-sample-rate',
                            (deviceInfo.sample_rate / 1e6).toFixed(3) + ' MS/s');
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
        // The Model training panel owns the choice; writing the polled value
        // back into it would replace the <select> and lose its options.
        const modelType = document.getElementById('model-type')?.value || 'isolation_forest';
        
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

/**
 * Reflect training progress.
 *
 * The markup for this block used to be injected here with innerHTML on first
 * use, which meant it could not be styled from CSS, was absent from the DOM
 * until training started, and collided with the #training-progress-bar already
 * in the template. It is now declared in the markup and only its values change.
 */
function updateTrainingProgressDisplay(status) {
    const display = $('training-progress-display');
    if (!display) return;

    const stage = status.stage || status.message || status.status || 'Working';
    const progress = typeof status.progress === 'number'
        ? Math.max(0, Math.min(100, status.progress)) : null;

    display.hidden = false;
    updateResultValue('training-status-text', status.status || stage);
    updateResultValue('training-model-type', status.model_type || '-');

    if (progress !== null) {
        updateResultValue('training-progress-percentage', progress.toFixed(0) + '%');
        const bar = $('training-progress-bar-detailed');
        if (bar) bar.style.width = progress + '%';
    }

    const note = $('training-metrics');
    if (note && status.metrics) {
        note.textContent = Object.entries(status.metrics)
            .map(([key, value]) => key + ': ' + value)
            .join('  ');
    }
}

function hideTrainingProgress() {
    const progressDisplay = $('training-progress-display');
    if (progressDisplay) {
        progressDisplay.hidden = true;
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

function displayTestResults(payload) {
    // Target the ids that actually exist. This previously wrote to
    // #accuracy / #precision / #recall / #f1-score, none of which are in the
    // markup, so the results panel silently stayed empty -- and it was never
    // unhidden either, so even correct values would not have been visible.
    const container = $('training-results');
    const metrics = (payload && (payload.results ? payload.results.metrics : payload.metrics)) || payload || {};

    const percent = function (key) {
        const value = metrics[key];
        return (value === undefined || value === null) ? '--' : (value * 100).toFixed(1) + '%';
    };

    updateResultValue('result-accuracy', percent('accuracy'));
    updateResultValue('result-precision', percent('precision'));
    updateResultValue('result-recall', percent('recall'));
    updateResultValue('result-f1', percent('f1_score'));
    updateResultValue('result-samples', metrics.samples !== undefined
        ? Number(metrics.samples).toLocaleString() : '--');
    updateResultValue('result-time', metrics.training_time !== undefined
        ? Number(metrics.training_time).toFixed(1) + ' s' : '--');

    if (container) container.style.display = 'block';

    if (metrics.accuracy !== undefined) {
        notify(`Model test: ${percent('accuracy')} accuracy`,
            { severity: 'info', title: 'Model test' });
    }
}

function updateResultValue(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function $(id) { return document.getElementById(id); }

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

/**
 * Render the data-file list into the existing panel.
 *
 * This used to look for #data-files-container, not find it, and inject a whole
 * second section -- so the Data Management panel showed two competing file
 * lists in two different containers. It now writes to the one the markup
 * already declares and says how many are hidden instead of silently capping.
 */
function displayDataFiles(files) {
    const list = $('data-file-list');
    if (!list) return;

    if (!files.length) {
        list.innerHTML =
            '<div class="sa-state sa-state--empty">' +
            '<div class="sa-state-title">No data files</div>' +
            '<div class="sa-state-detail">Start monitoring or run data collection to create some.</div>' +
            '</div>';
        return;
    }

    var MAX_SHOWN = 20;
    var shown = files.slice(0, MAX_SHOWN);
    var hidden = files.length - shown.length;

    list.innerHTML = '<div class="files-grid">' + shown.map(function (file) {
        var icon = file.type === 'csv' ? 'fa-file-csv' : 'fa-file-code';
        var freq = (file.frequency_range && isFinite(file.frequency_range.min_hz))
            ? (file.frequency_range.min_hz / 1e6).toFixed(1) + '-' +
              (file.frequency_range.max_hz / 1e6).toFixed(1) + ' MHz'
            : 'unknown band';
        // Escape anything that came from a filename before it reaches innerHTML.
        var name = SAUI.escapeHtml(file.filename);
        var when = SAUI.format.relativeTime(file.modified || file.created);
        var size = SAUI.format.bytes((file.size_mb || 0) * 1024 * 1024);
        var samples = isFinite(file.samples) ? file.samples.toLocaleString() + ' samples' : '';
        return '<div class="file-item" data-file="' + name + '" tabindex="0" role="button" ' +
            'title="Analyse ' + name + '">' +
            '<div class="file-icon"><i class="fas ' + icon + '"></i></div>' +
            '<div class="file-info">' +
            '<div class="file-name">' + name + '</div>' +
            '<div class="file-details">' +
            '<span class="file-size">' + size + '</span>' +
            (samples ? '<span class="file-samples">' + samples + '</span>' : '') +
            '<span class="file-freq">' + freq + '</span>' +
            '</div>' +
            '<div class="file-date">' + when + '</div>' +
            '</div></div>';
    }).join('') + '</div>' +
        (hidden > 0
            ? '<div class="sa-hint">Showing ' + MAX_SHOWN + ' of ' + files.length +
              ' files. Filter or export the rest from the API.</div>'
            : '');

    // Keyboard activation: the items are divs, so Enter and Space are wired here.
    SAUI.delegate(list, '.file-item', 'keydown', function (event, target) {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        analyzeDataFile(target.getAttribute('data-file'));
    });
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
    const overlay = $('file-analysis-modal');
    const body = $('file-analysis-body');
    if (!overlay || !body) return;

    const esc = SAUI.escapeHtml;
    const freq = analysis.frequency_analysis || {};
    const sig = analysis.signal_statistics || {};
    const columns = Array.isArray(analysis.columns) ? analysis.columns : [];
    const rows = Array.isArray(analysis.preview_data) ? analysis.preview_data.slice(0, 5) : [];

    function fixed(value, digits) {
        const n = Number(value);
        return isFinite(n) ? n.toFixed(digits === undefined ? 2 : digits) : '--';
    }

    let html = '<p class="sa-panel-desc">Source file: <strong>' +
        esc(analysis.filename) + '</strong></p>';

    html += '<div class="analysis-stats">' +
        '<div class="stat-box"><div class="stat-label">File size</div>' +
        '<div class="stat-value">' + esc(String(analysis.size_mb)) + ' MB</div></div>' +
        '<div class="stat-box"><div class="stat-label">Total samples</div>' +
        '<div class="stat-value">' + esc(Number(analysis.samples).toLocaleString()) + '</div></div>' +
        '<div class="stat-box"><div class="stat-label">Columns</div>' +
        '<div class="stat-value">' + esc(String(columns.length)) + '</div></div>' +
        '</div>';

    if (Object.keys(freq).length) {
        html += '<div class="analysis-section"><h4>Frequency analysis</h4>' +
            '<dl class="sa-define">' +
            '<dt>Range</dt><dd>' + fixed(Number(freq.min_hz) / 1e6) + ' &ndash; ' +
            fixed(Number(freq.max_hz) / 1e6) + ' MHz</dd>' +
            '<dt>Center</dt><dd>' + fixed(Number(freq.mean_hz) / 1e6) + ' MHz</dd>' +
            '<dt>Unique frequencies</dt><dd>' + esc(String(freq.unique_frequencies)) + '</dd>' +
            '</dl></div>';
    }

    if (Object.keys(sig).length) {
        html += '<div class="analysis-section"><h4>Signal statistics</h4>' +
            '<dl class="sa-define">' +
            '<dt>Range</dt><dd>' + fixed(sig.min) + ' to ' + fixed(sig.max) + ' dBFS</dd>' +
            '<dt>Average</dt><dd>' + fixed(sig.mean) + ' dBFS</dd>' +
            '<dt>Std dev</dt><dd>' + fixed(sig.std) + ' dBFS</dd>' +
            '</dl></div>';
    }

    if (rows.length) {
        let table = '<div class="sa-table-scroll"><table class="sa-table"><thead><tr>' +
            columns.map(function (c) { return '<th>' + esc(c) + '</th>'; }).join('') +
            '</tr></thead><tbody>';
        rows.forEach(function (row) {
            table += '<tr>' + columns.map(function (c) {
                const value = row[c];
                return '<td>' + esc(typeof value === 'number' ? value.toFixed(3) : String(value)) + '</td>';
            }).join('') + '</tr>';
        });
        table += '</tbody></table></div>';
        html += '<div class="analysis-section"><h4>Data preview</h4>' + table + '</div>';
    }

    body.innerHTML = html;
    currentAnalysisFilename = analysis.filename;

    const trainBtn = $('file-analysis-train');
    if (trainBtn) trainBtn.onclick = function () { useFileForTraining(analysis.filename); };
    const closeBtn = $('file-analysis-close');
    if (closeBtn) closeBtn.onclick = closeFileAnalysis;
    const exportBtn = $('file-analysis-export');
    if (exportBtn) exportBtn.onclick = function () { exportFileAnalysis(currentAnalysisFilename); };

    overlay.style.display = 'flex';
    showNotification('Analysis complete for ' + analysis.filename, 'success');
}

let currentAnalysisFilename = null;

function closeFileAnalysis() {
    const overlay = $('file-analysis-modal');
    if (overlay) overlay.style.display = 'none';
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
        trainingSection.style.boxShadow = '0 0 0 1px var(--accent), 0 0 24px rgba(76, 141, 255, 0.3)';
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
    const overlay = $('dataset-selector-modal');
    const list = $('dataset-list');
    if (!overlay || !list) return;

    const esc = SAUI.escapeHtml;
    const caption = $('dataset-modal-type');
    if (caption) {
        caption.textContent = datasets.length + ' dataset(s) found in the data directory.';
    }

    let html = '<div class="datasets-grid">';
    datasets.forEach(function (dataset) {
        html += '<button type="button" class="dataset-item" data-name="' + esc(dataset.name) +
            '" data-type="' + esc(dataset.type) + '">' +
            '<div class="dataset-item__head"><h4>' + esc(dataset.name) + '</h4>' +
            '<span class="sa-badge">' + esc(dataset.type) + '</span></div>' +
            '<div class="dataset-item__meta">' + esc(String(dataset.file_count)) + ' files (' +
            esc(String(dataset.csv_files)) + ' CSV, ' + esc(String(dataset.json_files)) + ' JSON) &middot; ' +
            esc(Number(dataset.size_mb).toFixed(1)) + ' MB</div>' +
            '<p class="dataset-item__desc">' + esc(dataset.description || '') + '</p>' +
            '</button>';
    });
    html += '</div>';
    list.innerHTML = html;

    Array.prototype.forEach.call(list.querySelectorAll('.dataset-item'), function (item) {
        item.addEventListener('click', function () {
            selectDataset(item.dataset.name, item.dataset.type);
        });
    });

    const clearBtn = $('dataset-modal-clear');
    if (clearBtn) clearBtn.onclick = function () {
        clearSelectedDataset();
        closeDatasetSelector();
    };
    const confirmBtn = $('dataset-modal-confirm');
    if (confirmBtn) confirmBtn.onclick = closeDatasetSelector;

    overlay.style.display = 'flex';
}

function closeDatasetSelector() {
    const overlay = $('dataset-selector-modal');
    if (overlay) overlay.style.display = 'none';
}
function selectDataset(datasetName, datasetType) {
    closeDatasetSelector();
    window.selectedDataset = { name: datasetName, type: datasetType };
    showNotification(`Selected dataset: ${datasetName}`, 'success');
    
    // Update training interface to show selected dataset
    const trainingSection = document.querySelector('.training-panel');
    if (trainingSection) {
        trainingSection.scrollIntoView({ behavior: 'smooth' });
        trainingSection.style.boxShadow = '0 0 0 1px var(--accent), 0 0 24px rgba(76, 141, 255, 0.3)';
        setTimeout(() => {
            trainingSection.style.boxShadow = '';
        }, 3000);
    }
    
    // Update dataset display in training section if it exists
    updateTrainingDatasetDisplay(datasetName, datasetType);
}

function updateTrainingDatasetDisplay(datasetName, datasetType) {
    // The selection chip used to be injected into the training panel with
    // innerHTML on demand. It is declared in the markup now so it can be
    // styled, focused and read by assistive tech, and so a stale selection can
    // never survive a re-render.
    const display = $('selected-dataset-display');
    if (!display) return;

    $('dataset-name-display').textContent = datasetName;
    $('dataset-type-display').textContent = datasetType || '';
    display.hidden = false;
}

function clearSelectedDataset() {
    window.selectedDataset = null;
    const datasetDisplay = $('selected-dataset-display');
    if (datasetDisplay) {
        datasetDisplay.hidden = true;
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
        showNotification(`Multi-band settings recorded for ${(data.data.bands || []).length} band(s)`, 'success');
    } else if (data.type === 'advanced_monitoring_started') {
        showNotification('Advanced settings recorded. Capture is started from Spectrum analysis.', 'success');
    } else if (data.type === 'mqtt_connected') {
        showNotification(`Broker settings saved for ${data.data.broker}. No connection is opened.`,
            'success');
        loadMQTTStatus();
    } else if (data.type === 'mqtt_disconnected') {
        showNotification('Broker settings cleared', 'info');
        loadMQTTStatus();
    } else if (data.type === 'mqtt_test_sent') {
        showNotification('Test event recorded by the server', 'info');
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
    const named = severity.charAt(0).toUpperCase() + severity.slice(1);
    showNotification(`${named} threat detected at ${anomaly.frequency} MHz`, 'warning');
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
            <span class="alert-severity">${cap(anomaly.severity || 'medium')}</span>
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
    // Field names come from /api/spectrum/analysis and the spectrum_update
    // WebSocket payload. The previous reader looked for avg_power, amplitudes,
    // peak_frequency and snr, none of which either endpoint returns, so the
    // stats row stayed on "--" while the panel appeared to work.
    //
    // Units: these are dBFS, not dBm. An RTL-SDR has no absolute power
    // reference, so labelling them dBm states something the hardware cannot
    // know. The labels in the markup were fixed to match.
    const dbfs = window.SAUI ? SAUI.format.dbfs
        : (v) => (isFinite(v) ? Number(v).toFixed(1) + ' dBFS' : '--');
    const fmt = window.SAUI ? SAUI.format : null;

    // Peak: the analysis endpoint reports a list, the WebSocket a scalar.
    //
    // NaN, not null: isFinite(null) is true in JavaScript (null coerces to 0),
    // so a null initial value passed every guard and the card rendered
    // "0.000000 MHz" for payloads that simply had no peak yet.
    let peakHz = NaN;
    let peakLevel = data.signal_strength;
    if (Array.isArray(data.peak_frequencies) && data.peak_frequencies.length) {
        const top = data.peak_frequencies.reduce((best, entry) =>
            (!best || entry.power_dbfs > best.power_dbfs) ? entry : best, null);
        if (top) {
            peakHz = top.frequency;
            if (isFinite(peakLevel)) peakLevel = top.power_dbfs;
        }
    }
    if (!isFinite(peakHz) && isFinite(data.peak_frequency)) {
        peakHz = data.peak_frequency * 1e6;
    }

    if (isFinite(peakHz)) {
        // toFixed guards against values like 146.00000000000003 arriving
        // from float arithmetic.
        updateElementText('peak-frequency', (peakHz / 1e6).toFixed(6) + ' MHz');
    }
    if (isFinite(peakLevel)) {
        updateElementText('avg-power', dbfs(peakLevel));
    }
    if (isFinite(data.background_level)) {
        updateElementText('noise-floor', dbfs(data.background_level));
    }
    if (isFinite(data.peak_level) && isFinite(data.background_level)) {
        updateElementText('snr-value',
            (data.peak_level - data.background_level).toFixed(1) + ' dB');
    } else if (isFinite(data.snr)) {
        updateElementText('snr-value', data.snr.toFixed(1) + ' dB');
    }
    if (isFinite(data.sample_rate) && isFinite(data.total_samples) && data.total_samples > 1) {
        // Resolution bandwidth of the Welch estimate: sample rate over the
        // number of FFT segments the estimate actually used.
        const rbw = data.sample_rate / Math.max(1, Math.min(data.total_samples, 1024));
        updateElementText('rbw-value', fmt ? fmt.hz(rbw) : rbw.toFixed(1) + ' Hz');
    }
    document.dispatchEvent(new CustomEvent('sa:spectrum', { detail: data }));
}

async function loadSpectrumAnalysis() {
    // While a scan is running, the live WebSocket pushes the real trace. Polling
    // the archived analysis here used to overwrite it every 5 s with an old
    // capture from a different centre — which is why the chart and the alerts
    // disagreed and the band buttons "did nothing".
    if (currentMonitoring) return;
    try {
        // No range parameter: the endpoint reads the most recent capture and
        // reports the centre it was actually taken at. The requested range is
        // only a fallback for the error path.
        const response = await fetch('/api/spectrum/analysis');
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'ok' && result.analysis) {
                updateSpectrumData(result.analysis);
                updateSpectrumChart(result.analysis);
                const c = result.analysis.center_frequency;
                const sr = result.analysis.sample_rate;
                if (isFinite(c) && isFinite(sr)) {
                    setMonitorWindowLabel((c - sr / 2) / 1e6, (c + sr / 2) / 1e6, '· last capture');
                }
            } else {
                setSpectrumState('empty', 'No spectrum data',
                    'The backend has not captured a trace yet.');
            }
        } else {
            setSpectrumState('error', 'Spectrum unavailable',
                'The backend returned HTTP ' + response.status + '.',
                loadSpectrumAnalysis);
        }
    } catch (error) {
        console.error('Error loading spectrum analysis:', error);
        setSpectrumState('error', 'Spectrum unavailable', error.message, loadSpectrumAnalysis);
    }
}

function startSpectrumRefresh() {
    // Refresh spectrum analysis every 5 seconds
    setInterval(loadSpectrumAnalysis, 5000);
}

function updateSpectrumChart(data) {
    const canvas = document.getElementById('spectrum-chart');
    if (!canvas || !data) return;

    // A destroyed Chart instance keeps a reference to its canvas. If the canvas
    // has since been detached (page navigation, a re-render) Chart.js throws
    // "Cannot read properties of null (reading 'ownerDocument')". Drop the stale
    // reference instead of feeding it another update.
    if (window.spectrumChart && !canvas.isConnected) {
        window.spectrumChart = null;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Prepare data for chart
    let frequencies = [];
    let amplitudes = [];
    
    // The analysis endpoint returns `frequencies` + `power_spectrum`; the
    // WebSocket returns `frequencies_mhz` + `powers_dbfs`. Both are dBFS.
    const freqsIn = data.frequencies || data.frequencies_mhz || data.frequency_data;
    const powerIn = data.power_spectrum || data.powers_dbfs || data.amplitudes
        || data.amplitude_data;
    if (Array.isArray(freqsIn) && Array.isArray(powerIn) && freqsIn.length === powerIn.length) {
        frequencies = freqsIn;
        amplitudes = powerIn;
    }

    // No synthetic fallback. The previous version invented a plausible-looking
    // Gaussian spectrum with a peak at the centre whenever the API returned
    // nothing, so a silent backend failure rendered as a convincing picture of
    // a signal that did not exist. An operator acting on that would be chasing
    // a phantom.
    if (!frequencies.length) {
        setSpectrumState('empty', 'No spectrum data',
            'The backend returned no trace. Start monitoring to capture one.');
        return;
    }

    setSpectrumState('ready');

    const dataset = {
        label: 'Power Spectrum (dBFS)',
        data: amplitudes,
        borderColor: chartColors.accent,
        backgroundColor: chartColors.accentFill,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.15
    };
    const labels = frequencies.map(f => Number(f).toFixed(2));

    // Update the existing chart in place. Destroying and recreating it on
    // every 5 s poll threw away all animation state and made the chart flash.
    try {
        if (window.spectrumChart && window.spectrumChart.data &&
            window.spectrumChart.data.labels.length === labels.length) {
            window.spectrumChart.data.labels = labels;
            window.spectrumChart.data.datasets[0].data = amplitudes;
            window.spectrumChart.update('none');
            return;
        }

        if (window.spectrumChart) {
            window.spectrumChart.destroy();
            window.spectrumChart = null;
        }
    } catch (err) {
        console.warn('Could not update the spectrum chart in place:', err);
        window.spectrumChart = null;
    }

    try {
    window.spectrumChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: frequencies.map(f => f.toFixed(2)),
            datasets: [dataset]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            animation: { duration: 220 },
            plugins: {
                legend: {
                    labels: chartLegendLabels(chartColors.text)
                },
                tooltip: chartTooltip()
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Frequency (MHz)',
                        color: chartColors.text,
                        font: { size: 11, weight: '500' }
                    },
                    ticks: {
                        color: chartColors.text,
                        maxTicksLimit: 9,
                        maxRotation: 0,
                        autoSkip: true,
                        padding: 6
                    },
                    grid: {
                        color: chartColors.grid
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Level (dBFS)',
                        color: chartColors.text,
                        font: { size: 11, weight: '500' }
                    },
                    ticks: {
                        color: chartColors.text,
                        padding: 6
                    },
                    grid: {
                        color: chartColors.grid
                    }
                }
            }
        }
    });
    } catch (err) {
        window.spectrumChart = null;
        setSpectrumState('error', 'Chart error', err.message);
        console.error('Spectrum chart error:', err);
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
    
    // Model type is a read-only stat here. Overwriting the training panel's
    // <select> by the same id destroyed its options on the first poll.
    if (info.model_type) {
        updateElementText('model-type-display', info.model_type);
    }
    if (info.is_deployed !== undefined) {
        const deployStatus = info.is_deployed ? 'Deployed' : 'Not deployed';
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
                backgroundColor: chartColors.accentFill,
                borderColor: chartColors.accent,
                borderWidth: 2,
                pointBackgroundColor: chartColors.accent,
                pointBorderColor: chartColors.surface,
                pointBorderWidth: 2,
                pointHoverBackgroundColor: chartColors.surface,
                pointHoverBorderColor: chartColors.accent
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 260 },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: chartTooltip()
            },
            scales: {
                r: {
                    angleLines: {
                        color: chartColors.grid
                    },
                    grid: {
                        color: chartColors.grid
                    },
                    pointLabels: {
                        color: chartColors.text,
                        font: { size: 11, weight: '500' }
                    },
                    ticks: {
                        color: chartColors.text,
                        backdropColor: 'transparent',
                        stepSize: 20,
                        font: { size: 10 }
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
//
// Every case below corresponds to a type the server actually emits (see the
// `_broadcast_to_websockets` calls in web/app.py). The previous handler
// listened for monitoring_update / training_complete / training_failed, none of
// which are ever sent, while the types that are sent - spectrum_update,
// training_update, model_test_complete, mqtt_*, multiband_monitoring_started -
// all fell through to the default branch. No real-time update reached the UI.
function handleWebSocketMessage(message) {
    if (!message || !message.type) return;
    const payload = message.data || {};

    switch (message.type) {
        case 'anomaly_detected': {
            recordAnomaly(payload);
            loadRecentAnomalies();
            const severity = payload.severity || 'medium';
            notify(
                `${SAUI.format.mhz(payload.frequency_mhz, 6)} - ${severity} anomaly`,
                {
                    severity: severity === 'critical' ? 'critical' : 'warning',
                    title: 'Anomaly detected',
                    actions: [{
                        label: 'Locate',
                        onClick: () => {
                            // Carry the frequency across so the operator does not
                            // have to retype it.
                            const target = SAUI.store.get('geoTargetHz', null);
                            if (target) {
                                window.open(`/geolocate?freq=${encodeURIComponent(target)}`, '_blank');
                            } else {
                                window.open('/geolocate', '_blank');
                            }
                        }
                    }]
                }
            );
            break;
        }

        case 'system_status':
            updateSystemStatus(payload);
            break;

        case 'spectrum_update':
            // Field names follow the server exactly: frequencies_mhz / powers_dbfs
            // / peak_frequency_mhz / noise_floor_dbfs / snr_db.
            if (currentMonitoring && isFinite(payload.window_start_mhz) &&
                isFinite(payload.window_end_mhz)) {
                window.currentMonitoredWindowMhz = [payload.window_start_mhz, payload.window_end_mhz];
                setMonitorWindowLabel(payload.window_start_mhz, payload.window_end_mhz, '· live');
            }
            updateSpectrumData({
                frequencies: payload.frequencies_mhz,
                amplitudes: payload.powers_dbfs,
                peak_frequency: payload.peak_frequency_mhz,
                avg_power: payload.avg_power_dbfs,
                snr: payload.snr_db,
                background_level: payload.noise_floor_dbfs
            });
            if (Array.isArray(payload.frequencies_mhz)) {
                updateSpectrumChart({
                    frequencies: payload.frequencies_mhz,
                    amplitudes: payload.powers_dbfs
                });
            }
            break;

        case 'training_update':
            if (payload.status) {
                updateTrainingStatus({
                    status: payload.status,
                    progress: payload.progress,
                    stage: payload.stage,
                    message: payload.message
                });
            }
            if (payload.status === 'completed') {
                notify('Model training completed', { severity: 'success', title: 'Training' });
                loadModelStatus();
                loadTrainingStatus();
            } else if (payload.status === 'failed') {
                notify(payload.message || 'Model training failed',
                    { severity: 'error', title: 'Training' });
                loadTrainingStatus();
            }
            break;

        case 'model_test_complete':
            displayTestResults(payload);
            break;

        case 'multiband_monitoring_started':
            notify(`Multi-band settings recorded (${(payload.bands || []).join(', ')})`,
                { severity: 'success', title: 'Configuration' });
            break;

        case 'advanced_monitoring_started':
            notify('Advanced settings recorded', { severity: 'success', title: 'Configuration' });
            break;

        case 'mqtt_connected':
            mqttConnected = true;
            loadMQTTStatus();
            break;

        case 'mqtt_disconnected':
            mqttConnected = false;
            loadMQTTStatus();
            break;

        case 'mqtt_test_sent':
            notify('Test event recorded by the server; nothing is published to a broker',
                { severity: 'info', title: 'MQTT' });
            break;

        default:
            console.debug('Unhandled WebSocket message type:', message.type);
    }
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
        updateElementText('avg-signal', data.avg_signal.toFixed(1) + ' dBFS');
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
    // Theme tokens rather than fixed neon: #00ff00 on a light panel is 1.3:1.
    const color = change >= 0 ? 'var(--sa-ok)' : 'var(--sa-critical)';
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
                borderColor: chartColors.accent,
                backgroundColor: chartColors.accentFill,
                borderWidth: 2,
                pointRadius: 2.5,
                pointHoverRadius: 4,
                pointBackgroundColor: chartColors.accent,
                pointBorderColor: chartColors.surface,
                pointBorderWidth: 1,
                tension: 0.3,
                fill: true
            }, {
                label: 'Anomalies',
                data: timelineData.map(d => d.anomalies),
                borderColor: chartColors.danger,
                backgroundColor: chartColors.dangerFill,
                borderWidth: 2,
                pointRadius: 2.5,
                pointHoverRadius: 4,
                pointBackgroundColor: chartColors.danger,
                pointBorderColor: chartColors.surface,
                pointBorderWidth: 1,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            animation: { duration: 220 },
            plugins: {
                legend: {
                    labels: chartLegendLabels(chartColors.text)
                },
                tooltip: chartTooltip()
            },
            scales: {
                x: {
                    ticks: {
                        color: chartColors.text,
                        maxTicksLimit: 7,
                        maxRotation: 0,
                        autoSkip: true,
                        padding: 6
                    },
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: chartColors.text, precision: 0, padding: 6 },
                    grid: { color: chartColors.grid }
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
                backgroundColor: chartColors.accentBar,
                borderColor: chartColors.accent,
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
                maxBarThickness: 46
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 240 },
            plugins: {
                legend: { display: false },
                tooltip: chartTooltip()
            },
            scales: {
                x: {
                    ticks: {
                        color: chartColors.text,
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8,
                        padding: 6
                    },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: chartColors.text, padding: 6 },
                    grid: { color: chartColors.grid }
                }
            }
        }
    });
}
