/* ============================================================================
   SpectrumAlert dashboard enhancements
   ----------------------------------------------------------------------------
   Adds the pieces the dashboard never had, using the shared SAUI primitives:
   state rendering, alert management, export, keyboard shortcuts, the anomaly
   timeline, a theme toggle and honest connection reporting.

   Loaded after dashboard.js and augments it rather than replacing it.
   ========================================================================== */

(function (window, document) {
    'use strict';

    var sessionStart = Date.now();

    /** Alert buffer, newest first. Doubles as the export source. */
    var alerts = [];

    /**
     * Optional frequency scope (Hz) for the current scan window. When set, only
     * detections inside it are counted or listed, so a persisted alert from an
     * earlier scan at another frequency is not read as "on a frequency not being
     * monitored". Cleared when monitoring stops.
     */
    var windowFilter = null;

    /** Detections bucketed by minute for the anomaly timeline. */
    var timeline = [];

    var anomalyChart = null;

    // ------------------------------------------------------------- helpers --

    function $(id) { return document.getElementById(id); }

    function notify(message, options) {
        if (window.SAUI) {
            SAUI.notify.show(message, options);
        } else {
            console.log('[notify]', message, options);
        }
    }

    var SEVERITY_ORDER = ['low', 'medium', 'high', 'critical'];

    function severityRank(severity) {
        var index = SEVERITY_ORDER.indexOf(severity);
        return index < 0 ? 0 : index;
    }

    function alertInWindow(alert) {
        if (!windowFilter) return true;
        var hz = isFinite(alert.frequencyHz) ? alert.frequencyHz
            : (isFinite(alert.frequencyMhz) ? alert.frequencyMhz * 1e6 : null);
        if (hz === null) return true;
        return hz >= windowFilter.minHz && hz <= windowFilter.maxHz;
    }

    /**
     * Set (or clear) the scan-window scope. Exposed so dashboard.js can apply it
     * from the monitoring start/stop paths and the status poll, including for a
     * session this page did not start.
     */
    function setWindowFilter(minHz, maxHz) {
        if (isFinite(minHz) && isFinite(maxHz) && maxHz > minHz) {
            windowFilter = { minHz: minHz, maxHz: maxHz };
        } else {
            windowFilter = null;
        }
        updateCounters();
        updateThreatIndicator();
        renderAlerts();
    }
    window.SAAlertsSetWindow = setWindowFilter;

    /** A glyph per severity, so severity survives greyscale and colour blindness. */
    var SEVERITY_GLYPH = { critical: '!!', high: '!', medium: '~', low: '.' };

    /** Sentence-case a word for display; severities are lowercase in the data. */
    function cap(word) {
        return String(word || '').charAt(0).toUpperCase() + String(word || '').slice(1);
    }

    // -------------------------------------------------------------- alerts --

    /**
     * Record an anomaly, render the list, and bump the counters.
     * Exposed on window because dashboard.js's WebSocket handler calls it.
     */
    function recordAnomaly(payload) {
        payload = payload || {};
        var severity = payload.severity || 'medium';
        var record = {
            id: payload.id || ('a' + Date.now() + Math.random().toString(36).slice(2, 7)),
            at: payload.at ? new Date(payload.at) : new Date(),
            severity: severity,
            frequencyMhz: payload.frequency_mhz,
            frequencyHz: payload.frequency_hz,
            score: payload.anomaly_score,
            powerDbfs: payload.power_dbfs,
            noiseFloorDbfs: payload.noise_floor_dbfs,
            snrDb: payload.snr_db,
            type: payload.anomaly_type,
            description: payload.description || '',
            acked: false
        };

        // Persisted anomalies arrive with their own ids; re-rendering one already
        // in the buffer would duplicate it on every 5 s poll.
        var existingIndex = -1;
        for (var i = 0; i < alerts.length; i++) {
            if (alerts[i].id === record.id) { existingIndex = i; break; }
        }
        if (existingIndex >= 0) {
            alerts.splice(existingIndex, 1);
        }

        alerts.unshift(record);
        if (alerts.length > 500) alerts.pop();

        renderAlerts();
        // Only a genuinely new detection moves the timeline; a re-render of the
        // same persisted record must not inflate the counts.
        if (existingIndex < 0) {
            recordTimeline(severity, record.at);
            renderTimeline();
        }
        updateCounters();
        updateThreatIndicator();
    }
    window.recordAnomaly = recordAnomaly;

    function updateCounters() {
        var counts = { critical: 0, high: 0, medium: 0, low: 0 };
        alerts.forEach(function (alert) {
            if (!alertInWindow(alert)) return;
            if (counts[alert.severity] !== undefined) counts[alert.severity] += 1;
        });
        Object.keys(counts).forEach(function (key) {
            var node = $(key + '-count');
            if (node) node.textContent = counts[key];
        });
        // Mirror the open-alert total onto the Threats module tab so triage
        // does not require visiting the module to learn there is anything in it.
        var open = 0;
        alerts.forEach(function (alert) {
            if (!alert.acked && alertInWindow(alert)) open += 1;
        });
        if (window.SAModuleTabs) window.SAModuleTabs.setCount('threats', open, 'critical');
    }

    function updateThreatIndicator() {
        var worst = null;
        alerts.forEach(function (alert) {
            if (alert.acked || !alertInWindow(alert)) return;
            if (!worst || severityRank(alert.severity) > severityRank(worst)) {
                worst = alert.severity;
            }
        });
        var indicator = $('threat-status');
        if (!indicator) return;
        var led = indicator.querySelector('.led');
        // The indicator is colour-coded; the title and sr-only text carry the
        // same information for anyone who cannot see the colour.
        indicator.title = worst ? 'Highest unacknowledged severity: ' + worst : 'No open anomalies';
        if (!indicator.querySelector('.sa-sr-only')) {
            var sr = document.createElement('span');
            sr.className = 'sa-sr-only';
            indicator.appendChild(sr);
        }
        var sr = indicator.querySelector('.sa-sr-only');
        sr.textContent = indicator.title;
        if (led) {
            led.style.background = worst
                ? getComputedStyle(document.documentElement)
                    .getPropertyValue('--sa-' + worst)
                : '';
        }
    }

    function passesFilter(alert, threshold) {
        if (!alertInWindow(alert)) return false;
        if (threshold === 'all') return true;
        return severityRank(alert.severity) >= severityRank(threshold);
    }

    function renderAlerts() {
        var list = $('alerts-list');
        var empty = $('alerts-state');
        if (!list) return;

        var threshold = $('alert-severity') ? $('alert-severity').value : 'all';
        var inWindow = alerts.filter(alertInWindow);
        var visible = inWindow.filter(function (alert) { return passesFilter(alert, threshold); });

        if (empty) {
            empty.hidden = visible.length > 0;
            if (alerts.length > 0 && inWindow.length === 0) {
                empty.querySelector('.sa-state-title').textContent = 'No alerts in this window';
                empty.querySelector('.sa-state-detail').textContent =
                    'Detections outside the monitored window are hidden.';
            } else if (inWindow.length > 0 && visible.length === 0) {
                empty.querySelector('.sa-state-title').textContent = 'No alerts at this severity';
                empty.querySelector('.sa-state-detail').textContent =
                    'Lower the filter to see the ' + inWindow.length + ' lower-severity alert(s).';
            } else if (alerts.length === 0) {
                empty.querySelector('.sa-state-title').textContent = 'No alerts';
                empty.querySelector('.sa-state-detail').textContent =
                    'Nothing has been detected in this session.';
            }
        }

        if (!visible.length) { list.innerHTML = ''; return; }

        list.innerHTML = visible.slice(0, 50).map(function (alert) {
            var freq = isFinite(alert.frequencyMhz)
                ? SAUI.format.mhz(alert.frequencyMhz, 6)
                : '--';
            var level = isFinite(alert.powerDbfs) ? SAUI.format.dbfs(alert.powerDbfs) : '';
            var glyph = SEVERITY_GLYPH[alert.severity] || '';
            return '<div class="sa-alert sa-alert--' + alert.severity +
                (alert.acked ? ' sa-alert--acked' : '') + '" data-id="' + alert.id + '">' +
                '<div class="sa-alert__body">' +
                '<div class="sa-alert__freq">' +
                '<span class="severity-glyph" aria-hidden="true">' + glyph + '</span>' +
                '<span class="sa-sr-only">' + alert.severity + ' severity. </span>' +
                SAUI.escapeHtml(freq) + '</div>' +
                '<div class="sa-alert__meta">' +
                SAUI.escapeHtml(cap(alert.severity)) +
                (level ? ' &middot; ' + SAUI.escapeHtml(level) : '') +
                (isFinite(alert.snrDb) ? ' &middot; SNR ' + alert.snrDb.toFixed(1) + ' dB' : '') +
                ' &middot; ' + SAUI.escapeHtml(SAUI.format.relativeTime(alert.at)) +
                '</div></div>' +
                '<div class="sa-alert__actions">' +
                '<button class="sa-alert__action" data-act="ack" title="Acknowledge" ' +
                'aria-label="Acknowledge this alert">&#10003;</button>' +
                '<button class="sa-alert__action" data-act="locate" title="Open in geolocation" ' +
                'aria-label="Open this frequency in geolocation">&#9678;</button>' +
                '</div></div>';
        }).join('');
    }

    function findAlert(id) {
        for (var i = 0; i < alerts.length; i++) {
            if (alerts[i].id === id) return alerts[i];
        }
        return null;
    }

    function ackAlert(id) {
        var alert = findAlert(id);
        if (!alert) return;
        alert.acked = !alert.acked;
        renderAlerts();
        updateThreatIndicator();
    }

    function ackAll() {
        alerts.forEach(function (alert) { alert.acked = true; });
        renderAlerts();
        updateThreatIndicator();
        notify('Acknowledged ' + alerts.length + ' alert(s)',
            { severity: 'info', title: 'Alerts' });
    }

    /**
     * Empty the alert buffer. The list on screen is a view over this array, so
     * clearing the DOM alone would bring every alert back on the next render.
     * Exposed for dashboard.js's Clear button, which owns the confirmation.
     */
    function clearAllAlerts() {
        var removed = alerts.length;
        alerts.length = 0;
        renderAlerts();
        updateCounters();
        updateThreatIndicator();
        return removed;
    }
    window.SAAlertsClear = clearAllAlerts;

    /** Hand a frequency to the geolocation page so it opens on that target. */
    function locateAlert(id) {
        var alert = findAlert(id);
        if (!alert) return;
        var hz = isFinite(alert.frequencyHz)
            ? alert.frequencyHz
            : (isFinite(alert.frequencyMhz) ? alert.frequencyMhz * 1e6 : null);

        // The workspace is a section of this page now, so locating an alert
        // scrolls to it and hands the frequency over in-page instead of opening
        // a second window at the old /geolocate URL.
        if (hz) SAUI.store.set('geoTargetHz', hz);
        var workspace = document.getElementById('workspace');
        if (workspace) workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (hz) workspaceFocusTarget(hz);
    }

    // ----------------------------------------------------------- timeline ---

    /**
     * Bucket a detection into the timeline.
     *
     * The bucket size follows the span of the data: a live session is minutes
     * wide and wants per-minute bars, while restored history reaches back hours
     * and needs coarser buckets or every bar collapses into one. The previous
     * fixed 60 s bucket put an entire day of history into the current minute,
     * which is what drew the timeline as one solid block.
     */
    var TIMELINE_MAX_BUCKETS = 24;

    function timelineBucketSize() {
        if (!timeline.length) return 60000;
        var oldest = timeline[0].bucket;
        var newest = timeline[timeline.length - 1].bucket;
        var span = Math.max(0, newest - oldest);
        // Aim for ~TIMELINE_MAX_BUCKETS readable bars across the whole span.
        var size = Math.ceil(span / TIMELINE_MAX_BUCKETS);
        // Snap to a readable step so the axis labels mean something.
        var steps = [60000, 300000, 900000, 1800000, 3600000, 10800000, 21600000, 43200000];
        for (var i = 0; i < steps.length; i++) {
            if (steps[i] >= size) return steps[i];
        }
        return steps[steps.length - 1];
    }

    function recordTimeline(severity, at) {
        var size = timelineBucketSize();
        var when = at instanceof Date && !isNaN(at.getTime()) ? at.getTime() : Date.now();
        var bucket = Math.floor(when / size) * size;

        // Find the bucket rather than only appending: restored history arrives
        // newest-first, so appending would order the axis backwards.
        var entry = null;
        for (var i = 0; i < timeline.length; i++) {
            if (timeline[i].bucket === bucket) { entry = timeline[i]; break; }
        }
        if (!entry) {
            entry = { bucket: bucket, critical: 0, high: 0, medium: 0, low: 0 };
            timeline.push(entry);
            timeline.sort(function (a, b) { return a.bucket - b.bucket; });
        }
        entry[severity] = (entry[severity] || 0) + 1;
        while (timeline.length > TIMELINE_MAX_BUCKETS * 2) timeline.shift();
    }

    /** Detections per minute, stacked by severity. */
    function renderTimeline() {
        var canvas = $('anomaly-chart');
        var state = $('anomaly-chart-state');
        if (!canvas || !window.Chart) return;

        if (!timeline.length) {
            if (state) state.hidden = false;
            if (anomalyChart) {
                anomalyChart.destroy();
                anomalyChart = null;
            }
            return;
        }
        if (state) state.hidden = true;

        var labels = timeline.map(function (entry) {
            return new Date(entry.bucket).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        });

        function series(key, colour) {
            return {
                label: key.toUpperCase(),
                data: timeline.map(function (entry) { return entry[key] || 0; }),
                backgroundColor: colour,
                borderColor: colour,
                borderWidth: 1
            };
        }

        var root = getComputedStyle(document.documentElement);
        // Critical first: the legend and the stack should both read from the
        // most severe down, which is the order an operator triages in.
        var datasets = [
            series('critical', root.getPropertyValue('--sa-critical').trim() || '#ff0055'),
            series('high', root.getPropertyValue('--sa-high').trim() || '#ff4500'),
            series('medium', root.getPropertyValue('--sa-medium').trim() || '#ffaa00'),
            series('low', root.getPropertyValue('--sa-low').trim() || '#00ccff')
        ];

        if (anomalyChart) {
            // Update in place; recreating resets the animation on every tick.
            anomalyChart.data.labels = labels;
            anomalyChart.data.datasets = datasets;
            anomalyChart.update('none');
            return;
        }

        anomalyChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: { labels: labels, datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false },
                        border: { color: root.getPropertyValue('--border-color').trim() },
                        ticks: {
                            color: root.getPropertyValue('--text-muted').trim(),
                            maxRotation: 0,
                            autoSkipPadding: 12,
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        grid: { color: root.getPropertyValue('--border-color').trim() },
                        border: { display: false },
                        ticks: {
                            color: root.getPropertyValue('--text-muted').trim(),
                            precision: 0,
                            maxTicksLimit: 6
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end',
                        labels: {
                            color: root.getPropertyValue('--text-secondary').trim(),
                            boxWidth: 10,
                            boxHeight: 10,
                            usePointStyle: true,
                            pointStyle: 'rectRounded'
                        }
                    },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        });
    }

    // -------------------------------------------------------------- export ---

    function exportAlerts() {
        if (!alerts.length) {
            notify('No alerts to export', { severity: 'warning' });
            return;
        }
        var rows = alerts.map(function (alert) {
            return {
                timestamp: alert.at.toISOString(),
                severity: alert.severity,
                frequency_mhz: isFinite(alert.frequencyMhz)
                    ? Number(alert.frequencyMhz.toFixed(6)) : '',
                frequency_hz: alert.frequencyHz || '',
                level_dbfs: isFinite(alert.powerDbfs) ? alert.powerDbfs : '',
                noise_floor_dbfs: isFinite(alert.noiseFloorDbfs) ? alert.noiseFloorDbfs : '',
                snr_db: isFinite(alert.snrDb) ? alert.snrDb : '',
                anomaly_score: isFinite(alert.score) ? alert.score : '',
                anomaly_type: alert.type || '',
                acknowledged: alert.acked,
                description: alert.description
            };
        });
        SAUI.exporter.csv('spectrumalert_alerts_' + SAUI.format.fileStamp() + '.csv', rows);
    }

    // ----------------------------------------------------------- shortcuts ---

    function toggleHelp(open) {
        var overlay = $('sa-help');
        if (!overlay) return;
        if (open) {
            SAUI.shortcuts.renderHelp($('sa-help-body'));
            overlay.classList.add('is-open');
            // Focus the panel, not the Close button at the bottom: focusing
            // the button scrolled a tall shortcut list past its own heading.
            var panel = overlay.querySelector('.sa-help');
            if (panel) {
                panel.scrollTop = 0;
                if (panel.focus) panel.focus();
            }
        } else {
            overlay.classList.remove('is-open');
            var opener = $('help-open');
            if (opener) opener.focus();
        }
    }

    function registerShortcuts() {
        var S = SAUI.shortcuts;

        S.register('?', 'Show this shortcut list', function () { toggleHelp(true); }, 'Help');
        S.register('mod+/', 'Show this shortcut list', function () { toggleHelp(true); }, 'Help');
        S.register('escape', 'Close dialogs', function () {
            toggleHelp(false);
            var modal = document.querySelector('.modal-overlay, #file-analysis-modal, #dataset-selector-modal');
            if (modal && modal.style.display !== 'none') {
                modal.style.display = 'none';
            }
        }, 'Help');

        S.register('s', 'Start / stop monitoring', function () {
            if (monitoringActive()) { stopMonitoring(); } else { startMonitoring(); }
        }, 'Monitoring');
        S.register('r', 'Refresh all data now', refreshAll, 'Monitoring');
        // 'g' is taken by the workspace (triangulate), so scrolling to it uses
        // 'j'. The shared registry is last-write-wins, so both scripts must
        // agree on who owns which key.
        S.register('j', 'Jump to the analysis workspace', function () {
            var target = document.getElementById('workspace');
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 'Monitoring');

        S.register('a', 'Acknowledge every alert', ackAll, 'Alerts');
        S.register('e', 'Export alerts as CSV', exportAlerts, 'Alerts');
        S.register('x', 'Clear the alert list', function () { clearAlerts(); }, 'Alerts');
        S.register('d', 'Cycle the minimum alert severity', cycleSeverityFilter, 'Alerts');

        S.register('t', 'Toggle light / dark theme', function () {
            SAUI.theme.toggle();
            refreshChartThemes();
        }, 'Display');
        S.register('m', 'Mute / unmute notifications', toggleMute, 'Display');

        S.attach(document);
    }

    function cycleSeverityFilter() {
        var select = $('alert-severity');
        if (!select) return;
        var order = ['all', 'medium', 'high', 'critical'];
        var next = order[(order.indexOf(select.value) + 1) % order.length];
        select.value = next;
        renderAlerts();
        notify('Alert filter: ' + next, { severity: 'info', title: 'Alerts' });
    }

    function toggleMute() {
        var prefs = SAUI.notify.getPrefs();
        var muted = !prefs.muted;
        SAUI.notify.setPref('muted', muted);
        var button = $('notify-mute');
        if (button) {
            button.setAttribute('aria-pressed', String(muted));
            button.style.opacity = muted ? '0.5' : '';
        }
        notify(muted ? 'Notifications muted (criticals still shown)' : 'Notifications enabled',
            { severity: 'info' });
    }

    /** Apply the current palette to one existing Chart instance. */
    function applyChartTheme(chart) {
        if (!chart) return;
        var scales = chart.options.scales || {};
        Object.keys(scales).forEach(function (key) {
            var scale = scales[key];
            if (!scale) return;
            if (scale.ticks) scale.ticks.color = chartColors.text;
            if (scale.grid && scale.grid.display !== false) scale.grid.color = chartColors.grid;
            if (scale.pointLabels) scale.pointLabels.color = chartColors.text;
            if (scale.title) scale.title.color = chartColors.text;
        });
        var plugins = chart.options.plugins || {};
        if (plugins.legend && plugins.legend.labels) {
            plugins.legend.labels.color = chartColors.text;
        }
        if (plugins.tooltip) {
            plugins.tooltip.backgroundColor = chartColors.surface;
            plugins.tooltip.borderColor = chartColors.border;
            plugins.tooltip.titleColor = chartColors.textBright;
            plugins.tooltip.bodyColor = chartColors.text;
        }
        chart.update('none');
    }

    /** Charts bake colours in at construction, so they need rebuilding on theme change. */
    function refreshChartThemes() {
        if (typeof refreshChartColors === 'function') refreshChartColors();

        var spectrum = window.spectrumChart;
        if (spectrum && spectrum.data.datasets[0]) {
            spectrum.data.datasets[0].borderColor = chartColors.accent;
            spectrum.data.datasets[0].backgroundColor = chartColors.accentFill;
        }
        applyChartTheme(spectrum);

        var timeline = window.detectionsChart;
        if (timeline) {
            [chartColors.accent, chartColors.danger].forEach(function (colour, index) {
                var dataset = timeline.data.datasets[index];
                if (!dataset) return;
                dataset.borderColor = colour;
                dataset.backgroundColor = index === 0 ? chartColors.accentFill : chartColors.dangerFill;
                dataset.pointBackgroundColor = colour;
                dataset.pointBorderColor = chartColors.surface;
            });
            applyChartTheme(timeline);
        }

        var frequency = window.frequencyChart;
        if (frequency && frequency.data.datasets[0]) {
            frequency.data.datasets[0].backgroundColor = chartColors.accentBar;
            frequency.data.datasets[0].borderColor = chartColors.accent;
        }
        applyChartTheme(frequency);

        var model = window.modelChart;
        if (model && model.data.datasets[0]) {
            model.data.datasets[0].borderColor = chartColors.accent;
            model.data.datasets[0].backgroundColor = chartColors.accentFill;
            model.data.datasets[0].pointBackgroundColor = chartColors.accent;
            model.data.datasets[0].pointBorderColor = chartColors.surface;
        }
        applyChartTheme(model);

        // The anomaly timeline resolves its severity colours at construction.
        if (anomalyChart) {
            anomalyChart.destroy();
            anomalyChart = null;
        }
        renderTimeline();
    }

    // -------------------------------------------------------------- uptime ---

    function startUptimeClock() {
        var node = $('session-uptime');
        if (!node) return;
        window.setInterval(function () {
            node.textContent = SAUI.format.duration((Date.now() - sessionStart) / 1000);
        }, 1000);
    }

    function refreshAll() {
        ['loadSystemStatus', 'loadRecentAnomalies', 'loadSpectrumAnalysis',
            'loadModelStatus', 'loadAnalytics', 'loadDataStats', 'loadMQTTStatus',
            'loadTrainingStatus'].forEach(function (name) {
            if (typeof window[name] === 'function') {
                try { window[name](); } catch (err) { console.error(name, err); }
            }
        });
        notify('Refreshing all panels', { severity: 'info', title: 'Dashboard' });
    }

    // ---------------------------------------------------------------- guide ---
    //
    // The strip at the top of the dashboard states the order of work and,
    // crucially, whether each step is satisfied yet: the analyses refuse to run
    // until receivers exist, and nothing on the page used to say so before the
    // first press. Every status shown here is read from live state.

    /**
     * The monitoring flag is a top-level `let` in dashboard.js: a global
     * lexical binding, which window never sees. Reading window.currentMonitoring
     * always gave undefined, so the "s" shortcut could never stop a run.
     */
    function monitoringActive() {
        if (typeof window.currentMonitoring === 'boolean') return window.currentMonitoring;
        if (typeof currentMonitoring !== 'undefined') return !!currentMonitoring;
        return false;
    }

    /** Enabled receiver count, preferring the roster's own readiness hook. */
    function enabledReceiverCount() {
        if (typeof window.workspaceReadiness === 'function') {
            try {
                var ready = window.workspaceReadiness();
                if (ready && typeof ready.enabled === 'number') return ready.enabled;
            } catch (err) { /* roster not built yet; fall through to the text */ }
        }
        var node = $('sa-receiver-count');
        if (!node) return null;
        var text = node.textContent || '';
        var enabled = (/,\s*(\d+)\s+enabled/.exec(text) || [])[1];
        if (enabled !== undefined) return Number(enabled);
        var total = (/(\d+)\s+total/.exec(text) || [])[1];
        return total !== undefined ? Number(total) : null;
    }

    function setStep(stepId, statusId, state, status) {
        var step = $(stepId);
        var statusNode = $(statusId);
        if (step) step.setAttribute('data-state', state);
        if (statusNode) statusNode.textContent = status;
    }

    function refreshGuide() {
        var guide = $('sa-guide');
        if (!guide || guide.hidden) return;

        var enabled = enabledReceiverCount();
        var monitoring = monitoringActive();

        if (enabled === null) {
            setStep('sa-step-receivers', 'sa-step-receivers-status', 'pending',
                'Waiting for the receiver roster to load…');
        } else if (enabled >= 2) {
            setStep('sa-step-receivers', 'sa-step-receivers-status', 'ready',
                enabled + ' enabled — enough to triangulate');
        } else if (enabled === 1) {
            setStep('sa-step-receivers', 'sa-step-receivers-status', 'ready',
                '1 enabled — ready to fingerprint; add one more to triangulate');
        } else {
            setStep('sa-step-receivers', 'sa-step-receivers-status', 'attention',
                'No enabled receivers — add one to start monitoring and fingerprinting');
        }

        var monitorButton = $('sa-guide-monitor');
        if (monitorButton) {
            monitorButton.textContent = monitoring ? 'Stop monitoring' : 'Start monitoring';
        }
        if (monitoring) {
            setStep('sa-step-monitor', 'sa-step-monitor-status', 'active',
                'Running');
        } else {
            setStep('sa-step-monitor', 'sa-step-monitor-status', 'pending',
                'Not started — press Start monitoring');
        }

        var verdict = $('sa-fp-verdict');
        var estimate = $('sa-estimate');
        var located = !!(estimate && estimate.children.length > 0);
        var identified = !!(verdict && !verdict.hidden);
        var analysed = located || identified;

        // Step 3 — triangulate the location.
        if (located) {
            setStep('sa-step-triangulate', 'sa-step-triangulate-status', 'ready',
                'Located — see the fix on the map');
        } else if (enabled !== null && enabled < 2) {
            setStep('sa-step-triangulate', 'sa-step-triangulate-status', 'attention',
                'Triangulation needs at least 2 enabled receivers');
        } else if (monitoring) {
            setStep('sa-step-triangulate', 'sa-step-triangulate-status', 'active',
                'Monitoring — click the waterfall to set the target, then press Triangulate');
        } else {
            setStep('sa-step-triangulate', 'sa-step-triangulate-status', 'pending',
                'Waiting for step 2');
        }

        // Step 4 — identify the device.
        if (identified) {
            setStep('sa-step-fingerprint', 'sa-step-fingerprint-status', 'ready',
                'Identified — see the verdict in the workspace');
        } else if (monitoring) {
            setStep('sa-step-fingerprint', 'sa-step-fingerprint-status', 'active',
                'Monitoring — capture a fingerprint of a suspected source');
        } else {
            setStep('sa-step-fingerprint', 'sa-step-fingerprint-status', 'pending',
                'Waiting for step 2');
        }

        // Step 5 — review the results.
        if (analysed) {
            setStep('sa-step-analyse', 'sa-step-analyse-status', 'ready',
                'Located or identified — review in the workspace');
        } else if (monitoring) {
            setStep('sa-step-analyse', 'sa-step-analyse-status', 'active',
                'Scanning — complete the triangulation and fingerprint steps');
        } else {
            setStep('sa-step-analyse', 'sa-step-analyse-status', 'pending',
                (enabled === null || enabled < 2) ? 'Waiting for step 1' : 'Waiting for step 2');
        }

        // Step 6 — review anomalies.
        var alertsNode = $('alerts-list');
        var hasAlerts = !!(alertsNode && alertsNode.querySelectorAll('.sa-alert').length > 0);
        if (hasAlerts) {
            setStep('sa-step-anomalies', 'sa-step-anomalies-status', 'ready',
                'An anomaly is on the list — act on it below');
        } else if (monitoring) {
            setStep('sa-step-anomalies', 'sa-step-anomalies-status', 'active',
                'Monitoring — an anomaly turns up here when one is found');
        } else {
            setStep('sa-step-anomalies', 'sa-step-anomalies-status', 'pending',
                'Waiting for step 2');
        }
    }

    /**
     * Jump to a workspace control. The receiver roster and the analysis
     * controls live inside the Workspace section, which is hidden while the
     * dashboard tab is showing; scrollIntoView on a hidden element does
     * nothing, so the section is activated (and laid out) first.
     */
    function gotoWorkspace(selector, box) {
        if (typeof window.activateTab === 'function') window.activateTab('workspace');
        window.requestAnimationFrame(function () {
            var target = selector ? document.querySelector(selector) : null;
            if (target) target.scrollIntoView({ behavior: 'smooth', block: box || 'start' });
        });
    }

    /** Wire the step actions once; each one does what its label says. */
    function bindGuideActions() {
        var receiversButton = $('sa-guide-receivers');
        if (receiversButton) {
            receiversButton.onclick = function () {
                gotoWorkspace('#sa-name', 'center');
            };
        }

        var monitorButton = $('sa-guide-monitor');
        if (monitorButton) {
            monitorButton.onclick = function () {
                if (monitoringActive()) {
                    if (typeof stopMonitoring === 'function') stopMonitoring();
                } else if (typeof startMonitoring === 'function') {
                    startMonitoring();
                }
            };
        }

        var triangulateButton = $('sa-guide-triangulate');
        if (triangulateButton) {
            triangulateButton.onclick = function () {
                gotoWorkspace('#sa-triangulate');
            };
        }

        var fingerprintButton = $('sa-guide-fingerprint');
        if (fingerprintButton) {
            fingerprintButton.onclick = function () {
                gotoWorkspace('#sa-fp-capture');
            };
        }

        var workspaceButton = $('sa-guide-workspace');
        if (workspaceButton) {
            workspaceButton.onclick = function () {
                gotoWorkspace('#workspace');
            };
        }

        var anomaliesButton = $('sa-guide-anomalies');
        if (anomaliesButton) {
            anomaliesButton.onclick = function () {
                var panel = $('anomaly-panel');
                if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            };
        }
    }

    var GUIDE_KEY = 'spectrumAlertGuideHidden';

    function applyGuideVisibility() {
        var guide = $('sa-guide');
        var showButton = $('sa-guide-show');
        if (!guide) return;
        var hidden = SAUI.store.get(GUIDE_KEY, false) === true;
        guide.hidden = hidden;
        if (showButton) showButton.hidden = !hidden;
    }

    function bindGuideChrome() {
        var hideButton = $('sa-guide-hide');
        if (hideButton) {
            hideButton.onclick = function () {
                // The tour keeps every panel hidden except the step's own, and
                // it is presented by the guide, so closing the guide mid-tour
                // would leave a blank page. Exit the tour first.
                if (tourActive()) setTour(false);
                var guide = $('sa-guide');
                var showButton = $('sa-guide-show');
                if (guide) guide.hidden = true;
                if (showButton) showButton.hidden = false;
                SAUI.store.set(GUIDE_KEY, true);
                var opener = $('sa-guide-show');
                if (opener) opener.focus();
            };
        }

        var showButton = $('sa-guide-show');
        if (showButton) {
            showButton.onclick = function () {
                var guide = $('sa-guide');
                if (guide) guide.hidden = false;
                showButton.hidden = true;
                SAUI.store.set(GUIDE_KEY, false);
                refreshGuide();
                var title = $('sa-guide-title');
                if (title) title.focus && title.focus();
            };
        }
    }

    // -------------------------------------------------- guided mode ---------
    //
    // The strip normally shows every step at once. Guided mode shows one step
    // at a time with a walk-through control row: Previous, Next and progress.

    var GUIDE_MODE_KEY = 'spectrumAlertGuideMode';
    var GUIDE_STEPS = ['sa-step-receivers', 'sa-step-monitor', 'sa-step-triangulate',
        'sa-step-fingerprint', 'sa-step-analyse', 'sa-step-anomalies'];
    var guideStep = 0;

    function tourActive() {
        var guide = $('sa-guide');
        return !!guide && guide.getAttribute('data-mode') === 'tour';
    }

    /**
     * Guided focus: while a tour is running every panel on the page hides
     * except the ones tagged (data-guide-panel) for the current step, so the
     * screen shows the guide plus one thing at a time. Leaving the tour
     * restores everything.
     */
    function applyTourPanels() {
        document.body.classList.toggle('sa-tour', tourActive());
        if (!tourActive()) return;
        var stepKey = GUIDE_STEPS[guideStep].replace('sa-step-', '');
        document.querySelectorAll('[data-guide-panel]').forEach(function (panel) {
            var keys = (panel.getAttribute('data-guide-panel') || '').split(/\s+/);
            // The receiver roster and its add/remove controls live in the rail
            // and are needed at every step, so the rail never closes during the
            // tour — otherwise there is no way to add or remove a receiver once
            // the walkthrough moves past step 1.
            var open = keys.indexOf(stepKey) !== -1 || !!panel.closest('.sa-rail');
            panel.classList.toggle('is-tour-open', open);
        });
    }

    function renderTour() {
        var guide = $('sa-guide');
        if (!guide || !tourActive()) return;

        guide.querySelectorAll('.sa-step').forEach(function (step) {
            step.classList.toggle('is-active', step.id === GUIDE_STEPS[guideStep]);
        });

        var n = GUIDE_STEPS.length;
        var fill = $('sa-guide-bar-fill');
        if (fill) fill.style.width = ((guideStep + 1) / n * 100) + '%';

        var counter = $('sa-guide-counter');
        if (counter) counter.textContent = 'Step ' + (guideStep + 1) + ' of ' + n;

        var bar = $('sa-guide-bar');
        if (bar) bar.setAttribute('aria-valuenow', String(guideStep + 1));

        var prev = $('sa-guide-prev');
        var next = $('sa-guide-next');
        if (prev) prev.disabled = guideStep === 0;
        if (next) next.disabled = guideStep >= n - 1;

        applyTourPanels();
        // Deferred: this runs from a DOMContentLoaded listener registered
        // before the inline section-tab script, so window.activateTab does not
        // exist yet on a reload. Without the deferral the tour kept the first
        // step's panels hidden on the Dashboard and the page looked empty.
        window.setTimeout(followStepSection, 0);
    }

    /* A step's panels do not all live on one tab: receivers, triangulation and
       fingerprinting are workspace panels, monitoring and anomalies are
       dashboard panels. Carry the guide into whichever section the step owns,
       otherwise advancing to a workspace step hid the guide along with the
       Dashboard tab and the walkthrough lost its own controls. */
    function guideHome() {
        return document.querySelector('.tab-view[data-tab="dashboard"]');
    }

    function placeGuideIn(view) {
        var guide = $('sa-guide');
        if (!guide || !view || guide.parentElement === view) return;
        view.insertBefore(guide, view.firstChild);
    }

    function restoreGuide() {
        placeGuideIn(guideHome());
    }

    function followStepSection() {
        if (!tourActive() || typeof window.activateTab !== 'function') return;
        var stepKey = GUIDE_STEPS[guideStep].replace('sa-step-', '');
        var panel = document.querySelector('[data-guide-panel~="' + stepKey + '"]');
        if (!panel) return;
        var view = panel.closest('.tab-view');
        if (!view || !view.dataset.tab) return;
        placeGuideIn(view);
        if (!view.classList.contains('active')) {
            window.activateTab(view.dataset.tab);
        }
    }

    function goStep(delta) {
        var next = guideStep + delta;
        if (next < 0) next = 0;
        if (next >= GUIDE_STEPS.length) next = GUIDE_STEPS.length - 1;
        if (next !== guideStep) {
            guideStep = next;
            renderTour();
        }
    }

    function setTour(on) {
        var guide = $('sa-guide');
        var toggle = $('sa-guide-mode');
        var tour = $('sa-guide-tour');
        if (!guide) return;
        guide.setAttribute('data-mode', on ? 'tour' : '');
        if (toggle) {
            toggle.setAttribute('aria-pressed', String(on));
            toggle.textContent = on ? 'Exit guided mode' : 'Guided mode';
        }
        if (tour) tour.hidden = !on;
        if (on) guideStep = 0;
        SAUI.store.set(GUIDE_MODE_KEY, on);
        guide.querySelectorAll('.sa-step').forEach(function (step) {
            step.classList.toggle('is-active', on && step.id === GUIDE_STEPS[guideStep]);
        });
        applyTourPanels();
        if (on) {
            var title = $('sa-guide-title');
            if (title && title.focus) title.focus();
        } else {
            restoreGuide();
        }
    }

    function applyGuideMode() {
        var guide = $('sa-guide');
        var toggle = $('sa-guide-mode');
        var tour = $('sa-guide-tour');
        if (!guide) return;
        // Opt-in. A tour hides every panel except the current step's, so
        // defaulting it on meant the Dashboard, Config and Training tabs
        // rendered as an empty page with only the guide and the tab bar.
        var on = SAUI.store.get(GUIDE_MODE_KEY, false) === true;
        // A tour is presented by the guide; if the guide was left hidden the
        // panels would stay hidden too, so carry-over mode is not restored.
        if (on && guide.hidden) on = false;
        guide.setAttribute('data-mode', on ? 'tour' : '');
        if (toggle) {
            toggle.setAttribute('aria-pressed', String(on));
            toggle.textContent = on ? 'Exit guided mode' : 'Guided mode';
        }
        if (tour) tour.hidden = !on;
        if (on) {
            guideStep = 0;
            renderTour();
        }
        else {
            restoreGuide();
            applyTourPanels();
        }
    }

    function bindGuideMode() {
        var toggle = $('sa-guide-mode');
        if (toggle) {
            toggle.onclick = function () {
                setTour(tourActive() ? false : true);
            };
        }

        var prev = $('sa-guide-prev');
        if (prev) prev.onclick = function () { goStep(-1); };

        var next = $('sa-guide-next');
        if (next) next.onclick = function () { goStep(1); };
    }

    // ---------------------------------------------------------------- init ---

    document.addEventListener('DOMContentLoaded', function () {
        if (!window.SAUI) {
            console.error('SAUI failed to load; dashboard enhancements disabled.');
            return;
        }

        // Expose the alert buffer so the notification history and the dashboard
        // share one source of truth.
        connectionPill = SAUI.connectionPill($('sa-conn'));
        window.setConnectionState = window.setConnectionState || function () {};

        // One theme listener covers the button and the keyboard shortcut; the
        // shortcut previously toggled the palette but left every chart on the
        // old theme.
        document.addEventListener('sa:themechange', refreshChartThemes);

        registerShortcuts();
        startUptimeClock();
        renderAlerts();
        updateCounters();
        updateThreatIndicator();

        // Getting-started strip: honour the last hide/show choice, wire the
        // step actions and the guided-mode walk-through, then report the live
        // state of each step.
        applyGuideVisibility();
        bindGuideActions();
        bindGuideChrome();
        bindGuideMode();
        applyGuideMode();
        refreshGuide();
        window.setInterval(refreshGuide, 3000);

        // Hover readout on the spectrum chart: exact frequency and level under
        // the pointer. Without it the only way to read a value is to guess from
        // the shape.
        var spectrumCanvas = $('spectrum-chart');
        if (spectrumCanvas) {
            SAUI.crosshair(spectrumCanvas, {
                xLabel: 'Freq',
                yLabel: 'Level',
                formatX: function (label) { return Number(label).toFixed(3) + ' MHz'; },
                formatY: function (value) { return SAUI.format.dbfs(value); }
            });
        }

        SAUI.delegate($('alerts-list'), '[data-act]', 'click', function (event, target) {
            var row = target.closest('.sa-alert');
            if (!row) return;
            var id = row.getAttribute('data-id');
            if (target.getAttribute('data-act') === 'ack') ackAlert(id);
            else if (target.getAttribute('data-act') === 'locate') locateAlert(id);
        });

        var filter = $('alert-severity');
        if (filter) filter.addEventListener('change', renderAlerts);

        var exportBtn = $('export-alerts');
        if (exportBtn) exportBtn.addEventListener('click', exportAlerts);

        var ackAllBtn = $('ack-all-alerts');
        if (ackAllBtn) ackAllBtn.addEventListener('click', ackAll);

        var themeBtn = $('theme-toggle');
        if (themeBtn) themeBtn.addEventListener('click', function () {
            SAUI.theme.toggle();
        });

        var muteBtn = $('notify-mute');
        if (muteBtn) {
            muteBtn.addEventListener('click', toggleMute);
            var prefs = SAUI.notify.getPrefs();
            muteBtn.setAttribute('aria-pressed', String(!!prefs.muted));
            muteBtn.style.opacity = prefs.muted ? '0.5' : '';
        }

        var helpBtn = $('help-open');
        if (helpBtn) helpBtn.addEventListener('click', function () { toggleHelp(true); });
        ['help-close', 'help-close-top'].forEach(function (id) {
            var button = $(id);
            if (button) button.addEventListener('click', function () { toggleHelp(false); });
        });
        var helpOverlay = $('sa-help');
        if (helpOverlay) {
            helpOverlay.addEventListener('click', function (event) {
                if (event.target === helpOverlay) toggleHelp(false);
            });
        }

        // Adopt any alert the server pushes before this script finished loading.
        if (typeof window.loadRecentAnomalies === 'function') {
            window.loadRecentAnomalies();
        }
    });

    // Exposed for testing and for the geolocation page.
    window.dashboardExtras = {
        recordAnomaly: recordAnomaly,
        renderAlerts: renderAlerts,
        ackAlert: ackAlert,
        ackAll: ackAll,
        exportAlerts: exportAlerts,
        renderTimeline: renderTimeline,
        toggleHelp: toggleHelp,
        refreshGuide: refreshGuide,
        monitoringActive: monitoringActive,
        alerts: function () { return alerts.slice(); }
    };

    // geolocate.js owns the receiver roster and reloads it independently, so it
    // pokes the guide when the roster changes rather than waiting for the poll.
    window.SAGuideRefresh = refreshGuide;
})(window, document);
    // Collapsible panel toggles
    (function () {
        document.addEventListener('click', function (e) {
            var btn = e.target.closest('[data-toggle="panel"]');
            if (!btn) return;
            var panel = btn.closest('.panel.collapsible');
            if (!panel) return;
            var collapsed = panel.classList.toggle('collapsed');
            btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            btn.textContent = collapsed ? 'Show' : 'Hide';
        });
    })();
