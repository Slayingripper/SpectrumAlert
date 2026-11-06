/* Geolocation page controller.
 *
 * Ported from the RTL_Triangulator client, restructured and hardened.
 *
 * Patterns carried over, and why:
 *
 * - Per-row normalisation in the waterfall. Each scan is rescaled to the fixed
 *   -120..-40 dBFS display window, so a 30 dB swing between captures is visible
 *   rather than saturating the colour map.
 * - In-flight locking on the live scan. Ticks that arrive while a scan is
 *   running are dropped rather than queued, which prevents a backlog of stale
 *   scans when the SDR is slower than the poll interval.
 * - Auto-stop after N consecutive failures. A dongle unplugged mid-session would
 *   otherwise poll forever at full rate.
 * - Last-known-good receiver list. If the API call fails, the UI keeps showing
 *   the previous roster instead of blanking.
 *
 * Patterns deliberately changed:
 * - The live scan uses recursive setTimeout rather than setInterval, so the
 *   period is the interval plus the actual request latency instead of overlapping
 *   requests.
 * - Failed requests back off instead of hammering at a fixed rate.
 * - Frequency keys are sorted numerically. Lexicographic ordering put
 *   "k103700" before "k88200".
 * - Every interpolated receiver name is HTML-escaped.
 */

(function () {
    'use strict';

    // ------------------------------------------------------------------ state

    var state = {
        receivers: [],
        lastKnownReceivers: [],
        scanInFlight: false,
        consecutiveScanErrors: 0,
        liveTimer: null,
        liveIntervalMs: 800,
        editingId: null,
        lastSpectrum: null,
        lastScanAt: null,
        // Replay buffer so a change to the display floor can recolour the
        // existing waterfall instead of starting blank.
        frames: [],
        lastTriangulation: null,
        lastFingerprint: null,
        library: [],
        estimateHistory: [],
        mapLayers: { markers: [], estimate: null, circle: null, ellipse: null, heat: null, history: null },
        pmrMarkers: {},
        dbFloorInput: null,
        conn: null,
        pmr: { timer: null, running: false, cooldown: {}, intervalMs: 3000, errorIntervalMs: 6000 }
    };

    var $ = function (id) { return document.getElementById(id); };

    // ----------------------------------------------------------------- helpers

    function escapeHtml(value) {
        return SAUI.escapeHtml(value);
    }

    var fmtHz = SAUI.format.hz;
    var fmtDb = SAUI.format.dbfs;

    /** Bounded log history so the log panel can be exported. */
    var logHistory = [];

    function log(message, kind) {
        var box = $('sa-log');
        if (!box) return;
        var stamp = new Date().toISOString().substring(11, 19);
        var span = document.createElement('span');
        if (kind) span.className = kind;
        span.textContent = '[' + stamp + '] ' + message + '\n';
        box.appendChild(span);
        box.scrollTop = box.scrollHeight;

        logHistory.push({ timestamp: new Date().toISOString(), level: kind || 'info', message: message });
        if (logHistory.length > 2000) logHistory.shift();

        // Cap the DOM too; an unattended page should not grow without bound.
        while (box.childElementCount > 500) {
            if (logHistory.length > 500) logHistory.shift();
            box.removeChild(box.firstChild);
        }
    }

    function toast(message, kind) {
        // kind here is the legacy vocabulary (error/success/info); map it onto
        // the shared severity scale so notification preferences apply.
        var severity = { error: 'error', success: 'success', info: 'info', warning: 'warning' }[kind] || 'info';
        SAUI.notify.show(message, { severity: severity });
    }

    var modalState = { onCancel: null, startedAt: 0 };

    function showModal(title, body) {
        $('sa-modal-title').textContent = title || 'Working';
        $('sa-modal-body').textContent = body || '';
        $('sa-modal').classList.add('open');
        modalState.startedAt = Date.now();
        modalState.onCancel = function () { hideModal(); };
        var close = $('sa-modal-cancel');
        if (close) close.focus();
    }

    function hideModal() {
        $('sa-modal').classList.remove('open');
        modalState.onCancel = null;
    }

    /** Timed modal body for long-running operations such as calibration. */
    function tickModalElapsed() {
        var body = $('sa-modal-body');
        if (!body || !$('sa-modal').classList.contains('open')) return;
        var prefix = body.dataset.prefix || body.textContent.split('…')[0];
        body.dataset.prefix = prefix;
        body.textContent = prefix + '… elapsed ' +
            SAUI.format.duration((Date.now() - modalState.startedAt) / 1000);
    }

    async function api(path, options) {
        return SAUI.request(path, options);
    }

    function qualityColour(quality) {
        if (quality === 'good') return '#22c55e';
        if (quality === 'fair') return '#f59e0b';
        if (quality === 'poor') return '#ef4444';
        return '#8b93a3';
    }

    function setConnection(nextState) {
        // Named nextState rather than state: a `state` parameter would shadow
        // the module-level state object and this would silently never connect.
        if (state.conn) state.conn.set(nextState);
    }

    /* The footer's pill is the app-wide *data connection*, shared with
       dashboard.js and labelled "Data connection". A waterfall scan is not
       that: while the dashboard's monitoring loop holds the one dongle, a
       workspace scan legitimately fails, and reporting that as OFFLINE told
       the operator the whole app had lost its feed when the live chart was
       still streaming beside it. Scan status belongs to the scan panel, which
       already has one. */
    function setScanConnection() {}

    // -------------------------------------------------------------- waterfall

    var canvas = null, ctx = null, cursorLine = null;
    var displayFloor = -120, displayCeiling = -40;

    /**
     * Match the canvas backing store to its displayed size.
     *
     * The element carries a fixed 960x240 attribute and is stretched by CSS to
     * fill the panel. Once the page went full-bleed that meant drawing ~1630
     * device pixels from a 960-pixel buffer, which visibly softens the trace.
     * Resizing the backing store to the rendered size (times devicePixelRatio)
     * keeps one buffer column per CSS column, and the replay buffer is redrawn
     * so history survives the change.
     */
    function sizeWaterfall() {
        if (!canvas || !ctx) return;
        var rect = canvas.getBoundingClientRect();
        if (!rect.width) return;
        var dpr = Math.min(2, window.devicePixelRatio || 1);
        var width = Math.max(320, Math.round(rect.width * dpr));
        var height = Math.max(120, Math.round(rect.height * dpr));
        if (canvas.width === width && canvas.height === height) return;
        canvas.width = width;
        canvas.height = height;
        ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (state.frames.length) replayFrames(); else clearWaterfall();
    }

    function initWaterfall() {
        canvas = $('sa-waterfall');
        if (!canvas) return;
        ctx = canvas.getContext('2d', { willReadFrequently: true });
        sizeWaterfall();

        canvas.addEventListener('click', function (event) {
            if (!state.lastSpectrum) return;
            // getBoundingClientRect, not canvas.width: the canvas is CSS-scaled.
            var rect = canvas.getBoundingClientRect();
            var fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
            var span = Number($('sa-span').value) || state.lastSpectrum.spanHz || 0;
            var startHz = (state.lastSpectrum.startHz !== undefined)
                ? state.lastSpectrum.startHz
                : state.lastSpectrum.centerHz - span / 2;
            var hz = Math.round(startHz + fraction * span);
            $('sa-target-freq').value = String(hz);
            $('sa-picked').textContent = fmtHz(hz);
            log('Target set to ' + fmtHz(hz));
        });

        // Hover readout: exact frequency and level under the pointer. Without
        // this the only way to read a value is to click, which also changes the
        // target frequency.
        canvas.addEventListener('mousemove', function (event) {
            var points = state.lastSpectrum && state.lastSpectrum.points;
            if (!points || !points.length) return;
            var rect = canvas.getBoundingClientRect();
            var fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
            var index = Math.min(points.length - 1, Math.round(fraction * (points.length - 1)));
            var point = points[index];
            if (!point) return;

            if (cursorLine) {
                cursorLine.style.left = (fraction * rect.width) + 'px';
                cursorLine.classList.add('is-visible');
            }
            var floor = state.lastSpectrum.noiseFloorDbm;
            var readout = $('sa-cursor-readout');
            readout.innerHTML =
                '<div><span class="sa-readout__label">Freq</span> ' +
                '<span class="sa-readout__freq">' + escapeHtml(fmtHz(point.hz)) + '</span></div>' +
                '<div><span class="sa-readout__label">Level</span> ' +
                '<span class="sa-readout__db">' + escapeHtml(fmtDb(point.dbm)) + '</span></div>' +
                '<div><span class="sa-readout__label">Above floor</span> ' +
                (isFinite(floor) ? (point.dbm - floor).toFixed(1) + ' dB' : '--') + '</div>';
            readout.classList.add('is-visible');
            readout.setAttribute('aria-hidden', 'false');
        });

        function hideCursor() {
            if (cursorLine) cursorLine.classList.remove('is-visible');
            var readout = $('sa-cursor-readout');
            if (readout) {
                readout.classList.remove('is-visible');
                readout.setAttribute('aria-hidden', 'true');
            }
        }
        canvas.addEventListener('mouseleave', hideCursor);

        if (state.dbFloorInput) {
            state.dbFloorInput.addEventListener('input', function () {
                var value = Number(state.dbFloorInput.value);
                if (!isFinite(value)) return;
                displayFloor = Math.min(-10, value);
                displayCeiling = displayFloor + 80;
                // Re-colour the stored history. The canvas must be cleared
                // first: each draw shifts the existing image up by one row, so
                // replaying onto an already-drawn canvas would compound the
                // scroll instead of reconstructing it.
                replayFrames();
            });
        }

        if (typeof ResizeObserver !== 'undefined') {
            // The panel resizes with the viewport and with the responsive
            // breakpoints, so watch the element rather than the window.
            var observer = new ResizeObserver(function () { sizeWaterfall(); });
            observer.observe(canvas);
        } else {
            window.addEventListener('resize', sizeWaterfall);
        }

        // The workspace performs its first scan on page load, while its tab is
        // hidden. The waterfall then draws into a zero-sized canvas and the
        // tab looked empty when the operator finally opened it. Redraw once the
        // canvas actually enters the viewport.
        if (typeof IntersectionObserver === 'function') {
            new IntersectionObserver(function (entries) {
                if (!entries.some(function (entry) { return entry.isIntersecting; })) return;
                sizeWaterfall();
                if (state.frames.length) replayFrames();
            }).observe(canvas);
        }

        $('sa-set-target').addEventListener('click', function () {
            var hz = Number($('sa-target-freq').value);
            if (isFinite(hz)) $('sa-center').value = String(Math.round(hz));
        });
    }

    function dbmToColour(dbm) {
        var min = displayFloor;
        var max = displayCeiling;
        var t = Math.max(0, Math.min(1, (dbm - min) / Math.max(1e-6, max - min)));

        // Subdued thermal ramp: near-black noise -> deep blue -> teal -> amber
        // -> red at the strongest bin. The previous blue-to-red interpolation
        // painted the entire noise floor saturated blue and read like an alarm
        // even when nothing was present.
        var stops = [
            [8, 15, 28],
            [18, 48, 92],
            [32, 122, 148],
            [224, 164, 66],
            [235, 82, 72]
        ];
        var scaled = t * (stops.length - 1);
        var index = Math.min(stops.length - 2, Math.floor(scaled));
        var fraction = scaled - index;
        var a = stops[index];
        var b = stops[index + 1];
        return [
            Math.round(a[0] + (b[0] - a[0]) * fraction),
            Math.round(a[1] + (b[1] - a[1]) * fraction),
            Math.round(a[2] + (b[2] - a[2]) * fraction)
        ];
    }

    function drawScanRow(points) {
        if (!ctx || !points || !points.length) return;

        // Scroll the whole canvas up one pixel. The canvas itself is the history;
        // no separate buffer is kept.
        var existing = ctx.getImageData(0, 0, canvas.width, canvas.height);
        ctx.putImageData(existing, 0, -1);

        var row = ctx.createImageData(canvas.width, 1);
        var values = points.map(function (p) { return Number(p.dbm); })
            .filter(function (v) { return isFinite(v); });

        // Per-row normalisation: rescale into the display window so contrast is
        // preserved regardless of the absolute level or span.
        var rowMin = values.length ? Math.min.apply(null, values) : displayFloor;
        var rowMax = values.length ? Math.max.apply(null, values) : displayCeiling;
        var range = Math.max(1e-6, rowMax - rowMin);

        for (var x = 0; x < canvas.width; x++) {
            var index = Math.floor((x / (canvas.width - 1)) * Math.max(0, points.length - 1));
            var dbm = Number(points[index] && points[index].dbm);
            var normalised = isFinite(dbm) ? (displayFloor + ((dbm - rowMin) / range) * 80) : displayFloor;
            var rgb = dbmToColour(normalised);
            var offset = x * 4;
            row.data[offset] = rgb[0];
            row.data[offset + 1] = rgb[1];
            row.data[offset + 2] = rgb[2];
            row.data[offset + 3] = 255;
        }
        ctx.putImageData(row, 0, canvas.height - 1);
    }

    function clearWaterfall() {
        if (!ctx || !canvas) return;
        ctx.fillStyle = '#12141a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    /** Rebuild the waterfall from the replay buffer, newest frame at the bottom. */
    function replayFrames() {
        clearWaterfall();
        state.frames.forEach(drawScanRow);
    }

    /**
     * Top peaks above the noise floor, as a table.
     *
     * A spectrum is easier to act on when the strongest few occupants are named:
     * the chart shows shape, this says which frequencies to actually go look at.
     */
    function renderPeaks(spectrum, limit) {
        var body = $('sa-peaks');
        var wrap = $('sa-peaks-wrap');
        if (!body || !wrap) return;
        if (!spectrum || !spectrum.points || !spectrum.points.length) {
            wrap.hidden = true;
            return;
        }

        var floor = spectrum.noiseFloorDbm;
        var sorted = spectrum.points.slice().sort(function (a, b) {
            return Number(b.dbm) - Number(a.dbm);
        });

        // Keep only distinct emissions: bins within a couple of RBW of an
        // already-listed peak are sidelobes of the same signal, not new ones.
        var minSeparationHz = Math.max(spectrum.binHz * 2 || 0, 1000);
        var picked = [];
        for (var i = 0; i < sorted.length && picked.length < (limit || 8); i++) {
            var candidate = sorted[i];
            var distinct = picked.every(function (existing) {
                return Math.abs(existing.hz - candidate.hz) >= minSeparationHz;
            });
            if (distinct) picked.push(candidate);
        }

        var html = '';
        picked.forEach(function (point, index) {
            var above = isFinite(floor) ? point.dbm - floor : null;
            // Strength graded by prominence over the floor, so the marker means
            // something beyond absolute level (which varies with gain).
            var cls = above === null ? '' : (above >= 25 ? 'ok' : (above >= 12 ? '' : 'poor'));
            html += '<tr>' +
                '<td>' + (index + 1) + '</td>' +
                '<td>' + escapeHtml(fmtHz(point.hz)) + '</td>' +
                '<td class="' + cls + '">' + Number(point.dbm).toFixed(1) + '</td>' +
                '<td>' + (above === null ? '--' : above.toFixed(1) + ' dB') + '</td>' +
                '</tr>';
        });
        body.innerHTML = html;
        wrap.hidden = picked.length === 0;
    }

    function updateAxis(spectrum) {
        if (!spectrum) return;
        var span = spectrum.spanHz || 0;
        var startHz = spectrum.startHz !== undefined
            ? spectrum.startHz
            : spectrum.centerHz - span / 2;
        $('sa-axis-lo').textContent = fmtHz(startHz);
        $('sa-axis-mid').textContent = fmtHz(spectrum.centerHz);
        $('sa-axis-hi').textContent = fmtHz(startHz + span);
    }

    // ------------------------------------------------------------------- scan

    async function runScan() {
        // Drop overlapping ticks instead of queueing them.
        if (state.scanInFlight) return;

        var receiverSelect = $('sa-scan-receiver');
        var sdrId = receiverSelect && receiverSelect.value ? receiverSelect.value : null;
        if (!state.receivers.length) {
            setScanState('empty', 'No receivers configured',
                'Add a receiver in the left panel to start scanning.');
            log('Add at least one receiver before scanning.', 'warn');
            return;
        }

        state.scanInFlight = true;
        setScanConnection('connecting');
        if (!state.liveTimer) setScanState('loading', 'Scanning', null);

        try {
            var result = await api('/api/spectrum', {
                method: 'POST',
                timeoutMs: 30000,
                body: {
                    centerHz: Number($('sa-center').value),
                    spanHz: Number($('sa-span').value),
                    integrationSec: Number($('sa-integration').value),
                    bins: 256,
                    sdrId: sdrId
                }
            });

            state.consecutiveScanErrors = 0;
            state.lastScanAt = Date.now();
            state.lastSpectrum = result;
            state.lastSpectrum.startHz = result.centerHz - result.spanHz / 2;

            state.frames.push(result.points);
            // Cap the replay buffer used when the display floor changes.
            while (state.frames.length > 240) state.frames.shift();

            // Redraw the whole buffer from a cleared canvas. Drawing every
            // retained frame on top of the previous canvas compounded the
            // scrolling, so after a few scans the waterfall showed the same
            // samples smeared into bands.
            replayFrames();
            updateAxis(result);
            renderPeaks(result, 8);
            setScanState('ready');

            $('sa-peak').textContent = fmtDb(result.peakDbm);
            $('sa-floor').textContent = fmtDb(result.noiseFloorDbm);
            $('sa-rbw').textContent = isFinite(result.binHz) ? fmtHz(result.binHz) : '--';
            $('sa-source').textContent = result.source || '--';
            $('sa-freshness').textContent = 'updated ' + SAUI.format.relativeTime(state.lastScanAt);
            setScanConnection('live');

            if (result.warning) log(result.warning, 'warn');
        } catch (error) {
            state.consecutiveScanErrors += 1;
            setScanConnection('down');
            log('Scan failed: ' + error.message, 'err');

            // Distinguish "no receivers yet" (an empty state, user action
            // required) from "the backend is unreachable" (an error, retryable).
            if (!state.receivers.length) {
                setScanState('empty', 'No receivers configured',
                    'Add a receiver in the left panel to start scanning.');
            } else {
                setScanState('error', 'Scan failed', error.message, runScan);
            }

            if (state.liveTimer && state.consecutiveScanErrors >= 3) {
                stopLive('stopped after 3 consecutive failures');
            }
        } finally {
            state.scanInFlight = false;
        }
    }

    /**
     * Show a state under the waterfall. A failed scan must not look identical
     * to a quiet band, so error and empty are visually and textually distinct.
     */
    function setScanState(kind, title, message, onRetry) {
        var node = $('sa-scan-state');
        if (!node) return;
        node.className = 'sa-state sa-state--' + (kind === 'ready' ? 'ready' : kind);
        node.hidden = kind === 'ready';
        if (kind === 'ready') {
            node.innerHTML = '';
            return;
        }
        node.innerHTML = '';
        if (kind === 'loading') {
            var spinner = document.createElement('div');
            spinner.className = 'sa-spinner';
            spinner.setAttribute('aria-hidden', 'true');
            node.appendChild(spinner);
        }
        var titleNode = document.createElement('div');
        titleNode.className = 'sa-state-title';
        titleNode.textContent = title || kind;
        node.appendChild(titleNode);
        if (message) {
            var detail = document.createElement('div');
            detail.className = 'sa-state-detail';
            detail.textContent = message;
            node.appendChild(detail);
        }
        if (kind === 'error' && onRetry) {
            var retry = document.createElement('button');
            retry.className = 'sa-retry';
            retry.textContent = 'Retry';
            retry.addEventListener('click', onRetry);
            node.appendChild(retry);
        }
    }

    function scheduleLive() {
        if (!state.liveTimer) return;
        // Error backoff: stop hammering a failing device at full rate.
        var delay = state.consecutiveScanErrors > 0
            ? state.liveIntervalMs * 2
            : state.liveIntervalMs;
        // Recursive setTimeout, so the period is interval + actual latency and
        // no requests ever overlap.
        state.liveTimer = setTimeout(async function () {
            if (!state.liveTimer) return;
            await runScan();
            scheduleLive();
        }, delay);
    }

    function startLive() {
        if (state.liveTimer) return;
        state.liveTimer = setTimeout(function () {}, 0);
        state.consecutiveScanErrors = 0;
        runScan();
        scheduleLive();
        $('sa-toggle-live').textContent = 'Stop live';
        log('Live scan started', 'ok');
    }

    function stopLive(reason) {
        if (state.liveTimer) {
            clearTimeout(state.liveTimer);
            state.liveTimer = null;
        }
        $('sa-toggle-live').textContent = 'Start live';
        if (reason) log('Live scan ' + reason, 'warn');
    }

    // -------------------------------------------------------------- receivers

    function modeLabel(receiver) {
        if (receiver.mode === 'rtltcp') return receiver.host + ':' + receiver.port;
        return 'USB ' + (receiver.deviceIndex === null || receiver.deviceIndex === undefined ? 0 : receiver.deviceIndex);
    }

    /**
     * A receiver pinned to a Manual dBFS level reports one number instead of a
     * stream, so it can take part in triangulation but cannot be fingerprinted.
     */
    function isPinned(receiver) {
        return receiver.manualRssiDb !== null && receiver.manualRssiDb !== undefined;
    }

    /**
     * What the analyses can use right now. Exposed for the dashboard's
     * getting-started strip, which must not invent its own count.
     */
    function workspaceReadiness() {
        var enabled = 0;
        var enabledUnpinned = 0;
        state.receivers.forEach(function (receiver) {
            if (!receiver.enabled) return;
            enabled += 1;
            if (!isPinned(receiver)) enabledUnpinned += 1;
        });
        return {
            total: state.receivers.length,
            enabled: enabled,
            enabledUnpinned: enabledUnpinned
        };
    }
    window.workspaceReadiness = workspaceReadiness;

    /** State the cue next to Triangulate: requirement, and whether it is met. */
    function updateRequirements() {
        var ready = workspaceReadiness();
        var node = $('sa-triangulate-req');
        var text = $('sa-triangulate-req-text');
        if (node && text) {
            if (ready.enabled >= 2) {
                node.setAttribute('data-state', 'ready');
                text.textContent = ready.enabled +
                    ' enabled receiver(s) — enough to triangulate.';
            } else if (ready.enabled === 1) {
                node.setAttribute('data-state', 'pending');
                text.textContent =
                    'Triangulation needs at least 2 enabled receivers; you have 1.';
            } else {
                node.setAttribute('data-state', 'blocked');
                text.textContent =
                    'Triangulation needs at least 2 enabled receivers; you have none yet.';
            }
        }

        updateFingerprintRequirement();

        if (typeof window.SAGuideRefresh === 'function') window.SAGuideRefresh();
    }

    /** The identify panel reads IQ from whichever receiver is selected above. */
    function updateFingerprintRequirement() {
        var node = $('sa-fp-req');
        var text = $('sa-fp-req-text');
        if (!node || !text) return;

        var ready = workspaceReadiness();
        if (!ready.enabled) {
            node.setAttribute('data-state', 'blocked');
            text.textContent =
                'Fingerprinting reads the selected receiver\'s IQ — enable one receiver to start.';
            return;
        }

        var id = selectedReceiverId();
        var chosen = null;
        state.receivers.forEach(function (receiver) {
            if (String(receiver.id) === String(id)) chosen = receiver;
        });
        if (!chosen) {
            node.setAttribute('data-state', 'pending');
            text.textContent = 'Pick a receiver in the waterfall controls first.';
        } else if (isPinned(chosen)) {
            node.setAttribute('data-state', 'blocked');
            text.textContent = chosen.name + ' is pinned to a Manual dBFS level, so it has ' +
                'no IQ samples to read. Clear the pin first.';
        } else {
            node.setAttribute('data-state', 'ready');
            text.textContent = chosen.name + ' can supply IQ samples. Ready to capture.';
        }
    }

    function renderReceivers() {
        var host = $('sa-receiver-list');
        host.innerHTML = '';

        if (!state.receivers.length) {
            host.innerHTML = '<p class="sa-hint">No receivers configured. Add one below to start scanning.</p>';
            // An empty roster still answers "how many?"; a blank count read as
            // "still loading" forever.
            $('sa-receiver-count').textContent = '0 total, 0 enabled';
            updateRequirements();
            return;
        }

        state.receivers.forEach(function (receiver) {
            var row = document.createElement('div');
            row.className = 'sa-receiver' + (receiver.enabled ? '' : ' disabled');

            var details = document.createElement('div');
            var tuning = [];
            if (receiver.gainDb !== null && receiver.gainDb !== undefined) tuning.push('gain ' + receiver.gainDb + ' dB');
            if (receiver.ppmError) tuning.push('ppm ' + receiver.ppmError);
            if (receiver.manualRssiDb !== null && receiver.manualRssiDb !== undefined) {
                tuning.push('manual ' + Number(receiver.manualRssiDb).toFixed(1) + ' dBFS');
            }
            var reading = null;
            if (state.lastTriangulation && state.lastTriangulation.readings) {
                state.lastTriangulation.readings.forEach(function (r) {
                    if (r.id === receiver.id) reading = r;
                });
            }
            var rssiText = reading && isFinite(reading.rssiDbm)
                ? reading.rssiDbm.toFixed(1) + ' dBFS via ' + (reading.source || '?')
                : 'no reading yet';

            details.innerHTML =
                '<div><strong>' + escapeHtml(receiver.name) + '</strong> ' +
                '<span class="sa-badge ' + (receiver.enabled ? 'ok' : 'fail') + '">' +
                (receiver.enabled ? 'on' : 'off') + '</span></div>' +
                '<div class="meta">' + escapeHtml(modeLabel(receiver)) + ' &middot; ' +
                Number(receiver.lat).toFixed(5) + ', ' + Number(receiver.lon).toFixed(5) + '</div>' +
                (tuning.length ? '<div class="meta">' + escapeHtml(tuning.join(' &middot; ')) + '</div>' : '') +
                '<div class="meta">' + escapeHtml(rssiText) + '</div>';

            var actions = document.createElement('div');
            actions.className = 'actions';

            var toggle = document.createElement('button');
            toggle.className = 'sa-btn tiny';
            toggle.textContent = receiver.enabled ? 'Disable' : 'Enable';
            toggle.addEventListener('click', async function () {
                // Optimistic toggle with rollback on failure.
                var desired = !receiver.enabled;
                receiver.enabled = desired;
                renderReceivers();
                try {
                    await api('/api/sdrs', { method: 'POST', body: receiver });
                    await loadReceivers();
                } catch (error) {
                    receiver.enabled = !desired;
                    renderReceivers();
                    toast('Could not toggle receiver: ' + error.message, 'error');
                }
            });
            actions.appendChild(toggle);

            var check = document.createElement('button');
            check.className = 'sa-btn tiny';
            check.textContent = 'Check';
            check.addEventListener('click', async function () {
                check.disabled = true;
                try {
                    var result = await api('/api/sdrs/' + encodeURIComponent(receiver.id) + '/check', {
                        method: 'POST',
                        body: { centerHz: Number($('sa-center').value) }
                    });
                    toast(result.message, result.ok ? 'success' : 'error');
                    log(receiver.name + ': ' + result.message, result.ok ? 'ok' : 'err');
                } catch (error) {
                    toast(receiver.name + ': ' + error.message, 'error');
                } finally {
                    check.disabled = false;
                }
            });
            actions.appendChild(check);

            var edit = document.createElement('button');
            edit.className = 'sa-btn tiny';
            edit.textContent = 'Edit';
            edit.addEventListener('click', function () { startEdit(receiver); });
            actions.appendChild(edit);

            var remove = document.createElement('button');
            remove.className = 'sa-btn tiny danger';
            remove.textContent = 'Remove';
            remove.addEventListener('click', async function () {
                if (!confirm('Remove receiver "' + receiver.name + '"?')) return;
                try {
                    await api('/api/sdrs/' + encodeURIComponent(receiver.id), { method: 'DELETE' });
                    if (state.editingId === receiver.id) cancelEdit();
                    // Drop the stale scan: it came from hardware that is gone.
                    if (state.lastSpectrum && state.lastSpectrum.sdrId === receiver.id) {
                        state.lastSpectrum = null;
                        state.frames = [];
                        replayFrames();
                        renderPeaks(null);
                        $('sa-peaks-wrap').hidden = true;
                        $('sa-peak').textContent = '--';
                        $('sa-rbw').textContent = '--';
                        $('sa-freshness').textContent = '';
                        setScanState('empty', 'Receiver removed',
                            'Add a receiver to resume scanning.');
                    }
                    await loadReceivers();
                    toast('Receiver removed', 'info');
                } catch (error) {
                    toast('Remove failed: ' + error.message, 'error');
                }
            });
            actions.appendChild(remove);

            row.appendChild(details);
            row.appendChild(actions);
            host.appendChild(row);
        });

        $('sa-receiver-count').textContent =
            state.receivers.length + ' total, ' +
            state.receivers.filter(function (r) { return r.enabled; }).length + ' enabled';
        updateRequirements();
    }

    function populateReceiverSelect() {
        var select = $('sa-scan-receiver');
        var previous = select.value;
        select.innerHTML = '';
        state.receivers.filter(function (r) { return r.enabled; }).forEach(function (receiver) {
            var option = document.createElement('option');
            option.value = receiver.id;
            option.textContent = receiver.name + ' (' + receiver.mode + ')';
            select.appendChild(option);
        });
        // Preserve the selection when possible.
        var exists = state.receivers.some(function (r) { return r.id === previous; });
        if (exists) select.value = previous;
        updateFingerprintRequirement();
    }

    async function loadReceivers() {
        try {
            var data = await api('/api/sdrs');
            state.receivers = Array.isArray(data.sdrs) ? data.sdrs : [];
            state.lastKnownReceivers = state.receivers.slice();
        } catch (error) {
            // Keep showing the last good roster rather than blanking the UI.
            state.receivers = state.lastKnownReceivers;
            toast('Could not load receivers: ' + error.message, 'error');
        }
        populateReceiverSelect();
        renderReceivers();
        renderMapReceivers();
    }

    function updateModeFields() {
        var mode = $('sa-mode').value;
        document.querySelectorAll('.sa-field-group').forEach(function (group) {
            var modes = (group.getAttribute('data-modes') || '').split(',');
            group.hidden = modes.indexOf(mode) === -1;
        });
    }

    function startEdit(receiver) {
        state.editingId = receiver.id;
        $('sa-form-title').textContent = 'Edit receiver';
        $('sa-save-receiver').textContent = 'Save receiver';
        $('sa-form-cancel').hidden = false;
        $('sa-name').value = receiver.name;
        $('sa-mode').value = receiver.mode;
        $('sa-lat').value = receiver.lat;
        $('sa-lon').value = receiver.lon;
        $('sa-device').value = receiver.deviceIndex === null || receiver.deviceIndex === undefined ? 0 : receiver.deviceIndex;
        $('sa-host').value = receiver.host || '';
        $('sa-port').value = receiver.port || 1234;
        $('sa-gain').value = receiver.gainDb === null || receiver.gainDb === undefined ? '' : receiver.gainDb;
        $('sa-ppm').value = receiver.ppmError || '';
        $('sa-manual').value = receiver.manualRssiDb === null || receiver.manualRssiDb === undefined
            ? '' : receiver.manualRssiDb;
        updateModeFields();
    }

    function cancelEdit() {
        state.editingId = null;
        $('sa-form-title').textContent = 'Add receiver';
        $('sa-save-receiver').textContent = 'Add receiver';
        $('sa-form-cancel').hidden = true;
        $('sa-name').value = '';
        $('sa-lat').value = '';
        $('sa-lon').value = '';
        $('sa-host').value = '';
        $('sa-port').value = '1234';
        $('sa-gain').value = '';
        $('sa-ppm').value = '';
        $('sa-manual').value = '';
        $('sa-device').value = '0';
        updateModeFields();
    }

    async function saveReceiver() {
        var lat = Number($('sa-lat').value);
        var lon = Number($('sa-lon').value);
        if (!isFinite(lat) || lat < -90 || lat > 90) { toast('Latitude must be -90..90', 'error'); return; }
        if (!isFinite(lon) || lon < -180 || lon > 180) { toast('Longitude must be -180..180', 'error'); return; }

        var payload = {
            name: $('sa-name').value.trim(),
            mode: $('sa-mode').value,
            lat: lat,
            lon: lon,
            deviceIndex: $('sa-mode').value === 'local' ? Number($('sa-device').value || 0) : null,
            host: $('sa-mode').value === 'rtltcp' ? $('sa-host').value.trim() : null,
            port: $('sa-mode').value === 'rtltcp' ? Number($('sa-port').value || 1234) : null,
            gainDb: $('sa-gain').value === '' ? null : Number($('sa-gain').value),
            ppmError: $('sa-ppm').value === '' ? null : Number($('sa-ppm').value),
            manualRssiDb: $('sa-manual').value === '' ? null : Number($('sa-manual').value),
            enabled: true
        };
        if (!payload.name) { toast('Name is required', 'error'); return; }
        if (state.editingId) payload.id = state.editingId;

        try {
            await api('/api/sdrs', { method: 'POST', body: payload });
            toast(payload.name + ' saved', 'success');
            cancelEdit();
            var hadReceivers = state.receivers.length > 0;
            await loadReceivers();

            // Adding the first receiver is the moment the page becomes usable,
            // so scan straight away. Without this the user adds hardware and
            // then has to notice and press "Scan once" themselves.
            if (!hadReceivers && state.receivers.length && !state.lastSpectrum) {
                log('First receiver added; starting the first scan');
                runScan();
            }
        } catch (error) {
            toast('Save failed: ' + error.message, 'error');
        }
    }

    // -------------------------------------------------------------------- map

    var map = null;

    function initMap() {
        if (map || typeof L === 'undefined') return;
        map = L.map('sa-map').setView([35.1, 33.35], 12);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);

        /* The workspace tab is display:none on load, so the map's container
           measured 0x0 when Leaflet sized itself and the result was a
           postage-stamp of tiles floating in the middle of an empty 672px
           black box. Two watchers fix it: one on the container for resizes and
           panel drags, one on the section so the first activation is caught
           even though the size never changed while the tab was hidden. */
        var host = document.getElementById('sa-map');
        if (!host) return;

        function resize() {
            if (!map) return;
            var box = host.getBoundingClientRect();
            if (box.width < 1 || box.height < 1) return;
            map.invalidateSize({ animate: false });
            redrawMap();
        }

        if (typeof ResizeObserver === 'function') new ResizeObserver(resize).observe(host);

        var section = document.getElementById('workspace');
        if (section && typeof MutationObserver === 'function') {
            new MutationObserver(function () {
                if (section.offsetParent !== null) resize();
            }).observe(section, { attributes: true, attributeFilter: ['class'] });
        }
    }

    function clearLayer(key) {
        if (state.mapLayers[key] && map) {
            map.removeLayer(state.mapLayers[key]);
            state.mapLayers[key] = null;
        }
    }

    function renderMapReceivers() {
        if (!map) return;
        state.mapLayers.markers.forEach(function (layer) { map.removeLayer(layer); });
        state.mapLayers.markers = [];
        if (!state.receivers.length) return;

        var bounds = [];
        state.receivers.forEach(function (receiver) {
            // A receiver saved before its position was set has no coordinates.
            // L.marker rejects undefined outright, and that one bad row took
            // down the whole map render for every other receiver.
            var lat = Number(receiver.lat);
            var lon = Number(receiver.lon);
            if (!isFinite(lat) || !isFinite(lon)) {
                log('Receiver "' + receiver.name +
                    '" has no position yet; it is not shown on the map.',
                    'warn');
                return;
            }
            var marker = L.marker([lat, lon])
                .bindPopup('<strong>' + escapeHtml(receiver.name) + '</strong><br>' + escapeHtml(receiver.mode))
                .addTo(map);
            state.mapLayers.markers.push(marker);
            bounds.push([lat, lon]);
        });
        if (bounds.length > 1) {
            map.fitBounds(bounds, { padding: [30, 30], maxZoom: 15 });
        }
    }

    function renderEstimate(result) {
        if (!map || !result || !result.estimate) return;
        var estimate = result.estimate;

        // Same guard as the receiver markers: a fix without a position would
        // throw inside Leaflet and abandon the whole render.
        if (!isFinite(Number(estimate.lat)) || !isFinite(Number(estimate.lon))) {
            toast('The fix did not include a position; map layers were left unchanged.', 'error');
            return;
        }

        var colour = qualityColour(estimate.quality);

        ['estimate', 'circle', 'ellipse', 'heat'].forEach(clearLayer);

        // Confidence ellipse as a polyline: Leaflet has no rotated ellipse.
        if ($('sa-show-ellipse').checked && estimate.confidence &&
            isFinite(estimate.confidence.semiMajorMeters) &&
            isFinite(estimate.confidence.semiMinorMeters)) {
            var angle = (estimate.confidence.angleDeg || 0) * Math.PI / 180;
            var points = [];
            for (var a = 0; a <= 360; a += 6) {
                var radians = a * Math.PI / 180;
                var ex = estimate.confidence.semiMinorMeters * Math.cos(radians);
                var ey = estimate.confidence.semiMajorMeters * Math.sin(radians);
                var rx = ex * Math.cos(angle) - ey * Math.sin(angle);
                var ry = ex * Math.sin(angle) + ey * Math.cos(angle);
                points.push([
                    estimate.lat + ry / 111320,
                    estimate.lon + rx / (111320 * Math.cos(estimate.lat * Math.PI / 180))
                ]);
            }
            state.mapLayers.ellipse = L.polyline(points, {
                color: colour, weight: 1, opacity: 0.6, fill: false
            }).addTo(map);
        }

        if (isFinite(estimate.uncertaintyMeters) && estimate.uncertaintyMeters > 0) {
            state.mapLayers.circle = L.circle([estimate.lat, estimate.lon], {
                radius: estimate.uncertaintyMeters, color: colour,
                fillColor: colour, fillOpacity: 0.12, weight: 1
            }).addTo(map);
        }

        var popup = [
            '<strong>Estimated emitter</strong>',
            'Lat ' + estimate.lat.toFixed(6) + ', Lon ' + estimate.lon.toFixed(6),
            'Uncertainty &plusmn;' + Math.round(estimate.uncertaintyMeters) + ' m',
            'RMS residual ' + Math.round(estimate.rmsResidualMeters) + ' m (' +
                (estimate.rmsRssiResidual || 0).toFixed(1) + ' dB)',
            'Quality <strong>' + escapeHtml(estimate.quality) + '</strong>'
        ];
        if (estimate.residuals) {
            estimate.residuals.forEach(function (r) {
                popup.push(escapeHtml(r.name) + ': ' + r.measuredRssi + ' measured / ' +
                    r.predictedRssi + ' predicted (' + (r.residualDb > 0 ? '+' : '') +
                    r.residualDb + ' dB)');
            });
        }
        if (estimate.qualityReasons) {
            popup.push('<em>' + escapeHtml(estimate.qualityReasons.join('; ')) + '</em>');
        }

        state.mapLayers.estimate = L.circleMarker([estimate.lat, estimate.lon], {
            radius: 8, color: colour, fillColor: colour, fillOpacity: 0.9, weight: 2
        }).bindPopup(popup.join('<br>')).addTo(map);

        // Temporal outlier rejection on the client as a second opinion.
        if (state.estimateHistory.length >= 3) {
            var recent = state.estimateHistory.slice(-5);
            var meanLat = recent.reduce(function (s, p) { return s + p.lat; }, 0) / recent.length;
            var meanLon = recent.reduce(function (s, p) { return s + p.lon; }, 0) / recent.length;
            if (Math.abs(estimate.lat - meanLat) > 0.05 || Math.abs(estimate.lon - meanLon) > 0.05) {
                toast('Estimate jumped over 5 km from recent fixes; treated as an outlier', 'error');
            }
        }

        state.estimateHistory.push({
            lat: estimate.lat, lon: estimate.lon,
            uncertaintyMeters: estimate.uncertaintyMeters, quality: estimate.quality
        });
        if (state.estimateHistory.length > 50) state.estimateHistory.shift();
        renderHistory();

        if ($('sa-show-heatmap').checked) renderHeatmap(result);
        renderEstimateTable(result);
    }

    function renderHistory() {
        if (!map || !$('sa-show-history').checked) { clearLayer('history'); return; }
        clearLayer('history');
        var group = L.layerGroup().addTo(map);
        var history = state.estimateHistory;
        history.forEach(function (point, index) {
            var age = index / Math.max(1, history.length - 1);
            var colour = qualityColour(point.quality);
            L.circleMarker([point.lat, point.lon], {
                radius: Math.max(3, Math.min(24, (point.uncertaintyMeters || 500) * 0.05)),
                color: colour, fillColor: colour,
                fillOpacity: 0.08 + 0.25 * age, weight: 0
            }).addTo(group);
        });
        state.mapLayers.history = group;
    }

    function renderHeatmap(result) {
        if (!map || typeof L === 'undefined' || !L.heatLayer) return;
        clearLayer('heat');
        var readings = (result.readings || []).filter(function (r) { return isFinite(r.rssiDbm); });
        if (!readings.length) return;

        var model = result.model || {};
        var reference = model.referenceRssiDbm !== undefined ? model.referenceRssiDbm : -40;
        var exponent = model.pathLossExponent !== undefined ? model.pathLossExponent : 3.5;

        var lats = readings.map(function (r) { return r.lat; });
        var lons = readings.map(function (r) { return r.lon; });
        var midLat = (Math.min.apply(null, lats) + Math.max.apply(null, lats)) / 2;
        var latPad = Math.max((Math.max.apply(null, lats) - Math.min.apply(null, lats)) * 0.35, 500 / 111320);
        var lonPad = Math.max((Math.max.apply(null, lons) - Math.min.apply(null, lons)) * 0.35,
            500 / (111320 * Math.cos(midLat * Math.PI / 180)));
        var minLat = Math.min.apply(null, lats) - latPad, maxLat = Math.max.apply(null, lats) + latPad;
        var minLon = Math.min.apply(null, lons) - lonPad, maxLon = Math.max.apply(null, lons) + lonPad;

        var points = [], best = -Infinity, worst = Infinity;
        for (var yi = 0; yi < 36; yi++) {
            var lat = minLat + (yi / 35) * (maxLat - minLat);
            for (var xi = 0; xi < 36; xi++) {
                var lon = minLon + (xi / 35) * (maxLon - minLon);
                var error = 0;
                readings.forEach(function (r) {
                    var modelled = Math.pow(10, (reference - r.rssiDbm) / (10 * exponent));
                    var actual = haversine(lat, lon, r.lat, r.lon);
                    error += Math.abs(actual - modelled);
                });
                var typical = readings.reduce(function (sum, r) {
                    return sum + Math.pow(10, (reference - r.rssiDbm) / (10 * exponent));
                }, 0) / readings.length;
                var score = Math.exp(-error / Math.max(250, typical * 1.5));
                best = Math.max(best, score);
                worst = Math.min(worst, score);
                points.push([lat, lon, score]);
            }
        }
        var span = Math.max(1e-9, best - worst);
        state.mapLayers.heat = L.heatLayer(
            points.map(function (p) { return [p[0], p[1], (p[2] - worst) / span]; }),
            { radius: 28, blur: 22, maxZoom: 18, minOpacity: 0.3,
              gradient: { 0.2: '#1d4ed8', 0.45: '#0ea5e9', 0.7: '#22c55e', 0.85: '#f59e0b', 1.0: '#ef4444' } }
        ).addTo(map);
    }

    function haversine(lat1, lon1, lat2, lon2) {
        var R = 6371000, toRad = function (d) { return d * Math.PI / 180; };
        var dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
        var a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
        var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    function renderEstimateTable(result) {
        var host = $('sa-estimate');
        var estimate = result.estimate;
        var warnings = (result.measurement && result.measurement.warnings) || [];

        var html = '<table class="sa-table"><thead><tr>' +
            '<th>Receiver</th><th>dBFS</th><th>Floor</th><th>Prom</th><th>Rel</th><th>Model m</th><th>Resid dB</th>' +
            '</tr></thead><tbody>';

        var byName = {};
        (estimate.residuals || []).forEach(function (r) { byName[r.name] = r; });

        (result.readings || []).forEach(function (r) {
            var residual = byName[r.name];
            html += '<tr><td>' + escapeHtml(r.name) + '</td>' +
                '<td>' + (isFinite(r.rssiDbm) ? r.rssiDbm.toFixed(1) : '—') + '</td>' +
                '<td>' + (isFinite(r.noiseFloorDbm) ? r.noiseFloorDbm.toFixed(1) : '—') + '</td>' +
                '<td>' + (isFinite(r.prominenceDb) ? r.prominenceDb.toFixed(1) : '—') + '</td>' +
                '<td>' + (isFinite(r.reliability) ? r.reliability.toFixed(2) : '—') + '</td>' +
                '<td>' + (isFinite(r.modeledDistMeters) ? Math.round(r.modeledDistMeters) : '—') + '</td>' +
                '<td>' + (residual ? (residual.residualDb > 0 ? '+' : '') + residual.residualDb.toFixed(1) : '—') + '</td></tr>';
        });
        html += '</tbody></table>';

        var quality = String(estimate.quality || 'unknown');
        var reasons = (estimate.qualityReasons || []).join('; ');
        html += '<p class="sa-fit">' +
            '<span class="sa-fit__label">Fit quality</span>' +
            '<strong class="sa-fit__quality" data-quality="' + escapeHtml(quality) + '">' +
                escapeHtml(quality) + '</strong>' +
            '<span>Uncertainty &plusmn;' +
                Math.round(Number(estimate.uncertaintyMeters) || 0) + ' m</span>' +
            '<span>RMS ' + Math.round(Number(estimate.rmsResidualMeters) || 0) + ' m / ' +
                (Number(estimate.rmsRssiResidual || 0)).toFixed(1) + ' dB</span>' +
            '<span>' + (Number(estimate.anchorCount) || 0) + ' receivers</span>' +
            (reasons ? '<span class="sa-fit__reason">' + escapeHtml(reasons) + '</span>' : '') +
            '</p>';

        html += '<p class="sa-hint">Model: RSSI = ' + result.model.referenceRssiDbm +
            ' &minus; 10&times;' + result.model.pathLossExponent + '&times;log&#8321;&#8320;(m), band ' +
            escapeHtml(result.model.band || '?') + '</p>';

        if (warnings.length) {
            html += '<div class="sa-log" style="margin-top:8px">';
            warnings.forEach(function (w) { html += '<span class="warn">! ' + escapeHtml(w) + '\n</span>'; });
            html += '</div>';
        }
        host.innerHTML = html;
    }

    // ---------------------------------------------------------- triangulation

    async function triangulate() {
        var button = $('sa-triangulate');
        button.disabled = true;
        showModal('Triangulating', 'Scanning all enabled receivers…');
        try {
            var result = await api('/api/triangulate', {
                method: 'POST',
                timeoutMs: 60000,
                body: {
                    frequencyHz: Number($('sa-target-freq').value),
                    minReliability: 0.2
                }
            });
            state.lastTriangulation = result;
            renderEstimate(result);
            renderReceivers();
            log('Fix at ' + fmtHz(result.frequencyHz) + ': quality ' + result.estimate.quality +
                ', ±' + Math.round(result.estimate.uncertaintyMeters) + ' m', 'ok');
            toast('Fix quality: ' + result.estimate.quality, result.estimate.quality === 'good' ? 'success' : 'info');
        } catch (error) {
            log('Triangulation failed: ' + error.message, 'err');
            toast('Triangulation failed: ' + error.message, 'error');
        } finally {
            hideModal();
            button.disabled = false;
        }
    }

    // ------------------------------------------------------------ calibration

    function renderCalibration(calibration) {
        var host = $('sa-calibration-table');
        var offsets = (calibration && calibration.offsets) || {};
        var receiverIds = Object.keys(offsets);

        if (!receiverIds.length) {
            host.innerHTML = '<p class="sa-hint">No offsets stored. Preview then apply to generate them.</p>';
            return;
        }

        var html = '<table class="sa-table"><thead><tr><th>Receiver</th><th>Frequency</th><th>Offset dB</th><th></th></tr></thead><tbody>';
        receiverIds.forEach(function (id) {
            var receiver = null;
            state.receivers.forEach(function (r) { if (r.id === id) receiver = r; });
            var name = receiver ? receiver.name : id.substring(0, 8);
            // Sort numerically. Lexicographic ordering put 103.7 MHz before 88.2 MHz.
            var keys = Object.keys(offsets[id]).sort(function (a, b) {
                return parseInt(a.replace('k', ''), 10) - parseInt(b.replace('k', ''), 10);
            });
            keys.forEach(function (key, index) {
                var hz = parseInt(key.replace('k', ''), 10) * 1000;
                html += '<tr>' +
                    '<td>' + (index === 0 ? escapeHtml(name) : '') + '</td>' +
                    '<td>' + (hz / 1e6).toFixed(4) + ' MHz</td>' +
                    '<td>' + Number(offsets[id][key]).toFixed(2) + '</td>' +
                    '<td>' + (index === keys.length - 1
                        ? '<button class="sa-btn tiny danger sa-cal-clear" data-id="' + escapeHtml(id) + '">clear</button>'
                        : '') + '</td></tr>';
            });
        });
        html += '</tbody></table>';
        host.innerHTML = html;

        host.querySelectorAll('.sa-cal-clear').forEach(function (button) {
            button.addEventListener('click', async function () {
                try {
                    var updated = await api('/api/calibration/' + encodeURIComponent(button.getAttribute('data-id')),
                        { method: 'DELETE' });
                    renderCalibration(updated);
                    toast('Offsets cleared for receiver', 'info');
                } catch (error) {
                    toast('Clear failed: ' + error.message, 'error');
                }
            });
        });
    }

    async function runCalibration(apply) {
        var button = apply ? $('sa-apply-calibration') : $('sa-preview-calibration');
        button.disabled = true;
        showModal(apply ? 'Applying calibration' : 'Previewing calibration',
            'Measuring known transmitters');
        var timer = window.setInterval(tickModalElapsed, 1000);

        try {
            var result = await api('/api/calibrate', {
                method: 'POST',
                timeoutMs: 180000,
                body: { apply: !!apply }
            });
            if (apply) {
                renderCalibration(result.calibration);
                toast('Calibration applied', 'success');
                log('Calibration applied across ' + Object.keys(result.previewOffsets).length + ' receiver(s)', 'ok');
            } else {
                var lines = ['=== Calibration preview ==='];
                Object.keys(result.previewOffsets).forEach(function (id) {
                    var receiver = null;
                    state.receivers.forEach(function (r) { if (r.id === id) receiver = r; });
                    lines.push((receiver ? receiver.name : id) + ':');
                    Object.keys(result.previewOffsets[id]).forEach(function (key) {
                        var entry = result.previewOffsets[id][key];
                        lines.push('  ' + (parseInt(key.replace('k', ''), 10) / 1000).toFixed(3) +
                            ' MHz  ' + Number(entry.offset).toFixed(2) + ' dB');
                    });
                });
                (result.rejected || []).forEach(function (r) { lines.push('rejected: ' + r); });
                lines.push('');
                lines.push('Press "Apply offsets" to store these.');
                var box = document.createElement('div');
                box.className = 'sa-log';
                box.textContent = lines.join('\n');
                $('sa-calibration-table').innerHTML = '';
                $('sa-calibration-table').appendChild(box);
                toast('Preview ready — review before applying', 'info');
            }
        } catch (error) {
            toast('Calibration failed: ' + error.message, 'error');
            log('Calibration failed: ' + error.message, 'err');
        } finally {
            clearInterval(timer);
            hideModal();
            button.disabled = false;
        }
    }

    async function fitPle() {
        var button = $('sa-fit-ple');
        button.disabled = true;
        showModal('Fitting path-loss exponent', 'Measuring reference transmitters across receivers…');
        try {
            var result = await api('/api/ple/fit', { method: 'POST', timeoutMs: 180000, body: { apply: true } });
            var lines = [
                '=== PLE fit ===',
                'n  = ' + Number(result.pathLossExponent).toFixed(3),
                'P0 = ' + Number(result.referenceRssiDbm).toFixed(2) + ' dBm',
                'R2 = ' + Number(result.r2).toFixed(4) +
                    '  (' + (result.points || []).filter(function (p) { return p.usedInFit; }).length +
                    '/' + (result.points || []).length + ' points used)'
            ];
            if (result.warning) lines.push('! ' + result.warning);
            if (result.fallbackApplied) lines.push('(band defaults were used; the data could not support a fit)');
            lines.push('');
            lines.push('  TX'.padEnd(24) + 'DIST(km)'.padEnd(10) + 'dBFS'.padEnd(9) + 'RESID'.padEnd(8) + 'IN?');
            lines.push('  ' + '-'.repeat(60));
            (result.points || []).forEach(function (p) {
                lines.push('  ' + String(p.transmitter).substring(0, 22).padEnd(22) +
                    (p.distMeters / 1000).toFixed(2).padEnd(10) +
                    Number(p.adjusted).toFixed(1).padEnd(9) +
                    (p.residualDb === null || p.residualDb === undefined ? '' : Number(p.residualDb).toFixed(1)).padEnd(8) +
                    (p.usedInFit ? 'Y' : 'N'));
            });
            (result.errors || []).forEach(function (e) { lines.push('  ! ' + e); });

            var box = document.createElement('div');
            box.className = 'sa-log';
            box.textContent = lines.join('\n');
            $('sa-calibration-table').innerHTML = '';
            $('sa-calibration-table').appendChild(box);
            toast(result.fallbackApplied
                ? 'Fit fell back to band defaults'
                : 'Fit n=' + Number(result.pathLossExponent).toFixed(2), 'info');
        } catch (error) {
            toast('PLE fit failed: ' + error.message, 'error');
        } finally {
            hideModal();
            button.disabled = false;
        }
    }

    // ------------------------------------------------------------------- PMR

    var PMR_CHANNELS = (function () {
        var colours = ['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#22c55e',
            '#10b981', '#14b8a6', '#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1',
            '#8b5cf6', '#a855f7', '#d946ef', '#ec4899'];
        var channels = [];
        for (var i = 0; i < 16; i++) {
            channels.push({ ch: i + 1, freqHz: 446006250 + i * 12500, colour: colours[i] });
        }
        return channels;
    })();

    var PMR_COOLDOWN_MS = 30000;

    function initPmrGrid() {
        var grid = $('sa-pmr-grid');
        grid.innerHTML = '';
        PMR_CHANNELS.forEach(function (channel) {
            var cell = document.createElement('div');
            cell.className = 'sa-pmr-cell';
            cell.id = 'sa-pmr-' + channel.ch;
            cell.innerHTML = '<div class="ch">CH' + channel.ch + '</div><div class="dbm">—</div>';
            cell.title = 'CH' + channel.ch + ' ' + (channel.freqHz / 1e6).toFixed(5) + ' MHz';
            grid.appendChild(cell);
        });
    }

    function updatePmrCell(channel, dbm, active, tri) {
        var cell = document.getElementById('sa-pmr-' + channel);
        if (!cell) return;
        cell.querySelector('.dbm').textContent =
            (dbm === null || dbm === undefined || !isFinite(dbm)) ? '—' : dbm.toFixed(0);
        cell.className = 'sa-pmr-cell' + (tri ? ' tri' : (active ? ' active' : ''));
        cell.style.borderColor = (active || tri) ? channel.colour : '#2b313d';
    }

    async function pollPmr() {
        if (!state.pmr.running) return;
        try {
            var result = await api('/api/pmr/scan', {
                method: 'POST',
                timeoutMs: 30000,
                body: { triggerDeltaDb: 20, minStrongReceivers: 3 }
            });

            // Per-channel readings across receivers, plus a majority vote.
            var perChannel = {};
            (result.receivers || []).forEach(function (receiver) {
                (receiver.channels || []).forEach(function (ch) {
                    perChannel[ch.ch] = perChannel[ch.ch] || [];
                    perChannel[ch.ch].push({
                        sdrId: receiver.sdrId, name: receiver.name,
                        lat: receiver.lat, lon: receiver.lon,
                        rssiDbm: ch.rssiDbm, noiseFloor: receiver.noiseFloorDbm,
                        source: receiver.source
                    });
                });
            });

            var triggered = [];
            PMR_CHANNELS.forEach(function (channel) {
                var readings = perChannel[channel.ch] || [];
                var strong = readings.filter(function (r) {
                    return isFinite(r.rssiDbm) && isFinite(r.noiseFloor) &&
                        (r.rssiDbm - r.noiseFloor) >= result.triggerDeltaDb;
                });
                var active = strong.length >= result.minStrongReceivers;
                if (active) triggered.push(channel.ch);
                updatePmrCell(channel,
                    readings.length ? readings[0].rssiDbm : null, active, false);
            });

            // More than a few channels active at once means interference, not
            // traffic. Discard the whole batch rather than emitting a burst of
            // bogus positions.
            if (triggered.length > result.maxSimultaneousChannels) {
                $('sa-pmr-status').textContent =
                    triggered.length + ' channels active simultaneously — broadband interference, discarded';
                PMR_CHANNELS.forEach(function (c) { updatePmrCell(c, null, false, false); });
            } else {
                $('sa-pmr-status').textContent =
                    new Date().toLocaleTimeString() + ' — ' + triggered.length + ' channel(s) active';
                for (var i = 0; i < triggered.length; i++) {
                    var ch = triggered[i];
                    var now = Date.now();
                    if (now - (state.pmr.cooldown[ch] || 0) < PMR_COOLDOWN_MS) continue;
                    if ((perChannel[ch] || []).length < 2) continue;
                    // Claim the cooldown before awaiting, so a slow request does
                    // not get re-entered on the next tick.
                    state.pmr.cooldown[ch] = now;
                    await triangulatePmr(perChannel[ch], ch);
                }
            }

            schedulePmr(state.pmr.intervalMs);
        } catch (error) {
            $('sa-pmr-status').textContent = 'error: ' + error.message;
            schedulePmr(state.pmr.errorIntervalMs);
        }
    }

    function schedulePmr(delay) {
        if (!state.pmr.running) return;
        state.pmr.timer = setTimeout(pollPmr, delay);
    }

    async function triangulatePmr(readings, channel) {
        var payload = readings.map(function (r) {
            return { id: r.sdrId, name: r.name, lat: r.lat, lon: r.lon, rssiDbm: r.rssiDbm, source: r.source };
        });
        try {
            var result = await api('/api/triangulate/readings', {
                method: 'POST',
                body: {
                    readings: payload,
                    referenceRssiDbm: -40,
                    pathLossExponent: 3.5
                }
            });
            var estimate = result.estimate;
            if (isFinite(estimate.lat)) {
                var colour = qualityColour(estimate.quality);
                if (state.pmrMarkers && state.pmrMarkers[channel]) {
                    map.removeLayer(state.pmrMarkers[channel]);
                }
                state.pmrMarkers = state.pmrMarkers || {};
                state.pmrMarkers[channel] = L.circleMarker([estimate.lat, estimate.lon], {
                    radius: 10, color: colour, fillColor: colour, fillOpacity: 0.4, weight: 2
                }).bindTooltip(
                    'CH' + channel + ' · ' + estimate.quality + ' · ±' +
                    Math.round(estimate.uncertaintyMeters) + ' m',
                    { permanent: false, direction: 'top' }
                ).addTo(map);
                updatePmrCell(PMR_CHANNELS[channel - 1], readings[0].rssiDbm, true, true);
                log('PMR CH' + channel + ' triangulated: ' + estimate.quality +
                    ' ±' + Math.round(estimate.uncertaintyMeters) + ' m', 'ok');
            }
        } catch (error) {
            log('PMR CH' + channel + ' triangulation failed: ' + error.message, 'err');
        }
    }

    function togglePmr() {
        if (state.pmr.running) {
            state.pmr.running = false;
            if (state.pmr.timer) clearTimeout(state.pmr.timer);
            $('sa-toggle-pmr').textContent = 'Start';
            $('sa-pmr-status').textContent = 'idle';
            return;
        }
        state.pmr.running = true;
        $('sa-toggle-pmr').textContent = 'Stop';
        pollPmr();
    }

    // ------------------------------------------------------------------ export

    /**
     * Export the current scan as CSV.
     *
     * The whole point of an analysis tool is being able to take the data
     * elsewhere, so a trace you cannot get out is only half useful.
     */
    function exportSpectrum() {
        var spectrum = state.lastSpectrum;
        if (!spectrum || !spectrum.points || !spectrum.points.length) {
            toast('No scan to export yet', 'error');
            return;
        }
        var floor = spectrum.noiseFloorDbm;
        var rows = spectrum.points.map(function (point) {
            return {
                frequency_hz: Math.round(point.hz),
                frequency_mhz: Number((point.hz / 1e6).toFixed(6)),
                level_dbfs: Number(point.dbm).toFixed(2),
                above_floor_db: isFinite(floor) ? (point.dbm - floor).toFixed(2) : ''
            };
        });
        SAUI.exporter.csv(
            'spectrum_' + Math.round(spectrum.centerHz) + 'Hz_' + SAUI.format.fileStamp() + '.csv',
            rows
        );
    }

    /** Export the position fix, its residuals and the model used. */
    function exportFix() {
        var result = state.lastTriangulation;
        if (!result || !result.estimate) {
            toast('No fix to export yet', 'error');
            return;
        }
        var payload = {
            exported_at: new Date().toISOString(),
            frequency_hz: result.frequencyHz,
            estimate: result.estimate,
            model: result.model,
            readings: result.readings,
            warnings: (result.measurement && result.measurement.warnings) || [],
            receiver_history: state.estimateHistory
        };
        SAUI.exporter.json(
            'fix_' + Math.round(result.frequencyHz) + 'Hz_' + SAUI.format.fileStamp() + '.json',
            payload
        );
    }

    function exportLog() {
        if (!logHistory.length) {
            toast('Log is empty', 'warning');
            return;
        }
        SAUI.exporter.csv('spectrumalert_log_' + SAUI.format.fileStamp() + '.csv', logHistory);
    }

    function exportReceiverReadings() {
        var result = state.lastTriangulation;
        if (!result || !result.readings || !result.readings.length) {
            toast('No readings to export yet', 'error');
            return;
        }
        var rows = result.readings.map(function (reading) {
            return {
                receiver: reading.name,
                latitude: reading.lat,
                longitude: reading.lon,
                level_dbfs: reading.rssiDbm,
                noise_floor_dbfs: reading.noiseFloorDbm,
                prominence_db: reading.prominenceDb,
                calibration_offset_db: reading.calOffsetDb,
                modelled_distance_m: reading.modeledDistMeters,
                reliability: reading.reliability,
                source: reading.source
            };
        });
        SAUI.exporter.csv(
            'readings_' + Math.round(result.frequencyHz) + 'Hz_' + SAUI.format.fileStamp() + '.csv',
            rows
        );
    }

    // ------------------------------------------------------------- shortcuts


    // =========================================================================
    // Identify (RF fingerprinting)
    //
    // Shares the receiver roster, the waterfall and the target frequency with
    // Locate; only the analysis differs. Every response is rendered as it came
    // back, including "not in the library" — the panel never invents a device.
    // =========================================================================

    var STATUS_TEXT = {
        'identified': 'Identified',
        'no-match': 'Not in library',
        'ambiguous': 'Ambiguous',
        'indeterminate': 'Cannot compare',
        'unknown': 'No references'
    };

        function currentTargetHz() {
        var hz = Number($('sa-target-freq').value);
        if (isFinite(hz) && hz > 0) return hz;
        hz = Number($('sa-center').value);
        return isFinite(hz) && hz > 0 ? hz : null;
    }

    function selectedReceiverId() {
        var select = $('sa-scan-receiver');
        return select && select.value ? select.value : null;
    }

    function fpState(kind, title, detail) {
        var node = $('sa-fp-state');
        if (!node) return;
        node.hidden = !kind;
        node.className = 'sa-state sa-state--' + (kind || 'ready');
        if (!kind) return;
        var titleNode = node.querySelector('.sa-state-title');
        if (!titleNode) {
            titleNode = document.createElement('div');
            titleNode.className = 'sa-state-title';
            node.insertBefore(titleNode, node.firstChild);
        }
        titleNode.textContent = title || '';
        var detailNode = node.querySelector('.sa-state-detail');
        if (detailNode) detailNode.textContent = detail || '';
    }

    /** Difference bars, in units of each feature's own spread. */
    function renderFeatureTable(result) {
        var body = $('sa-fp-features');
        if (!body) return;
        var names = result.features || [];
        var values = result.values || {};
        var scales = result.featureScales || {};
        var match = result.match;

        var referenceValues = {};
        if (match && match.signature && match.signature.values) {
            referenceValues = match.signature.values;
        }

        if (!names.length) {
            body.innerHTML = '<tr><td colspan="5" class="sa-hint">No signature yet.</td></tr>';
            return;
        }

        body.innerHTML = names.map(function (name, index) {
            var value = Number(values[name]);
            var scale = Number(scales[name]);
            var reference = referenceValues[name];
            var cells;

            if (reference === undefined || !isFinite(Number(reference))) {
                cells = '<td class="sa-hint">--</td><td class="sa-hint">--</td>' +
                        '<td class="sa-hint">no reference</td>';
            } else {
                // Difference in spreads; anything under 1 is inside the noise
                // this feature actually shows.
                var difference = scale > 0 ? Math.abs(value - Number(reference)) / scale : 0;
                var band = difference < 1 ? 'low' : (difference < 3 ? 'mid' : 'high');
                var width = Math.min(100, Math.max(2, difference * 25));
                cells = '<td>' + Number(reference).toFixed(4) + '</td>' +
                        '<td><span class="sa-diff"><span class="sa-diff__fill" data-band="' +
                        band + '" style="width:' + width.toFixed(1) + '%"></span></span> ' +
                        '<span class="sa-hint">' + difference.toFixed(2) + '&sigma;</span></td>';
            }

            return '<tr><td>' + (index + 1) + '</td>' +
                   '<td>' + escapeHtml(name.replace(/_/g, ' ')) + '</td>' +
                   '<td>' + (isFinite(value) ? value.toFixed(4) : '--') + '</td>' +
                   cells + '</tr>';
        }).join('');

        var note = $('sa-fp-features-note');
        if (note) {
            note.textContent = match
                ? 'compared with ' + match.label
                : 'capture only — no reference compared yet';
        }
    }

    function renderVerdict(result) {
        var card = $('sa-fp-verdict-card');
        var wrap = $('sa-fp-verdict');
        if (!card || !wrap) return;

        wrap.hidden = false;
        var status = result.status || 'unknown';
        card.setAttribute('data-verdict', status);
        $('sa-fp-status').textContent = STATUS_TEXT[status] || status;

        var match = result.match;
        $('sa-fp-label').textContent = match ? match.label : '';

        var meta = [];
        if (match) {
            if (match.vendor || match.model) {
                meta.push((match.vendor || '?') + ' ' + (match.model || ''));
            }
            if (typeof result.confidence === 'number') {
                meta.push('confidence ' + Math.round(result.confidence * 100) + '%');
            }
            meta.push('distance ' + Number(result.bestDistance || 0).toFixed(3));
        } else if (typeof result.bestDistance === 'number') {
            meta.push('closest ' + Number(result.bestDistance).toFixed(3));
        }
        if (result.librarySize !== undefined) {
            meta.push(result.librarySize + ' reference' + (result.librarySize === 1 ? '' : 's'));
        }
        $('sa-fp-meta').textContent = meta.join(' · ');
        $('sa-fp-reason').textContent = result.reason || '';

        renderFeatureTable(result);

        // Saving a reference needs the capture that produced this verdict.
        var enrol = $('sa-fp-enrol');
        if (enrol) enrol.disabled = !result.values;
        var hint = $('sa-fp-enrol-hint');
        if (hint) {
            hint.textContent = result.values
                ? 'Saves this exact capture as a new reference.'
                : 'Capture a signature first.';
        }
    }

    function renderLibrary() {
        var body = $('sa-fp-library');
        if (!body) return;
        var entries = state.library || [];

        if (!entries.length) {
            body.innerHTML = '<tr><td colspan="6" class="sa-hint">' +
                'No references yet. Capture a known transmitter to add the first one; ' +
                'until then every result is reported as unknown.</td></tr>';
        } else {
            body.innerHTML = entries.map(function (entry) {
                var freq = entry.frequencyHz
                    ? escapeHtml(SAUI.format.hz(entry.frequencyHz, 4))
                    : '--';
                var vendor = [entry.vendor, entry.model].filter(Boolean).join(' ');
                var when = entry.createdAt ? SAUI.format.relativeTime(entry.createdAt) : '--';
                return '<tr>' +
                    '<td>' + escapeHtml(entry.label) + '</td>' +
                    '<td>' + (vendor ? escapeHtml(vendor) : '<span class="sa-hint">--</span>') + '</td>' +
                    '<td>' + freq + '</td>' +
                    '<td>' + (entry.samples || 0).toLocaleString() + '</td>' +
                    '<td>' + escapeHtml(when) + '</td>' +
                    '<td><button class="sa-icon-btn" data-fp-delete="' +
                        escapeHtml(entry.entryId) + '" title="Remove this reference" ' +
                        'aria-label="Remove reference ' + escapeHtml(entry.label) + '">' +
                        '<span aria-hidden="true">&times;</span></button></td>' +
                    '</tr>';
            }).join('');
        }

        var count = $('sa-fp-library-count');
        if (count) {
            count.textContent = entries.length
                ? entries.length + ' reference' + (entries.length === 1 ? '' : 's')
                : '';
        }
    }

    async function loadLibrary() {
        try {
            var data = await api('/api/fingerprints');
            state.library = data.entries || [];
            renderLibrary();
        } catch (error) {
            state.library = [];
            renderLibrary();
            log('Could not load the reference library: ' + error.message, 'warn');
        }
    }

    async function captureFingerprint(extra) {
        var frequency = currentTargetHz();
        if (!frequency) {
            fpState('error', 'No target frequency',
                'Set the target frequency before capturing a signature.');
            return null;
        }
        var receiverId = selectedReceiverId();
        if (!receiverId) {
            fpState('error', 'No receiver selected',
                'Choose a receiver with live hardware. A pinned level carries no IQ.');
            return null;
        }

        var captures = parseInt($('sa-fp-captures').value, 10);
        var button = $('sa-fp-capture');
        if (button) button.disabled = true;
        fpState('loading', 'Capturing',
            'Averaging ' + (isFinite(captures) ? captures : 4) + ' short captures…');
        setConnection('connecting');

        var payload = {
            sdrId: receiverId,
            frequencyHz: frequency,
            captures: isFinite(captures) ? captures : 4
        };
        if (extra) Object.keys(extra).forEach(function (key) {
            if (extra[key]) payload[key] = extra[key];
        });

        try {
            var result = await api('/api/fingerprints/capture', {
                method: 'POST', timeoutMs: 120000, body: payload
            });
            state.lastFingerprint = result;
            fpState(null);
            setConnection('live');
            renderVerdict(result);
            if (result.enrolled) {
                log('Saved reference "' + result.enrolled.label + '".', 'ok');
                toast('Reference "' + result.enrolled.label + '" saved', 'success');
                $('sa-fp-new-label').value = '';
                $('sa-fp-vendor').value = '';
                $('sa-fp-model').value = '';
                // The response describes the new entry but not the whole
                // library, so the table and the tab badge would keep showing the
                // pre-enrolment state until a page reload.
                loadLibrary();
            } else if (result.identified) {
                toast('Identified: ' + result.match.label, 'success');
            } else {
                toast(result.reason || 'No identification', 'warning');
            }
            renderLibrary();
            return result;
        } catch (error) {
            state.lastFingerprint = null;
            setConnection('down');
            fpState('error', 'Capture failed', error.message);
            var retry = document.createElement('button');
            retry.className = 'sa-retry';
            retry.textContent = 'Retry';
            retry.addEventListener('click', function () { captureFingerprint(); });
            var node = $('sa-fp-state');
            if (node) node.appendChild(retry);
            return null;
        } finally {
            if (button) button.disabled = false;
        }
    }

    function enrolCurrent() {
        var label = ($('sa-fp-new-label').value || '').trim();
        if (!label) {
            toast('Give the reference a label so it can be recognised later', 'warning');
            $('sa-fp-new-label').focus();
            return;
        }
        // Two references with the same label are indistinguishable by
        // construction, and every capture then comes back "ambiguous" between
        // them. Warn rather than block: extra captures of one radio can still
        // be wanted, but the operator should know why.
        var clash = state.library.some(function (entry) {
            return entry.label.toLowerCase() === label.toLowerCase();
        });
        if (clash) {
            toast('A reference labelled "' + label + '" already exists. Captures of ' +
                  'it will be reported as ambiguous.', 'warning');
        }
        captureFingerprint({
            label: label,
            vendor: ($('sa-fp-vendor').value || '').trim(),
            model: ($('sa-fp-model').value || '').trim()
        });
    }

    async function deleteReference(entryId) {
        try {
            await api('/api/fingerprints/' + encodeURIComponent(entryId), { method: 'DELETE' });
            log('Reference removed.', 'ok');
            await loadLibrary();
        } catch (error) {
            toast('Could not remove the reference: ' + error.message, 'error');
        }
    }

    function exportSignature() {
        if (!state.lastFingerprint) {
            toast('Capture a signature before exporting', 'warning');
            return;
        }
        SAUI.export.json('signature-' + Date.now() + '.json',
                         state.lastFingerprint);
        toast('Signature exported', 'success');
    }

    function wireFingerprint() {
        $('sa-fp-capture').addEventListener('click', function () { captureFingerprint(); });
        $('sa-fp-enrol').addEventListener('click', enrolCurrent);
        $('sa-fp-export').addEventListener('click', exportSignature);

        SAUI.delegate($('sa-fp-library'), '[data-fp-delete]', 'click', function (event, target) {
            deleteReference(target.getAttribute('data-fp-delete'));
        });

        renderLibrary();
    }

    /**
     * Workspace keyboard shortcuts.
     *
     * SAUI.shortcuts keeps one registry keyed by combo, so the second script to
     * register a key silently replaces the first. When this controller ran on
     * its own page that was invisible; merged into the dashboard it would have
     * quietly stolen or lost half of each set. Only keys the dashboard does not
     * already own are registered here, and the shared ones (theme, mute, help,
     * escape) are left to the dashboard.
     */
    /**
     * Adopt a target frequency handed over by the dashboard's alert list.
     *
     * Exposed on window because the alert Locate button lives in
     * dashboard-extras.js, which is a separate controller with no access to
     * anything defined in here.
     */
    function workspaceFocusTarget(hz) {
        if (!isFinite(hz)) return;
        $('sa-target-freq').value = String(Math.round(hz));
        state.pickedHz = hz;
        var workspace = document.getElementById('workspace');
        if (workspace) workspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
        log('Target ' + fmtHz(hz) + ' received from the alert list.');
    }
    window.workspaceFocusTarget = workspaceFocusTarget;

    function registerShortcuts() {
        var S = SAUI.shortcuts;

        // Scoping: a workspace key should not fire while the operator is typing.
        S.register('w', 'Scan the workspace receiver once', runScan, 'Workspace');
        S.register('l', 'Start / stop the live waterfall', function () {
            if (state.liveTimer) stopLive(); else startLive();
        }, 'Workspace');
        S.register('g', 'Triangulate the target frequency', triangulate, 'Workspace');
        S.register('f', 'Capture a signature and identify it', function () {
            captureFingerprint();
        }, 'Workspace');
        S.register('p', 'Start / stop the PMR446 scanner', togglePmr, 'Workspace');
        S.register('c', 'Recentre the map on the receivers', function () {
            renderMapReceivers();
            toast('Map recentred on receivers', 'info');
        }, 'Workspace');
        S.register('h', 'Toggle the likelihood heatmap', function () {
            var box = $('sa-show-heatmap');
            box.checked = !box.checked;
            if (state.lastTriangulation) renderEstimate(state.lastTriangulation);
        }, 'Workspace');
        S.register('b', 'Reload receivers', loadReceivers, 'Workspace');
        S.register('o', 'Export the latest fix', function () {
            if (state.lastTriangulation) exportFix(); else exportSpectrum();
        }, 'Export');
        S.register('u', 'Export receiver readings as CSV', exportReceiverReadings, 'Export');
        S.register('v', 'Export the current spectrum trace', exportSpectrum, 'Export');
        S.register('y', 'Export the last signature as JSON', exportSignature, 'Export');

        S.attach(document);

        // The dashboard owns the theme toggle; the map has to be told, because
        // Leaflet does not notice a palette change on its own.
        document.addEventListener('sa:themechange', function () { redrawMap(); });
    }

    /** The dashboard owns the help overlay; kept so workspace code can close it. */
    function toggleHelp(open) {
        var overlay = $('sa-help');
        if (!overlay) return;
        if (open) {
            SAUI.shortcuts.renderHelp($('sa-help-body'));
            overlay.classList.add('is-open');
            var close = $('help-close');
            if (close) close.focus();
        } else {
            overlay.classList.remove('is-open');
            $('help-open').focus();
        }
    }

    function toggleMute() {
        var prefs = SAUI.notify.getPrefs();
        var muted = !prefs.muted;
        SAUI.notify.setPref('muted', muted);
        var button = $('sa-notify-mute');
        if (button) {
            button.textContent = muted ? 'Muted' : 'Notifications';
            button.setAttribute('aria-pressed', String(muted));
        }
        toast(muted ? 'Notifications muted (criticals still shown)' : 'Notifications enabled', 'info');
    }

    // ------------------------------------------------------------------ init

    async function loadTools() {
        try {
            var tools = await api('/api/tools');
            $('sa-tools').textContent =
                (tools.rtlPowerAvailable ? 'rtl_power ✓' : 'rtl_power ✗') +
                ' · ' + tools.enabledCount + '/' + tools.receiverCount + ' receivers';
        } catch (_err) {
            $('sa-tools').textContent = 'backend unreachable';
            setConnection('down');
        }
    }

    /** Leaflet tiles and the heatmap need repainting when the palette changes. */
    function redrawMap() {
        if (!map) return;
        renderMapReceivers();
        if (state.lastTriangulation) renderEstimate(state.lastTriangulation);
    }

    function wire() {
        $('sa-mode').addEventListener('change', updateModeFields);
        $('sa-save-receiver').addEventListener('click', saveReceiver);
        $('sa-form-cancel').addEventListener('click', cancelEdit);

        $('sa-clear-receivers').addEventListener('click', async function () {
            if (!state.receivers.length) return;
            if (!confirm('Remove all ' + state.receivers.length +
                    ' receiver(s)? Triangulation, calibration and the waterfall all stop ' +
                    'working until you add them again.')) return;
            try {
                await api('/api/sdrs', { method: 'DELETE' });
                await loadReceivers();
                toast('All receivers removed', 'info');
            } catch (error) {
                toast('Clear failed: ' + error.message, 'error');
            }
        });

        $('sa-scan-once').addEventListener('click', runScan);
        $('sa-toggle-live').addEventListener('click', function () {
            if (state.liveTimer) stopLive();
            else startLive();
        });
        $('sa-triangulate').addEventListener('click', triangulate);

        $('sa-export-spectrum').addEventListener('click', exportSpectrum);
        $('sa-export-fix').addEventListener('click', exportFix);
        $('sa-export-readings').addEventListener('click', exportReceiverReadings);
        $('sa-export-log').addEventListener('click', exportLog);
        $('sa-clear-log').addEventListener('click', function () {
            $('sa-log').innerHTML = '';
            logHistory = [];
        });

        $('sa-req-add-receiver').addEventListener('click', function () {
            var field = $('sa-name');
            if (!field) return;
            field.scrollIntoView({ behavior: 'smooth', block: 'center' });
            field.focus({ preventScroll: true });
        });
        $('sa-scan-receiver').addEventListener('change', updateFingerprintRequirement);

        $('sa-preview-calibration').addEventListener('click', function () { runCalibration(false); });
        $('sa-apply-calibration').addEventListener('click', function () { runCalibration(true); });
        $('sa-fit-ple').addEventListener('click', fitPle);
        $('sa-clear-calibration').addEventListener('click', async function () {
            if (!confirm('Clear every calibration offset? Readings will go back to being ' +
                    'uncorrected, which can move the reported position.')) return;
            try {
                renderCalibration(await api('/api/calibration', { method: 'DELETE' }));
                toast('All offsets cleared', 'info');
            } catch (error) {
                toast('Clear failed: ' + error.message, 'error');
            }
        });

        $('sa-toggle-pmr').addEventListener('click', togglePmr);
        // Theme, mute and help belong to the dashboard header now that the two
        // UIs are one; wiring them twice would toggle twice per click. The map
        // still re-tiles on change, via the sa:themechange listener below.
        var muteButton = $('notify-mute');
        if (muteButton) muteButton.addEventListener('click', toggleMute);
        var helpOpen = $('help-open');
        if (helpOpen) helpOpen.addEventListener('click', function () { toggleHelp(true); });
        var helpClose = $('help-close');
        if (helpClose) helpClose.addEventListener('click', function () { toggleHelp(false); });
        $('sa-modal-cancel').addEventListener('click', function () {
            if (modalState.onCancel) modalState.onCancel();
        });

        ['sa-show-ellipse', 'sa-show-heatmap', 'sa-show-history'].forEach(function (id) {
            $(id).addEventListener('change', function () {
                if (!state.lastTriangulation) return;
                renderEstimate(state.lastTriangulation);
            });
        });

        // Keep the freshness label honest between scans.
        window.setInterval(function () {
            if (!state.lastScanAt) return;
            var age = SAUI.format.relativeTime(state.lastScanAt);
            var node = $('sa-freshness');
            if (node) node.textContent = 'updated ' + age;
        }, 5000);

        window.addEventListener('beforeunload', function () {
            stopLive();
            if (state.pmr.running) togglePmr();
        });
    }

    /**
     * Adopt a target frequency handed over by the dashboard.
     *
     * The alert Locate button opens /geolocate?freq=<hz>, but the page never
     * read that parameter, so it always loaded the default band and the
     * operator had to retune by hand. The value is also stashed in localStorage
     * as a fallback for the no-parameter case, and consumed once so a later
     * manual retune is not overwritten.
     */
    function applyTargetFromUrl() {
        var hz = null;
        try {
            var params = new URLSearchParams(window.location.search);
            var raw = params.get('freq');
            if (raw !== null && raw.trim() !== '') {
                // Accept Hz, or MHz/MHz-suffixed values for convenience.
                var value = parseFloat(raw);
                if (isFinite(value)) hz = /\s*(m|mhz)$/i.test(raw) ? value * 1e6 : value;
            }
        } catch (_err) { /* URLSearchParams unavailable */ }

        // typeof, not isFinite: a missing key yields null, isFinite(null) is
        // true (null coerces to 0), and the centre used to be set to exactly
        // 0 Hz, which then failed every scan with an unreadable 422.
        if (typeof hz !== 'number' || !isFinite(hz)) {
            hz = SAUI.store.get('geoTargetHz', null);
        }
        if (typeof hz !== 'number' || !isFinite(hz)) return false;

        SAUI.store.set('geoTargetHz', null);
        $('sa-center').value = String(Math.round(hz));
        state.pickedHz = hz;
        log('Target ' + SAUI.format.hz(hz, 4) + ' received from dashboard.');
        return true;
    }

    document.addEventListener('DOMContentLoaded', async function () {
        state.dbFloorInput = $('sa-db-floor');
        if (state.dbFloorInput && isFinite(Number(state.dbFloorInput.value))) {
            displayFloor = Number(state.dbFloorInput.value);
            displayCeiling = displayFloor + 80;
        }
        state.conn = SAUI.connectionPill($('sa-conn'));

        initWaterfall();
        // Until the first scan resolves, the waterfall has nothing to draw. Show
        // a loading state instead of the blank black canvas the panel would
        // otherwise display for the first couple of seconds.
        setScanState('loading', 'Preparing workspace',
            'Loading the receiver roster and checking the SDR…');
        // The crosshair lives inside the waterfall shell so it tracks the canvas.
        cursorLine = document.createElement('div');
        cursorLine.className = 'sa-crosshair';
        var shell = $('sa-waterfall-shell');
        if (shell) shell.appendChild(cursorLine);

        initMap();
        initPmrGrid();
        wire();
        registerShortcuts();
        wireFingerprint();
        updateModeFields();

        ['sa-notify-mute', 'notify-mute'].forEach(function (id) {
            var button = $(id);
            if (!button) return;
            var prefs = SAUI.notify.getPrefs();
            if (id === 'sa-notify-mute') {
                button.textContent = prefs.muted ? 'Muted' : 'Notifications';
            }
            button.setAttribute('aria-pressed', String(!!prefs.muted));
        });

        applyTargetFromUrl();

        await loadReceivers();
        loadTools();
        loadLibrary();
        try {
            renderCalibration(await api('/api/calibration'));
        } catch (_err) { /* no calibration yet */ }

        log('Ready. ' + state.receivers.length + ' receiver(s) loaded.');
        if (!state.receivers.length) {
            setScanState('empty', 'No receivers configured',
                'Add a receiver in the left panel to start scanning.');
        } else {
            runScan();
        }
    });
})();