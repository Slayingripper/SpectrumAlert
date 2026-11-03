/* ============================================================================
   SpectrumAlert shared UI primitives
   ----------------------------------------------------------------------------
   Exposes a single global, window.SAUI. Loaded before dashboard.js so the
   dashboard can use these helpers, and standalone on the geolocation page.

   Why this exists: the dashboard had four separate notification paths
   (dashboard.js, websocket.js, enhanced-features.js, plus native alert() and
   confirm()), three independent WebSocket connections, and no shared notion of
   what a loading, empty or failed state looks like. Centralising those fixes
   the inconsistency and means a fix lands in one place.
   ========================================================================== */

(function (window, document) {
    'use strict';

    // ------------------------------------------------------------ format ---

    var KB = 1024, MB = KB * 1024, GB = MB * 1024, TB = GB * 1024;

    var format = {
        /**
         * Level in dBFS. Always suffixed dBFS: an RTL-SDR has no calibrated
         * reference, so calling these values dBm would be a lie. Shown to one
         * decimal because that is the resolution of the underlying estimate.
         */
        dbfs: function (value, digits) {
            if (value === null || value === undefined || !isFinite(value)) return '--';
            return Number(value).toFixed(digits === undefined ? 1 : digits) + ' dBFS';
        },

        /** Frequency with unit chosen by magnitude. */
        hz: function (value, digits) {
            if (value === null || value === undefined || !isFinite(value)) return '--';
            var hz = Number(value);
            var abs = Math.abs(hz);
            if (abs >= 1e9) return (hz / 1e9).toFixed(digits === undefined ? 6 : digits) + ' GHz';
            if (abs >= 1e6) return (hz / 1e6).toFixed(digits === undefined ? 6 : digits) + ' MHz';
            if (abs >= 1e3) return (hz / 1e3).toFixed(digits === undefined ? 3 : digits) + ' kHz';
            return hz.toFixed(digits === undefined ? 1 : digits) + ' Hz';
        },

        /** Megahertz, for places already in MHz. */
        mhz: function (value, digits) {
            if (value === null || value === undefined || !isFinite(value)) return '--';
            return Number(value).toFixed(digits === undefined ? 6 : digits) + ' MHz';
        },

        /** Sample rate in MS/s, which is how SDR tooling states it. */
        msps: function (value) {
            if (!isFinite(value)) return '--';
            return (Number(value) / 1e6).toFixed(3) + ' MS/s';
        },

        bytes: function (value) {
            if (value === null || value === undefined || !isFinite(value)) return '--';
            var bytes = Number(value);
            if (bytes < KB) return bytes + ' B';
            if (bytes < MB) return (bytes / KB).toFixed(1) + ' KB';
            if (bytes < GB) return (bytes / MB).toFixed(1) + ' MB';
            if (bytes < TB) return (bytes / GB).toFixed(2) + ' GB';
            return (bytes / TB).toFixed(2) + ' TB';
        },

        count: function (value) {
            if (value === null || value === undefined || !isFinite(value)) return '--';
            return Number(value).toLocaleString();
        },

        percent: function (value, digits) {
            if (!isFinite(value)) return '--';
            return Number(value).toFixed(digits === undefined ? 1 : digits) + '%';
        },

        /**
         * Elapsed time as H:MM:SS, or MM:SS under an hour.
         */
        duration: function (seconds) {
            if (!isFinite(seconds) || seconds < 0) return '--';
            var total = Math.floor(seconds);
            var h = Math.floor(total / 3600);
            var m = Math.floor((total % 3600) / 60);
            var s = total % 60;
            var pad = function (n) { return String(n).padStart(2, '0'); };
            return h > 0 ? h + ':' + pad(m) + ':' + pad(s) : pad(m) + ':' + pad(s);
        },

        /**
         * Human relative time ("3 min ago"). Preferred over absolute timestamps
         * for freshness, where the point is how old the number is.
         */
        relativeTime: function (timestamp) {
            var then = timestamp instanceof Date ? timestamp : new Date(timestamp);
            if (isNaN(then.getTime())) return 'unknown';
            var seconds = Math.round((Date.now() - then.getTime()) / 1000);
            if (seconds < 0) return 'just now';
            if (seconds < 5) return 'just now';
            if (seconds < 60) return seconds + 's ago';
            var minutes = Math.round(seconds / 60);
            if (minutes < 60) return minutes + ' min ago';
            var hours = Math.round(minutes / 60);
            if (hours < 24) return hours + 'h ago';
            var days = Math.round(hours / 24);
            if (days < 30) return days + 'd ago';
            return then.toLocaleDateString();
        },

        /** Absolute timestamp with timezone, for when the exact time matters. */
        timestamp: function (timestamp) {
            var when = timestamp instanceof Date ? timestamp : new Date(timestamp);
            if (isNaN(when.getTime())) return '--';
            return when.toLocaleString(undefined, {
                year: 'numeric', month: 'short', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
        },

        /** Filename-safe UTC stamp for exported files. */
        fileStamp: function (date) {
            var when = date || new Date();
            return when.toISOString().replace(/[:.]/g, '-').slice(0, 19);
        }
    };

    // -------------------------------------------------------- DOM helpers ---

    function el(id) { return document.getElementById(id); }

    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function setText(id, value) {
        var node = typeof id === 'string' ? el(id) : id;
        if (node) node.textContent = value;
    }

    function setHtml(id, value) {
        var node = typeof id === 'string' ? el(id) : id;
        if (node) node.innerHTML = value;
    }

    function show(node, visible) {
        if (!node) return;
        node.hidden = !visible;
        node.style.display = visible ? '' : 'none';
    }

    function on(node, event, handler, options) {
        if (node) node.addEventListener(event, handler, options);
    }

    function delegate(container, selector, event, handler) {
        if (!container) return;
        container.addEventListener(event, function (e) {
            var target = e.target.closest ? e.target.closest(selector) : null;
            if (target && container.contains(target)) handler(e, target);
        });
    }

    // ------------------------------------------------------------- store ---

    /** Namespaced localStorage that degrades to memory when unavailable. */
    var store = (function () {
        var PREFIX = 'spectrumAlert.';
        var memory = {};
        var available = (function () {
            try {
                var probe = '__sa_probe__';
                window.localStorage.setItem(probe, '1');
                window.localStorage.removeItem(probe);
                return true;
            } catch (err) {
                return false;
            }
        })();

        function read(key) {
            if (!available) return memory[key];
            try { return window.localStorage.getItem(PREFIX + key); } catch (err) { return null; }
        }

        function write(key, value) {
            if (!available) { memory[key] = value; return; }
            try { window.localStorage.setItem(PREFIX + key, value); } catch (err) { /* quota */ }
        }

        return {
            get: function (key, fallback) {
                var raw = read(key);
                if (raw === null || raw === undefined) return fallback;
                try { return JSON.parse(raw); } catch (err) { return fallback; }
            },
            set: function (key, value) {
                try { write(key, JSON.stringify(value)); } catch (err) { /* ignore */ }
            },
            remove: function (key) {
                if (!available) { delete memory[key]; return; }
                try { window.localStorage.removeItem(PREFIX + key); } catch (err) { /* ignore */ }
            },
            available: available
        };
    })();

    // ------------------------------------------------------- notifications ---

    var SEVERITY_RANK = { info: 0, success: 1, warning: 2, error: 3, critical: 4 };

    var SEVERITY_ICON = {
        info: 'i', success: 'OK', warning: '!', error: 'x', critical: '!!'
    };

    var notifications = {
        host: null,
        prefs: store.get('notifyPrefs', {
            muted: false,
            minSeverity: 'info',
            historyLimit: 100
        }),
        history: store.get('notifyHistory', []),
        active: [],
        timers: new WeakMap(),

        _ensureHost: function () {
            if (this.host && document.body.contains(this.host)) return this.host;
            this.host = document.createElement('div');
            this.host.className = 'sa-toast-host';
            // Announced politely so a screen reader hears notifications without
            // interrupting whatever the user is doing.
            this.host.setAttribute('role', 'region');
            this.host.setAttribute('aria-label', 'Notifications');
            document.body.appendChild(this.host);
            return this.host;
        },

        /**
         * @param {string} message
         * @param {object} [options]
         * @param {'info'|'success'|'warning'|'error'|'critical'} [options.severity]
         * @param {number} [options.timeout] ms; 0 keeps it until dismissed
         * @param {string} [options.title]
         * @param {Array}  [options.actions] [{label, onClick}]
         */
        show: function (message, options) {
            options = options || {};
            var self = this;
            var severity = options.severity || 'info';

            if (this.prefs.muted && severity !== 'critical') return null;
            var minRank = SEVERITY_RANK[this.prefs.minSeverity] || 0;
            if ((SEVERITY_RANK[severity] || 0) < minRank) return null;

            var record = { message: message, severity: severity, title: options.title || null, at: Date.now() };
            this.history.unshift(record);
            if (this.history.length > this.prefs.historyLimit) this.history.length = this.prefs.historyLimit;
            store.set('notifyHistory', this.history);

            var host = this._ensureHost();

            // Stacking cap: a burst of alerts should not push content off
            // screen. Collapsing the overflow into a count keeps the newest
            // visible without dropping information silently.
            var MAX_VISIBLE = 4;
            var toast = document.createElement('div');
            toast.className = 'sa-toast sa-toast--' + severity;
            if (options.timeout === 0 || severity === 'critical' || severity === 'error') {
                toast.style.position = 'relative';
            }
            toast.setAttribute('role', severity === 'critical' || severity === 'error' ? 'alert' : 'status');

            var icon = document.createElement('span');
            icon.className = 'sa-toast__icon severity-glyph';
            icon.setAttribute('aria-hidden', 'true');
            icon.textContent = SEVERITY_ICON[severity] || 'i';

            var body = document.createElement('div');
            body.className = 'sa-toast__body';
            if (options.title) {
                var title = document.createElement('div');
                title.className = 'sa-toast__title';
                title.textContent = options.title;
                body.appendChild(title);
            }
            var text = document.createElement('div');
            text.className = 'sa-toast__message';
            text.textContent = message;
            body.appendChild(text);

            if (options.actions && options.actions.length) {
                var actions = document.createElement('div');
                actions.className = 'sa-toolbar';
                actions.style.marginTop = '6px';
                options.actions.forEach(function (action) {
                    var button = document.createElement('button');
                    button.className = 'sa-icon-btn';
                    button.style.width = 'auto';
                    button.style.height = 'auto';
                    button.style.padding = '2px 8px';
                    button.textContent = action.label;
                    on(button, 'click', function () {
                        dismiss(toast);
                        try { action.onClick(); } catch (err) { console.error(err); }
                    });
                    actions.appendChild(button);
                });
                body.appendChild(actions);
            }

            var close = document.createElement('button');
            close.className = 'sa-toast__close';
            close.setAttribute('aria-label', 'Dismiss notification');
            close.textContent = '\u00d7';
            on(close, 'click', function () { dismiss(toast); });

            toast.appendChild(icon);
            toast.appendChild(body);
            toast.appendChild(close);
            host.appendChild(toast);
            this.active.push(toast);

            while (this.active.length > MAX_VISIBLE) {
                var overflow = this.active.shift();
                dismiss(overflow);
            }

            // Errors and criticals stay until dismissed: they are the ones a
            // user must actually read, and a flat 5 s is not enough time.
            var timeout = options.timeout !== undefined
                ? options.timeout
                : (severity === 'critical' || severity === 'error' ? 0 : 4500);

            if (timeout > 0) {
                var timer = window.setTimeout(function () { dismiss(toast); }, timeout);
                self.timers.set(toast, timer);
                // Pause on hover so a user reading a message is not interrupted.
                on(toast, 'mouseenter', function () {
                    window.clearTimeout(timer);
                    self.timers.delete(toast);
                });
                on(toast, 'mouseleave', function () {
                    if (document.body.contains(toast)) {
                        self.timers.set(toast, window.setTimeout(function () { dismiss(toast); }, 1500));
                    }
                });
            }

            function dismiss(node) {
                if (!node || !document.body.contains(node)) return;
                var pending = self.timers.get(node);
                if (pending) { window.clearTimeout(pending); self.timers.delete(node); }
                node.classList.add('sa-toast--leaving');
                var index = self.active.indexOf(node);
                if (index >= 0) self.active.splice(index, 1);
                window.setTimeout(function () {
                    if (node.parentNode) node.parentNode.removeChild(node);
                }, 160);
            }

            return toast;
        },

        dismissAll: function () {
            var self = this;
            this.active.slice().forEach(function (node) {
                self.timers.delete(node);
                if (node.parentNode) node.parentNode.removeChild(node);
            });
            this.active = [];
        },

        setPref: function (key, value) {
            this.prefs[key] = value;
            store.set('notifyPrefs', this.prefs);
        },

        getPrefs: function () { return Object.assign({}, this.prefs); },

        getHistory: function () { return this.history.slice(); },

        clearHistory: function () {
            this.history = [];
            store.set('notifyHistory', this.history);
        }
    };

    // ------------------------------------------------------ state rendering --

    /**
     * Render a loading / empty / error / stale state into a container.
     *
     * The dashboard previously showed "--" forever on any failure, so a dead
     * backend and a genuinely empty result looked identical. These three states
     * are distinct on purpose and each says what happened.
     */
    function renderState(container, state, options) {
        if (!container) return;
        options = options || {};

        // Remove any previous state node, but leave the real content alone so it
        // can be shown again without a refetch.
        var existing = container.querySelector(':scope > .sa-state');
        if (existing) existing.remove();

        var content = container.querySelector(':scope > .sa-content');
        if (content) show(content, state === 'ready');

        if (state === 'ready') {
            container.removeAttribute('data-sa-state');
            return;
        }

        container.setAttribute('data-sa-state', state);

        var node = document.createElement('div');
        node.className = 'sa-state sa-state--' + state;

        if (state === 'loading') {
            var spinner = document.createElement('div');
            spinner.className = 'sa-spinner';
            spinner.setAttribute('aria-hidden', 'true');
            node.appendChild(spinner);
        }

        var title = document.createElement('div');
        title.className = 'sa-state-title';
        title.textContent = options.title || ({
            loading: 'Loading',
            empty: 'No data',
            error: 'Unavailable',
            stale: 'Showing last known data'
        })[state] || state;
        node.appendChild(title);

        if (options.message) {
            var detail = document.createElement('div');
            detail.className = 'sa-state-detail';
            detail.textContent = options.message;
            node.appendChild(detail);
        }

        if (state === 'error' && options.onRetry) {
            var retry = document.createElement('button');
            retry.className = 'sa-retry';
            retry.textContent = 'Retry';
            on(retry, 'click', function () {
                try { options.onRetry(); } catch (err) { console.error(err); }
            });
            node.appendChild(retry);
        }

        container.appendChild(node);
    }

    /**
     * Wrap a fetch so a failure produces a visible state rather than a silent
     * console message, and so a slow response can be abandoned.
     */
    async function request(url, options) {
        options = options || {};
        var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
        var timeoutMs = options.timeoutMs || 15000;
        var timer = controller ? window.setTimeout(function () { controller.abort(); }, timeoutMs) : null;

        var init = Object.assign({}, options, { signal: controller ? controller.signal : undefined });
        delete init.timeoutMs;
        if (options.body && typeof options.body !== 'string') {
            init.headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
            init.body = JSON.stringify(options.body);
        }

        try {
            var response = await fetch(url, init);
            var payload = null;
            var text = await response.text();
            if (text) {
                try { payload = JSON.parse(text); } catch (err) { payload = { raw: text }; }
            }
            if (!response.ok) {
                var detail = (payload && (payload.detail || payload.error || payload.message))
                    || ('HTTP ' + response.status);
                // FastAPI validation failures return detail as an ARRAY of
                // {loc, msg} objects. new Error(array) stringifies to
                // "[object Object]", which is what the scan panel used to show.
                if (Array.isArray(detail)) {
                    detail = detail.map(function (entry) {
                        var where = Array.isArray(entry.loc)
                            ? entry.loc.filter(function (p) { return p !== 'body'; }).join('.')
                            : '';
                        return (where ? where + ': ' : '') + (entry.msg || String(entry));
                    }).join('; ');
                } else if (detail && typeof detail === 'object') {
                    detail = detail.msg || JSON.stringify(detail);
                }
                var wrapped = new Error(String(detail));
                wrapped.status = response.status;
                wrapped.payload = payload;
                throw wrapped;
            }
            return payload;
        } catch (err) {
            if (err.name === 'AbortError') {
                var timeoutError = new Error('Request timed out after ' + Math.round(timeoutMs / 1000) + 's');
                timeoutError.timedOut = true;
                throw timeoutError;
            }
            throw err;
        } finally {
            if (timer) window.clearTimeout(timer);
        }
    }

    // ------------------------------------------------------------- export ---

    function download(filename, content, mimeType) {
        var blob = content instanceof Blob ? content : new Blob([content], { type: mimeType || 'text/plain;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;
        anchor.style.display = 'none';
        document.body.appendChild(anchor);
        anchor.click();
        window.setTimeout(function () {
            if (anchor.parentNode) anchor.parentNode.removeChild(anchor);
            URL.revokeObjectURL(url);
        }, 100);
    }

    /** RFC 4180 CSV: quote fields containing a delimiter, quote or newline. */
    function toCsv(rows, columns) {
        if (!rows || !rows.length) return '';
        var keys = columns || Object.keys(rows[0]);
        function cell(value) {
            if (value === null || value === undefined) return '';
            var text = typeof value === 'object' ? JSON.stringify(value) : String(value);
            return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
        }
        var lines = [keys.join(',')];
        rows.forEach(function (row) {
            lines.push(keys.map(function (key) {
                return cell(row[key]);
            }).join(','));
        });
        return lines.join('\r\n');
    }

    var exporter = {
        csv: function (filename, rows, columns) {
            if (!rows || !rows.length) {
                notifications.show('Nothing to export', { severity: 'warning' });
                return false;
            }
            download(filename.replace(/\.csv$/, '') + '.csv', toCsv(rows, columns), 'text/csv;charset=utf-8');
            notifications.show('Exported ' + rows.length + ' row(s) to ' + filename, { severity: 'success' });
            return true;
        },
        json: function (filename, payload) {
            download(filename.replace(/\.json$/, '') + '.json',
                JSON.stringify(payload, null, 2), 'application/json');
            notifications.show('Exported to ' + filename, { severity: 'success' });
            return true;
        },
        download: download,
        toCsv: toCsv
    };

    // ------------------------------------------------------------- theme ----

    var theme = {
        init: function () {
            var saved = store.get('theme', null);
            var preferred = saved || (
                window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
                    ? 'light' : 'dark'
            );
            this.apply(preferred, false);
        },
        apply: function (name, persist) {
            document.documentElement.setAttribute('data-theme', name);
            var meta = document.querySelector('meta[name="theme-color"]');
            if (!meta) {
                meta = document.createElement('meta');
                meta.setAttribute('name', 'theme-color');
                document.head.appendChild(meta);
            }
            meta.setAttribute('content', name === 'light' ? '#f3f5f9' : '#0b0f15');
            if (persist !== false) store.set('theme', name);
            this.current = name;
            document.dispatchEvent(new CustomEvent('sa:themechange', { detail: { theme: name } }));
            return name;
        },
        toggle: function () {
            return this.apply(this.current === 'light' ? 'dark' : 'light');
        },
        current: 'dark'
    };

    // -------------------------------------------------- keyboard shortcuts --

    // Nodes that already carry the dispatcher. Both the dashboard and the
    // workspace call attach(document), and a second listener would run every
    // registered handler twice — a start/stop toggle would fire both halves
    // of the pair in one keypress.
    var attachedNodes = [];

    var shortcuts = {
        registry: {},

        register: function (combo, description, handler, group) {
            this.registry[combo.toLowerCase()] = {
                description: description, handler: handler, group: group || 'General'
            };
        },

        /**
         * Build the combo key from an event. Modifier order is fixed so
         * "ctrl+shift+k" always produces the same string regardless of the
         * order the browser reports the flags in.
         */
        comboFromEvent: function (event) {
            var parts = [];
            if (event.ctrlKey || event.metaKey) parts.push('mod');
            if (event.altKey) parts.push('alt');
            if (event.shiftKey) parts.push('shift');
            var key = event.key;
            if (key === ' ') key = 'space';
            parts.push(String(key).toLowerCase());
            return parts.join('+');
        },

        attach: function (target) {
            var node = target || document;
            if (attachedNodes.indexOf(node) !== -1) return;
            attachedNodes.push(node);
            on(node, 'keydown', function (event) {
                var combo = shortcuts.comboFromEvent(event);
                var entry = shortcuts.registry[combo];
                if (!entry) return;

                // Never steal keys from a text field, except the global ones.
                var tag = (event.target && event.target.tagName || '').toLowerCase();
                var typing = tag === 'input' || tag === 'textarea' || tag === 'select'
                    || (event.target && event.target.isContentEditable);
                var isGlobal = combo === '?' || combo === 'mod+/' || combo === 'escape';
                if (typing && !isGlobal) return;

                event.preventDefault();
                try { entry.handler(event); } catch (err) { console.error(err); }
            });
        },

        renderHelp: function (host) {
            if (!host) return;
            var groups = {};
            Object.keys(this.registry).forEach(function (combo) {
                var entry = shortcuts.registry[combo];
                if (!groups[entry.group]) groups[entry.group] = [];
                groups[entry.group].push({ combo: combo, description: entry.description });
            });

            var html = '';
            Object.keys(groups).sort().forEach(function (group) {
                html += '<div class="sa-help__group"><h3>' + escapeHtml(group) + '</h3>';
                groups[group].forEach(function (item) {
                    var pretty = item.combo
                        .replace('mod', 'Ctrl')
                        .replace('shift', 'Shift')
                        .replace('alt', 'Alt')
                        .replace('space', 'Space')
                        .split('+')
                        .map(function (part) { return '<kbd>' + escapeHtml(part) + '</kbd>'; })
                        .join(' + ');
                    html += '<div class="sa-help__row"><span class="sa-help__desc">'
                        + escapeHtml(item.description) + '</span><span>' + pretty + '</span></div>';
                });
                html += '</div>';
            });
            host.innerHTML = html;
        }
    };

    // ----------------------------------------------------- connection pill --

    /**
     * Connection state that tells the truth. The dashboard footer previously
     * hardcoded "CONNECTED" and nothing ever updated it, so it claimed a live
     * connection while the socket was down.
     */
    function connectionPill(node) {
        var target = typeof node === 'string' ? el(node) : node;
        if (!target) return { set: function () {} };

        // Idempotent: the dashboard and the workspace both claim the one pill
        // in the shared header, and a second construction used to append a
        // second dot and label, so the pill read "CONNECTINGLIVE" and carried
        // two dots at once.
        if (target.__saConn) return target.__saConn;

        target.textContent = '';
        var dot = document.createElement('span');
        dot.className = 'sa-conn__dot';
        var label = document.createElement('span');
        target.appendChild(dot);
        target.appendChild(label);
        target.className = 'sa-conn';

        var state = 'connecting';
        var lastChanged = Date.now();

        function render() {
            target.classList.remove('sa-conn--live', 'sa-conn--connecting', 'sa-conn--down');
            target.classList.add('sa-conn--' + state);
            var texts = {
                live: 'LIVE',
                connecting: 'CONNECTING',
                down: 'OFFLINE'
            };
            label.textContent = texts[state] || state;
            target.setAttribute('role', 'status');
            target.setAttribute('aria-label', 'Data connection: ' + label.textContent);
            target.title = 'State changed ' + format.relativeTime(new Date(lastChanged));
        }

        render();

        var pill = {
            set: function (next) {
                if (next === state) return;
                state = next;
                lastChanged = Date.now();
                render();
            },
            get: function () { return state; },
            refreshTimestamp: render
        };
        target.__saConn = pill;
        return pill;
    }

    // ------------------------------------------------- crosshair readout ----

    /**
     * Hover readout for a Chart.js canvas: exact frequency and level under the
     * pointer. This is the single most expected interaction in a spectrum tool
     * and it was missing entirely.
     */
    function crosshair(canvas, options) {
        options = options || {};
        if (!canvas || !window.Chart) return null;

        var shell = canvas.parentElement;
        if (!shell) return null;
        shell.classList.add('sa-chart-shell');

        var line = document.createElement('div');
        line.className = 'sa-crosshair';
        var readout = document.createElement('div');
        readout.className = 'sa-readout';
        readout.setAttribute('aria-hidden', 'true');
        shell.appendChild(line);
        shell.appendChild(readout);

        function hide() {
            line.classList.remove('is-visible');
            readout.classList.remove('is-visible');
        }

        on(canvas, 'mousemove', function (event) {
            if (!window.Chart) return;
            var chart = window.Chart.getChart
                ? window.Chart.getChart(canvas)
                : window.Chart.instances ? null : null;
            // getChart returns null when the chart was destroyed, which happens
            // on reload and whenever a theme or data change rebuilds it.
            if (!chart) { hide(); return; }
            var meta = chart.getDatasetMeta(0);
            if (!meta || !meta.data || !meta.data.length) return;

            var rect = canvas.getBoundingClientRect();
            var x = event.clientX - rect.left;

            var points = meta.data;
            var nearest = points[0], best = Infinity;
            for (var i = 0; i < points.length; i++) {
                var distance = Math.abs(points[i].x - x);
                if (distance < best) { best = distance; nearest = points[i]; }
            }

            var index = points.indexOf(nearest);
            var labels = chart.data.labels || [];
            var value = chart.data.datasets[0].data[index];
            var label = labels[index];

            line.style.left = nearest.x + 'px';
            line.classList.add('is-visible');

            var frequency = typeof label === 'number'
                ? options.formatX ? options.formatX(label) : label
                : label;
            var level = options.formatY ? options.formatY(value) : value;

            readout.innerHTML =
                '<div><span class="sa-readout__label">' + escapeHtml(options.xLabel || 'x') + '</span> ' +
                '<span class="sa-readout__freq">' + escapeHtml(frequency) + '</span></div>' +
                '<div><span class="sa-readout__label">' + escapeHtml(options.yLabel || 'y') + '</span> ' +
                '<span class="sa-readout__db">' + escapeHtml(level) + '</span></div>';
            readout.classList.add('is-visible');

            // Keep the readout inside the panel when hovering near the right edge.
            var readoutWidth = readout.offsetWidth;
            var maxLeft = rect.width - readoutWidth - 8;
            readout.style.left = Math.max(0, Math.min(maxLeft, nearest.x - readoutWidth / 2)) + 'px';
        });

        on(canvas, 'mouseleave', hide);

        return { hide: hide, element: readout };
    }

    // ------------------------------------------------------------- export ---

    var SAUI = {
        format: format,
        el: el,
        escapeHtml: escapeHtml,
        setText: setText,
        setHtml: setHtml,
        show: show,
        on: on,
        delegate: delegate,
        store: store,
        notify: notifications,
        renderState: renderState,
        request: request,
        exporter: exporter,
        theme: theme,
        shortcuts: shortcuts,
        connectionPill: connectionPill,
        crosshair: crosshair,
        severityRank: SEVERITY_RANK
    };

    window.SAUI = SAUI;

    // Boot theme immediately to avoid a flash of the wrong palette.
    theme.init();
})(window, document);