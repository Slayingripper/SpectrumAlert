/*
 * Movable, resizable panels.
 *
 * The dashboard's tab views are CSS grids; each panel occupies a span of the
 * twelve columns. This module lets an operator rearrange panels (drag the grip
 * in the header) and resize them (drag the corner), so a panel that is too
 * short to show all of its content can simply be made bigger. Layout is saved
 * per section in localStorage and restored on the next visit.
 *
 * Scope: direct `.panel` children of a `.tab-view` or of a `.sa-module-view`
 * (the dashboard groups its panels into module tabs). The Bootstrap-grid
 * workspace keeps its own responsive layout.
 */
(function (window, document) {
    'use strict';

    var STORE_KEY = 'saPanelLayoutV1';
    var MIN_SPAN = 2;
    var MAX_SPAN = 12;
    var MIN_HEIGHT = 160;

    function load() {
        try {
            return JSON.parse(window.localStorage.getItem(STORE_KEY)) || {};
        } catch (error) {
            return {};
        }
    }

    function save(data) {
        try {
            window.localStorage.setItem(STORE_KEY, JSON.stringify(data));
        } catch (error) {
            /* private mode / quota: layout just will not persist */
        }
    }

    /** A stable key for a panel across reloads. */
    function panelKey(panel) {
        if (!panel.dataset.panelKey) {
            var cls = String(panel.className || '').split(/\s+/).filter(function (c) {
                return c && c !== 'panel' && c !== 'sa-panel' && c !== 'collapsible';
            })[0];
            var heading = panel.querySelector('h2');
            panel.dataset.panelKey = panel.id || cls ||
                (heading ? 'h:' + heading.textContent.trim().slice(0, 24) : 'panel');
        }
        return panel.dataset.panelKey;
    }

    function isPanel(el) {
        return !!(el && el.classList && el.classList.contains('panel'));
    }

    function panelChildren(container) {
        return Array.prototype.filter.call(container.children, function (child) {
            // The guide strip is a `.sa-guide`, not a `.panel`, so it is left
            // in place at the top of the dashboard.
            return isPanel(child);
        });
    }

    function gridMetrics(container) {
        var cs = window.getComputedStyle(container);
        var gap = parseFloat(cs.columnGap || cs.gap) || 0;
        var tracks = (cs.gridTemplateColumns || '').split(' ').filter(Boolean).length || 12;
        var padding = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
        var content = Math.max(0, container.clientWidth - padding);
        var colWidth = (content - gap * (tracks - 1)) / tracks;
        return { tracks: tracks, gap: gap, colWidth: colWidth };
    }

    function spanOf(panel) {
        var cs = window.getComputedStyle(panel);
        // `grid-column: span N` sets the start and leaves the end `auto`; when
        // a rule sets an explicit end, read that instead.
        var start = cs.gridColumnStart;
        if (start && start.indexOf('span') === 0) {
            var parsedStart = parseInt(start.split(/\s+/)[1], 10);
            if (parsedStart) return parsedStart;
        }
        var end = cs.gridColumnEnd;
        if (end && end.indexOf('span') === 0) {
            var parsedEnd = parseInt(end.split(/\s+/)[1], 10);
            if (parsedEnd) return parsedEnd;
        }
        return MAX_SPAN;
    }

    /** Storage key for a grid container: its section, or its module. */
    function scopeKey(container) {
        return container.getAttribute('data-tab') ||
            container.getAttribute('data-module') || 'dashboard';
    }

    function persistOrder(container) {
        var data = load();
        var tab = scopeKey(container);
        data[tab] = data[tab] || { items: {}, order: [] };
        data[tab].order = panelChildren(container).map(panelKey);
        save(data);
    }

    function persistItem(container, panel) {
        var data = load();
        var tab = scopeKey(container);
        data[tab] = data[tab] || { items: {}, order: [] };
        data[tab].items = data[tab].items || {};
        data[tab].items[panelKey(panel)] = {
            span: spanOf(panel),
            minHeight: parseInt(panel.style.minHeight, 10) || null
        };
        data[tab].order = panelChildren(container).map(panelKey);
        save(data);
    }

    // ------------------------------------------------------------- reorder --

    var dragState = null;

    function onDragMove(event) {
        if (!dragState) return;
        var under = document.elementFromPoint(event.clientX, event.clientY);
        var target = under && under.closest('.panel');
        if (!target || target === dragState.panel) return;
        if (target.parentElement !== dragState.container) return;
        var rect = target.getBoundingClientRect();
        var before = (event.clientX - rect.left) < rect.width / 2;
        dragState.container.insertBefore(dragState.panel, before ? target : target.nextSibling);
    }

    function onDragEnd() {
        if (!dragState) return;
        dragState.panel.classList.remove('panel-dragging');
        document.querySelectorAll('.panel-drop-target').forEach(function (el) {
            el.classList.remove('panel-drop-target');
        });
        persistOrder(dragState.container);
        dragState = null;
        document.removeEventListener('pointermove', onDragMove);
    }

    function startDrag(event) {
        var handle = event.currentTarget;
        var panel = handle.closest('.panel');
        if (!panel) return;
        event.preventDefault();
        dragState = { panel: panel, container: panel.parentElement };
        panel.classList.add('panel-dragging');
        document.addEventListener('pointermove', onDragMove);
        document.addEventListener('pointerup', onDragEnd, { once: true });
    }

    function makeDraggable(panel) {
        var header = panel.querySelector('.panel-header') || panel.querySelector('h2');
        if (!header || header.dataset.panelDrag) return;
        header.dataset.panelDrag = '1';
        var handle = document.createElement('button');
        handle.type = 'button';
        handle.className = 'panel-drag-handle';
        handle.title = 'Drag to rearrange this panel';
        handle.setAttribute('aria-label', 'Drag to rearrange this panel');
        handle.innerHTML = '<span aria-hidden="true">\u283F</span>';
        handle.addEventListener('pointerdown', startDrag);
        header.insertBefore(handle, header.firstChild);
    }

    // -------------------------------------------------------------- resize --

    function startResize(event, panel) {
        event.preventDefault();
        event.stopPropagation();
        var container = panel.parentElement;
        var metrics = gridMetrics(container);
        var startSpan = spanOf(panel);
        var rect = panel.getBoundingClientRect();
        var startHeight = rect.height;

        function onMove(ev) {
            var spanDelta = Math.round((ev.clientX - event.clientX) / (metrics.colWidth + metrics.gap));
            var span = Math.max(MIN_SPAN, Math.min(metrics.tracks, startSpan + spanDelta));
            panel.style.gridColumn = 'span ' + span;
            var height = Math.max(MIN_HEIGHT, Math.round(startHeight + (ev.clientY - event.clientY)));
            panel.style.minHeight = height + 'px';
        }

        function onUp() {
            document.removeEventListener('pointermove', onMove);
            document.removeEventListener('pointerup', onUp);
            persistItem(container, panel);
        }

        document.addEventListener('pointermove', onMove);
        document.addEventListener('pointerup', onUp);
    }

    function makeResizable(panel) {
        if (panel.querySelector(':scope > .panel-resize-grip')) return;
        var grip = document.createElement('div');
        grip.className = 'panel-resize-grip';
        grip.title = 'Drag to resize this panel';
        grip.setAttribute('role', 'separator');
        grip.setAttribute('aria-label', 'Resize panel');
        grip.addEventListener('pointerdown', function (event) {
            startResize(event, panel);
        });
        panel.appendChild(grip);
    }

    // ---------------------------------------------------------------- init --

    function applySaved(container) {
        var data = load();
        var tab = scopeKey(container);
        var saved = data[tab];
        if (!saved) return;
        var panels = panelChildren(container);
        if (Array.isArray(saved.order) && saved.order.length) {
            saved.order.forEach(function (key) {
                var match = null;
                panels.forEach(function (panel) {
                    if (panelKey(panel) === key) match = panel;
                });
                if (match) container.appendChild(match);
            });
        }
        panelChildren(container).forEach(function (panel) {
            var item = saved.items && saved.items[panelKey(panel)];
            if (!item) return;
            if (item.span) panel.style.gridColumn = 'span ' + item.span;
            if (item.minHeight) panel.style.minHeight = item.minHeight + 'px';
        });
    }

    function resetLayout() {
        try {
            window.localStorage.removeItem(STORE_KEY);
        } catch (error) {
            /* nothing to reset */
        }
        document.querySelectorAll('.tab-view > .panel, .sa-module-view > .panel').forEach(function (panel) {
            panel.style.gridColumn = '';
            panel.style.minHeight = '';
        });
    }

    function init() {
        var containers = document.querySelectorAll('.tab-view, .sa-module-view');
        if (!containers.length) return;
        document.body.classList.add('panel-layout-enabled');
        containers.forEach(function (container) {
            panelChildren(container).forEach(function (panel) {
                makeDraggable(panel);
                makeResizable(panel);
            });
            applySaved(container);
        });

        var reset = document.getElementById('sa-reset-layout');
        if (reset) reset.addEventListener('click', resetLayout);

        window.SAPanelLayout = { reset: resetLayout, key: panelKey };
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(window, document);
