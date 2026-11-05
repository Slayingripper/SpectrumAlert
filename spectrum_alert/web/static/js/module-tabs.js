/*
 * Dashboard module tabs.
 *
 * The app bar switches whole sections (Dashboard / Config / Training /
 * Workspace). Inside the dashboard there are five panels, and laying them all
 * out at once produced a page several screens tall with everything competing
 * for attention. This groups them into three modules -- Monitor, Threats,
 * Analytics -- and shows one at a time.
 *
 * Three things make it more than a click handler:
 *   - keyboard support (arrows / Home / End) per the ARIA tabs pattern;
 *   - Chart.js canvases are sized while their container is display:none, so a
 *     chart created (or a window resized) while a module is hidden comes back
 *     with a zero-width canvas. Charts are resized on every activation;
 *   - guided mode tags panels with data-guide-panel. When the walkthrough
 *     moves to a step whose panel lives in another module, the module follows
 *     it, otherwise the tour would show an empty screen.
 */
(function (window, document) {
    'use strict';

    var MODULES = ['monitor', 'threats', 'analytics'];

    // Which module holds the panels a given guide step cares about.
    var STEP_MODULE = {
        receivers: 'monitor',
        monitor: 'monitor',
        analyse: 'monitor',
        anomalies: 'threats',
        fingerprint: 'analytics',
        results: 'analytics'
    };

    var current = 'monitor';

    function tabFor(name) {
        return document.querySelector('.sa-module-tab[data-module="' + name + '"]');
    }

    function viewFor(name) {
        return document.getElementById('sa-module-' + name);
    }

    /**
     * Chart.js only measures its container while that container is visible, so
     * every chart has to be nudged once the module is on screen.
     */
    function resizeCharts() {
        window.dispatchEvent(new Event('resize'));
        var registry = window.charts || window.__charts;
        if (!registry) return;
        var visit = function (value) {
            if (!value) return;
            if (typeof value.resize === 'function') {
                try { value.resize(); } catch (error) { /* not a chart */ }
            } else if (typeof value === 'object') {
                Object.keys(value).forEach(visit);
            }
        };
        visit(registry);
    }

    function activate(name, options) {
        if (MODULES.indexOf(name) === -1) name = 'monitor';
        if (name === current && !(options && options.force)) return;
        current = name;

        MODULES.forEach(function (mod) {
            var tab = tabFor(mod);
            if (tab) {
                var active = mod === name;
                tab.setAttribute('aria-selected', active ? 'true' : 'false');
                // Roving tabindex: only the selected tab is in the tab order.
                tab.tabIndex = active ? 0 : -1;
            }
            var view = viewFor(mod);
            if (view) view.classList.toggle('is-active', mod === name);
        });

        resizeCharts();
    }

    /** Follow the guided tour: show the module that holds the step's panels. */
    function syncWithTour() {
        var openPanel = document.querySelector('.panel.is-tour-open[data-guide-panel]');
        if (!openPanel) return;
        var view = openPanel.closest('.sa-module-view');
        if (view && view.dataset.module) activate(view.dataset.module);
    }

    /**
     * Deep-link and cross-panel navigation: anything asking for a module (a
     * hash like #threats, or a control that lives in another module) lands
     * there without the caller having to know about the tab bar.
     */
    function revealPanel(panel) {
        var view = panel && panel.closest ? panel.closest('.sa-module-view') : null;
        if (view && view.dataset.module) activate(view.dataset.module);
    }

    function onKeyDown(event) {
        var index = MODULES.indexOf(current);
        var next = null;
        if (event.key === 'ArrowRight') next = MODULES[(index + 1) % MODULES.length];
        else if (event.key === 'ArrowLeft') next = MODULES[(index - 1 + MODULES.length) % MODULES.length];
        else if (event.key === 'Home') next = MODULES[0];
        else if (event.key === 'End') next = MODULES[MODULES.length - 1];
        if (!next) return;
        event.preventDefault();
        activate(next);
        var tab = tabFor(next);
        if (tab) tab.focus();
    }

    /**
     * A control that lives in a hidden module cannot be operated, and its chart
     * never sized itself. Clicking a guide action, or anything tagged
     * data-module-target, reveals the module that owns it first.
     */
    function installRevealOnClick() {
        document.addEventListener('click', function (event) {
            var trigger = event.target.closest('[data-module-target]');
            if (trigger) {
                activate(trigger.getAttribute('data-module-target'));
                return;
            }
            var inHiddenPanel = event.target.closest('.sa-module-view:not(.is-active)');
            if (!inHiddenPanel) return;
            var action = event.target.closest(
                '.btn-start, .btn-stop, .btn-save, .btn-deploy, .btn-collect, ' +
                '.btn-analyze, .btn-test, #sa-guide-workspace, #sa-guide-triangulate, ' +
                '#sa-guide-fingerprint'
            );
            if (action) revealPanel(inHiddenPanel);
        }, true);
    }

    /** Badge counts on the tabs, so triage does not need visiting each module. */
    function setCount(module, value, tone) {
        var badge = document.getElementById('sa-module-count-' + module);
        if (!badge) return;
        if (value === null || value === undefined || !Number(value)) {
            badge.hidden = true;
            badge.textContent = '';
            return;
        }
        badge.hidden = false;
        badge.textContent = String(value);
        if (tone) badge.setAttribute('data-tone', tone);
        else badge.removeAttribute('data-tone');
    }

    function init() {
        var bar = document.getElementById('sa-module-tabs');
        if (!bar) return;

        bar.addEventListener('click', function (event) {
            var tab = event.target.closest('.sa-module-tab');
            if (tab) activate(tab.getAttribute('data-module'));
        });
        bar.addEventListener('keydown', onKeyDown);

        MODULES.forEach(function (mod) {
            var tab = tabFor(mod);
            if (tab) {
                tab.tabIndex = mod === 'monitor' ? 0 : -1;
                tab.addEventListener('click', function () { activate(mod); });
            }
        });

        // Honour a #module hash so a module is linkable like a section is.
        var hash = window.location.hash.slice(1);
        if (MODULES.indexOf(hash) !== -1) activate(hash);

        installRevealOnClick();

        // Guided mode moves the panels around after this runs, so watch for the
        // class it toggles rather than reading it once at load.
        if (window.MutationObserver) {
            new window.MutationObserver(syncWithTour).observe(document.body, {
                attributes: true,
                attributeFilter: ['class'],
                subtree: true
            });
        }

        window.SAModuleTabs = {
            activate: activate,
            setCount: setCount,
            revealPanel: revealPanel,
            current: function () { return current; }
        };
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(window, document);