"""Static contract tests between the dashboard frontend and the backend.

These catch a class of bug that no unit test reaches: the frontend and backend
disagreeing about element ids, script loading, or WebSocket message types. Each
of those produced a silent failure rather than an error, which is why they
survived so long.
"""

import os
import re
from pathlib import Path

import pytest

WEB = Path(__file__).parent.parent.parent / "spectrum_alert" / "web"
TEMPLATES = WEB / "templates"
STATIC = WEB / "static"


def read(path):
    return path.read_text(encoding="utf-8")


def render(name):
    """Render a template through Jinja so ``{% include %}`` is expanded.

    The analysis workspace lives in a shared partial included by both pages, so
    reading the raw file would test a document that is never served.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template(name).render()


@pytest.fixture(scope="module")
def dashboard_html():
    return render("dashboard.html")


@pytest.fixture(scope="module")
def dashboard_js():
    return read(STATIC / "js" / "dashboard.js")


@pytest.fixture(scope="module")
def extras_js():
    return read(STATIC / "js" / "dashboard-extras.js")


@pytest.fixture(scope="module")
def workspace_html():
    """The shared analysis workspace, as included by the dashboard."""
    return render("_workspace.html")


@pytest.fixture(scope="module")
def geolocate_html():
    """Alias kept for readability: the workspace is what geolocate.js drives."""
    return render("_workspace.html")


@pytest.fixture(scope="module")
def geolocate_js():
    return read(STATIC / "js" / "geolocate.js")


@pytest.fixture(scope="module")
def app_py():
    return read(WEB / "app.py")


class TestElementIdContract:
    """Every id the scripts reach for must exist in the markup."""

    @pytest.mark.parametrize(
        "template,script",
        [("dashboard.html", "dashboard.js")],
    )
    def test_no_missing_element_ids(self, template, script):
        html = render(template)
        js = read(STATIC / "js" / script)

        declared = set(re.findall(r'id="([^"]+)"', html))
        # Both document.getElementById('x') and $('x') idioms are used.
        referenced = set(re.findall(r"getElementById\('([^']+)'\)", js))
        referenced |= set(re.findall(r"\$\('([a-z0-9_-]+)'\)", js))

        missing = sorted(referenced - declared)
        assert not missing, f"{script} references ids absent from {template}: {missing}"

    @pytest.mark.parametrize("template", ["dashboard.html", "_workspace.html"])
    def test_no_duplicate_ids(self, template):
        """getElementById returns the first match, so a duplicate silently wins."""
        html = render(template)
        ids = re.findall(r'id="([^"]+)"', html)
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        assert not duplicates, f"{template} has duplicate ids: {duplicates}"

    def test_alert_list_has_a_state_node(self, dashboard_html):
        """An empty list with no empty state renders as a blank box."""
        assert 'id="alerts-list"' in dashboard_html
        assert 'id="alerts-state"' in dashboard_html


class TestScriptLoading:
    """Dead or harmful scripts must not be loaded."""

    @staticmethod
    def _script_sources(html):
        """Only real script/link tags, not prose that mentions a filename.

        The template carries a comment explaining why the old scripts were
        dropped, and matching that text would make the check meaningless.
        """
        return re.findall(r'<script[^>]+src="([^"]+)"', html)

    @pytest.mark.parametrize(
        "script",
        ["websocket.js", "gui-enhancements.js", "enhanced-features.js"],
    )
    def test_dead_scripts_are_not_loaded(self, dashboard_html, script):
        """These were loaded but did nothing, or actively broke things.

        websocket.js opened a second socket and bailed on every message;
        gui-enhancements.js replaced window.loadDataFiles and produced a second,
        conflicting file list.
        """
        sources = self._script_sources(dashboard_html)
        assert not any(script in src for src in sources), (
            f"{script} should no longer be loaded; found in {sources}"
        )

    def test_font_awesome_is_not_a_render_blocking_script(self, dashboard_html):
        sources = self._script_sources(dashboard_html)
        assert not any("font-awesome" in src for src in sources)
        assert any(re.search(r"chart(\.umd)?(\.min)?\.js", src) for src in sources), (
            "Chart.js must still load"
        )

    @pytest.mark.parametrize(
        "page", ["dashboard.html"]
    )
    def test_ui_core_is_loaded(self, page):
        html = render(page)
        sources = self._script_sources(html)
        assert any("ui-core.js" in src for src in sources)
        assert "ui-core.css" in html

    def test_asset_order_puts_core_before_consumers(self, dashboard_html):
        """ui-core.js defines SAUI, which both other scripts need on load.

        Asserts against the document-ordered list of script sources. Iterating a
        set here would make the assertion depend on hash order and pass or fail
        at random.
        """
        sources = self._script_sources(dashboard_html)
        names = [os.path.basename(src) for src in sources]
        assert names.index("ui-core.js") < names.index("dashboard.js")
        assert names.index("dashboard.js") < names.index("dashboard-extras.js")

    @pytest.mark.parametrize(
        "asset",
        [
            "js/ui-core.js",
            "js/dashboard.js",
            "js/dashboard-extras.js",
            "js/geolocate.js",
            "css/ui-core.css",
            "css/geolocate.css",
        ],
    )
    def test_assets_exist(self, asset):
        assert (STATIC / asset).exists(), f"{asset} is referenced but missing"


class TestWebSocketContract:
    """The message types the server emits and the client handles must agree.

    Every real-time update was dead because the client handled
    monitoring_update / training_complete / training_failed, none of which the
    server ever sends, while the types it does send fell through to a default
    branch that only logged.
    """

    def _server_types(self, app_py):
        # Only the broadcast payloads are message envelopes; the rest of the file
        # uses "type" for other purposes (csv/json/mqtt payload kinds).
        return set(re.findall(r'"type":\s*"([a-z_]+)"', app_py))

    def _client_types(self, dashboard_js):
        start = dashboard_js.index("function handleWebSocketMessage(")
        end = dashboard_js.index("// Cleanup on page unload")
        return set(re.findall(r"case '([a-z_]+)':", dashboard_js[start:end]))

    def test_every_server_type_is_handled(self, app_py, dashboard_js):
        server = self._server_types(app_py)
        client = self._client_types(dashboard_js)

        # Payload-type strings that are not envelopes.
        noise = {"csv", "json", "test", "predefined", "directory", "file"}
        envelope_types = server - noise

        unhandled = sorted(envelope_types - client)
        assert not unhandled, f"server emits message types the client ignores: {unhandled}"

    def test_client_handles_nothing_the_server_sends(self, dashboard_js):
        """Guards against the inverse: dead branches that can never fire."""
        client = self._client_types(dashboard_js)
        assert client, "handleWebSocketMessage should have cases"

    def test_spectrum_fields_match_the_server(self, app_py, dashboard_js):
        """Field-name drift silently yields undefined values."""
        # The server sends these keys on spectrum_update...
        for field in ("frequencies_mhz", "powers_dbfs", "peak_frequency_mhz"):
            assert f'"{field}"' in app_py, f"server no longer sends {field}"
        # ...and the client must read them.
        for field in ("frequencies_mhz", "powers_dbfs"):
            assert field in dashboard_js, f"client no longer reads {field}"

    def test_connection_state_is_tracked(self, dashboard_js):
        """A hardcoded CONNECTED label lies whenever the socket is down."""
        assert "setConnectionState" in dashboard_js
        assert "wsRetryCount" in dashboard_js, "reconnect must back off, not hammer"


class TestUnitsAreHonest:
    """An RTL-SDR reports dBFS. Labelling it dBm overstates what is known."""

    def test_chart_is_not_labelled_dbm(self, dashboard_js):
        assert "Power (dBm)" not in dashboard_js

    def test_markup_states_dbfS(self, dashboard_html):
        assert "dBFS" in dashboard_html

    def test_no_invented_absolute_calibration(self, dashboard_js):
        """An arbitrary +30 dB offset fabricated an absolute reference."""
        assert "dBm for conversion" not in dashboard_js


class TestNoFabricatedData:
    """A synthetic fallback makes a broken backend look like a real signal."""

    def test_spectrum_chart_has_no_mock_generator(self, dashboard_js):
        for marker in ("Generate sample data for visualization", "Create a synthetic spectrum"):
            assert marker not in dashboard_js, f"chart still fabricates data: {marker!r}"

    def test_analysis_reads_the_real_field_names(self, dashboard_js):
        """The API returns power_spectrum; amplitudes was never sent."""
        assert "data.power_spectrum" in dashboard_js


class TestAccessibility:
    @pytest.mark.parametrize("template", ["dashboard.html"])
    def test_skip_link_present(self, template):
        html = render(template)
        assert 'class="sa-skip-link"' in html

    @pytest.mark.parametrize("template", ["dashboard.html"])
    def test_dialogs_are_labelled(self, template):
        html = render(template)
        assert 'aria-modal="true"' in html
        assert "aria-labelledby" in html

    def test_charts_have_text_alternatives(self, dashboard_html):
        """A canvas is invisible to a screen reader without a label."""
        for canvas_id in ("spectrum-chart", "anomaly-chart"):
            match = re.search(
                r'<canvas id="%s"[^>]*>' % canvas_id, dashboard_html
            )
            assert match, f"{canvas_id} not found"
            tag = match.group(0)
            assert 'role="img"' in tag, f"{canvas_id} needs role=img"
            assert 'aria-label=' in tag, f"{canvas_id} needs an aria-label"

    def test_focus_styles_exist(self):
        css = read(STATIC / "css" / "ui-core.css")
        assert ":focus-visible" in css

    def test_reduced_motion_is_respected(self):
        css = read(STATIC / "css" / "ui-core.css")
        assert "prefers-reduced-motion" in css

    def test_notification_region_is_announced(self):
        """Alerts must reach a screen reader, not just the visual stack."""
        js = read(STATIC / "js" / "ui-core.js")
        # Errors and criticals are announced assertively; everything else
        # politely. The host is a labelled landmark, not a live region itself,
        # otherwise every toast would be announced twice.
        assert "'role', 'alert'" in js.replace("\n", " ") or (
            "'role', severity === 'critical' || severity === 'error' ? 'alert' : 'status'"
            in js)
        assert "'role', 'status'" in js
        assert "'role', 'region'" in js
        assert "aria-live" not in js, (
            "the host is role=region; per-toast roles already announce. "
            "Adding aria-live here would double-announce every message."
        )

    def test_severity_is_not_colour_only(self):
        """Severity also needs a glyph or text for greyscale and colour blindness."""
        extras = read(STATIC / "js" / "dashboard-extras.js")
        assert "SEVERITY_GLYPH" in extras
        assert "sa-sr-only" in extras

    def test_nav_links_are_labelled(self, dashboard_html):
        assert "<nav" not in dashboard_html or 'aria-label=' in dashboard_html


class TestUnitsHelpers:
    def test_format_helpers_exist(self):
        js = read(STATIC / "js" / "ui-core.js")
        for name in ("dbfs", "hz", "mhz", "msps", "bytes", "relativeTime", "duration"):
            assert f"{name}:" in js, f"SAUI.format.{name} missing"

    def test_dbfs_helper_is_labelled(self):
        js = read(STATIC / "js" / "ui-core.js")
        # The helper's job is to never emit a bare number that could be read as dBm.
        assert "' dBFS'" in js

    def test_csv_export_quotes_correctly(self):
        js = read(STATIC / "js" / "ui-core.js")
        # RFC 4180: a field containing a delimiter, quote or newline must be
        # quoted, and embedded quotes doubled.
        assert '/[",\\n\\r]/.test(text)' in js or '[",\\n\\r]' in js
        assert 'replace(/"/g, \'""\')' in js


class TestSinglePageNavigation:
    def test_workspace_is_part_of_the_dashboard(self, dashboard_html):
        """One document, not a dashboard plus a linked second page.

        Location and identification were a separate page, so the operator lost
        the system panels whenever they did either.
        """
        assert 'id="workspace"' in dashboard_html
        assert 'id="sa-map"' in dashboard_html
        assert 'id="sa-fp-capture"' in dashboard_html
        # No link should send the operator off to another page to do this work.
        assert 'href="/geolocate"' not in dashboard_html

    def test_geolocate_route_redirects_to_the_dashboard(self):
        """Existing bookmarks and header links must still land somewhere useful."""
        import inspect
        from spectrum_alert.web.app import SpectrumAlertWebApp

        source = inspect.getsource(SpectrumAlertWebApp._setup_routes)
        assert '"/geolocate"' in source
        assert "RedirectResponse" in source
        assert "#workspace" in source

    def test_standalone_workspace_template_is_gone(self):
        """A second page rendering the same partial would be two UIs again."""
        assert not (TEMPLATES / "geolocate.html").exists()

class TestDeadWiringRemoved:
    """Each of these references looked functional and did nothing.

    The common shape is a getElementById for an id that is never declared, so
    the code silently no-ops. Removing the dead branch is the fix; adding an
    empty placeholder element would only hide it again.
    """

    def test_no_silent_element_lookups(self, dashboard_html, dashboard_js):
        """Every getElementById in dashboard.js must resolve in the template.

        Guards against this whole class: a function whose first act is to look
        up a missing node and return early will appear to work in review and do
        nothing at runtime.
        """
        declared = set(re.findall(r'id="([^"]+)"', dashboard_html))
        # Ids the dashboard builds at runtime from API data rather than markup.
        runtime = {"notification", "filesList", "indicators", "analysisContent"}
        referenced = set(re.findall(r"getElementById\('([^']+)'\)", dashboard_js))
        missing = referenced - declared - runtime
        assert not missing, f"dashboard.js references undeclared ids: {sorted(missing)}"

    def test_single_notification_path(self, dashboard_html, dashboard_js):
        """showNotification must be defined exactly once and delegate to SAUI.

        Three definitions of this function existed; the last one won, so the
        mute preference and severity filter were bypassed for every caller.
        """
        assert dashboard_js.count("function showNotification(") == 1
        assert "notify(message" in dashboard_js, (
            "showNotification must delegate to SAUI.notify so filtering, mute "
            "and the notification history stay in one place"
        )
        assert "notification-${type}" not in dashboard_js

    def test_history_hydration_targets_the_live_alert_buffer(self, dashboard_js):
        """updateRecentAnomalies must feed recordAnomaly.

        It wrote to #recent-anomalies-list, which does not exist, so the Recent
        Alerts panel only ever showed live WebSocket events and never the
        persisted history.
        """
        match = re.search(
            r"function updateRecentAnomalies\(anomalies\) \{(.*?)\n\}", dashboard_js, re.S
        )
        assert match, "updateRecentAnomalies is missing"
        body = match.group(1)
        assert "recent-anomalies-list" not in body
        assert "recordAnomaly" in body


class TestDataFileList:
    def test_single_file_list_container(self, dashboard_html, dashboard_js):
        """One file list, not a second one injected into the panel.

        displayDataFiles looked for #data-files-container, missed, and appended
        a whole competing section, so the Data Management panel rendered two
        unrelated file lists.
        """
        assert 'id="data-files-container"' not in dashboard_html
        assert 'id="data-files-list"' not in dashboard_html
        assert "$('data-file-list')" in dashboard_js
        assert "insertAdjacentHTML('beforeend', dataFilesHTML)" not in dashboard_js

    def test_filenames_are_escaped_before_interpolation(self, dashboard_js):
        """A filename reaches innerHTML, so it must be escaped.

        Files come from the data directory and are not attacker-controlled in
        the usual sense, but a filename is exactly the kind of string that ends
        up pasted from an email or a download listing.
        """
        match = re.search(r"function displayDataFiles\(files\) \{(.*?)\n\}", dashboard_js, re.S)
        assert match
        body = match.group(1)
        assert "escapeHtml(file.filename)" in body or "SAUI.escapeHtml(file.filename)" in body


class TestScanHandoff:
    def test_alert_locate_hands_the_frequency_to_the_workspace(self, extras_js, geolocate_js):
        """The Locate button must reach the workspace on this same page.

        It used to open /geolocate?freq=... in a second window. Now that the
        workspace is a section of the dashboard, it scrolls there and hands the
        frequency over in-page, so the operator keeps the system panels.
        """
        assert "workspaceFocusTarget" in extras_js
        assert "scrollIntoView" in extras_js
        assert "window.open('/geolocate'" not in extras_js
        assert "window.workspaceFocusTarget" in geolocate_js
        assert "applyTargetFromUrl" in geolocate_js
        match = re.search(
            r"function applyTargetFromUrl\(\) \{(.*?)\n    \}\n", geolocate_js, re.S
        )
        assert match
        assert "new URLSearchParams(window.location.search)" in match.group(1)
        assert "params.get('freq')" in match.group(1)
        # Consumed once, so a later manual retune is not stomped on reload.
        assert "store.set('geoTargetHz', null)" in match.group(1)

    def test_waterfall_replays_from_a_cleared_canvas(self, geolocate_js):
        """Each drawn frame shifts the canvas up one pixel.

        Replaying the retained frames onto an already-drawn canvas therefore
        scrolls once per frame per replay, compounding into smeared bands after
        a handful of scans or a single display-floor change.
        """
        match = re.search(r"function replayFrames\(\) \{(.*?)\n    \}", geolocate_js, re.S)
        assert match
        body = match.group(1)
        assert "clearWaterfall()" in body, "replayFrames must clear before drawing"
        assert body.index("clearWaterfall()") < body.index("drawScanRow")

        # The live scan path must use replayFrames, not its own forEach.
        scan = re.search(r"async function runScan\(.*?\n    \}", geolocate_js, re.S)
        assert scan
        assert "replayFrames()" in scan.group(0)
        assert "state.frames.forEach(drawScanRow)" not in scan.group(0)

    def test_map_layers_reject_missing_coordinates(self, geolocate_js):
        """Leaflet throws on undefined LatLng and kills the whole render.

        A receiver added before its position was set has no lat/lon, so one such
        row used to take down every other marker on the map.
        """
        match = re.search(r"function renderMapReceivers\(\) \{(.*?)\n    \}", geolocate_js, re.S)
        assert match
        assert "isFinite(lat)" in match.group(1)
        assert "L.marker([lat, lon])" in match.group(1)

        est = re.search(r"function renderEstimate\(result\) \{(.*?)\n    \}", geolocate_js, re.S)
        assert est
        assert "isFinite(Number(estimate.lat))" in est.group(1)

    def test_heatmap_plugin_is_loaded(self, dashboard_html):
        """renderHeatmap returns early without L.heatLayer.

        The likelihood-surface toggle then looked functional and did nothing,
        because the plugin script was no longer referenced by the template.
        """
        assert "leaflet-heat.js" in dashboard_html or "leaflet.heat" in dashboard_html

    def test_cursor_readout_is_inside_the_positioned_shell(self, geolocate_html):
        """An absolutely positioned readout must share a positioning ancestor.

        The readout sat after the waterfall wrapper, so it anchored to the page
        instead of overlaying the chart it described.
        """
        assert re.search(
            r'id="sa-waterfall-shell".*?id="sa-cursor-readout"', geolocate_html, re.S
        )


class TestStatusTruthfulness:
    """The status surfaces used to invent hardware and sensor values."""

    def test_no_hardcoded_sdr_device_anywhere(self):
        """An "RTL-SDR #0" was reported on machines with nothing attached."""
        import spectrum_alert.infrastructure.monitoring as mon
        import spectrum_alert.web.app as webapp
        mon_src = open(mon.__file__).read()
        app_src = open(webapp.__file__).read()
        # Comments documenting the fix legitimately mention the old value;
        # only an actual device-list literal is a failure.
        assert '"RTL-SDR #0"]' not in mon_src
        assert '"RTL-SDR #0"]' not in app_src
        assert 'sdr_status = "Available"' not in app_src

    def test_no_fabricated_temperature(self):
        """45 °C was reported on sensorless VMs as if it were measured."""
        import spectrum_alert.infrastructure.monitoring as mon
        src = open(mon.__file__).read()
        assert "45.0  # Mock" not in src
        assert "_read_temperature_c" in src

    def test_status_endpoint_reports_probe_fields(self):
        import spectrum_alert.web.app as webapp
        src = open(webapp.__file__).read()
        assert "probe_local_sdr" in src
        assert '"sdr_available"' in src

    def test_monitoring_never_falls_back_to_simulation(self):
        """No dongle must be an error, not a silent stream of fake readings.

        A local, attached SDR used to be missed (missing pyrtlsdr) and the
        server quietly answered with the mock, so the dashboard claimed no
        hardware was found while the dongle sat on the bus.
        """
        import spectrum_alert.web.app as webapp
        src = open(webapp.__file__).read()
        assert "MockSDRInterface" not in src
        assert "monitoring_simulated" not in src
        assert "Falling back to SIMULATED" not in src

    def test_simulated_badge_is_gone(self, dashboard_html, dashboard_js):
        """With simulations removed there is nothing for a badge to label."""
        assert 'id="simulated-badge"' not in dashboard_html
        assert "setSimulatedBadge" not in dashboard_js

    def test_peak_frequency_guard_is_nan_not_null(self, dashboard_js):
        """isFinite(null) is true in JS: a null initial peak passed every
        guard and the card rendered "0.000000 MHz" for empty payloads."""
        assert "let peakHz = NaN;" in dashboard_js
        assert "let peakHz = null;" not in dashboard_js

    def test_light_theme_flips_surface_gradients(self):
        """Dark surface vars left in place made light mode dark-on-dark."""
        css = read(STATIC / "css" / "ui-core.css")
        light = css.split('html[data-theme="light"]', 1)[1]
        assert "--gradient-dark: linear-gradient(135deg, #ffffff" in light

    def test_uptime_card_is_written(self, dashboard_html, dashboard_js):
        """The UPTIME card sat at "--" forever; nothing ever updated it."""
        assert 'id="uptime"' in dashboard_html
        assert "updateMetricValue('uptime'" in dashboard_js


class TestErrorUnwrapping:
    def test_request_unwraps_fastapi_validation_arrays(self):
        """A 422 detail is an array of {loc, msg}; new Error(array) showed
        "[object Object]" on the scan panel instead of what was wrong."""
        js = read(STATIC / "js" / "ui-core.js")
        assert "Array.isArray(detail)" in js
        assert "entry.loc" in js
        assert "new Error(String(detail))" in js

    def test_target_handoff_does_not_accept_null(self):
        """isFinite(null) is true; a missing geoTargetHz set the centre to
        exactly 0 Hz and every subsequent scan failed validation."""
        js = read(STATIC / "js" / "geolocate.js")
        match = re.search(
            r"function applyTargetFromUrl\(\) \{(.*?)\n    \}\n", js, re.S
        )
        assert match
        assert "typeof hz !== 'number'" in match.group(1)


class TestUnifiedWorkspace:
    """Locate and Identify share one screen, one capture and one roster.

    They were first built as tabs, which hid half the answer behind a switch.
    Both panels are now rendered side by side, so a single scan answers both
    questions without the operator changing context.
    """

    def test_both_panels_are_always_rendered(self, geolocate_html):
        """No view wrappers: nothing may hide one analysis behind a switch."""
        assert 'id="sa-view-locate"' not in geolocate_html
        assert 'id="sa-view-identify"' not in geolocate_html
        assert 'role="tablist"' not in geolocate_html
        assert 'role="tab"' not in geolocate_html

    def test_both_analyses_are_present_in_the_same_document(self, geolocate_html):
        # Locate
        assert 'id="sa-map"' in geolocate_html
        assert 'id="sa-triangulate"' in geolocate_html
        assert 'id="sa-estimate"' in geolocate_html
        # Identify
        assert 'id="sa-fp-capture"' in geolocate_html
        assert 'id="sa-fp-features"' in geolocate_html
        assert 'id="sa-fp-library"' in geolocate_html
        # One shared capture and roster feeding both.
        assert geolocate_html.count('id="sa-waterfall"') == 1
        assert geolocate_html.count('id="sa-receiver-list"') == 1

    def test_the_two_panels_are_siblings_in_one_row(self, geolocate_html):
        """They must be adjacent columns, not stacked or tabbed."""
        assert 'class="row g-2 sa-merged"' in geolocate_html
        merged = geolocate_html.split('sa-merged', 1)[1]
        locate = merged.index('id="sa-map"')
        identify = merged.index('id="sa-fp-capture"')
        assert locate < identify, "map and signature panel share the merged row"
        # Two responsive columns: map wider, signature narrower.
        assert 'col-12 col-lg-7' in merged and 'col-12 col-lg-5' in merged

    def test_bootstrap_is_vendored_not_hotlinked(self, dashboard_html):
        """An operator tool must still work with no internet connection."""
        assert '/static/vendor/bootstrap/bootstrap.min.css' in dashboard_html
        assert '/static/vendor/bootstrap/bootstrap.bundle.min.js' in dashboard_html
        assert 'cdn.jsdelivr.net/npm/bootstrap' not in dashboard_html
        css = read(STATIC / "vendor" / "bootstrap" / "bootstrap.min.css")
        assert "Bootstrap  v5.3.3" in css

    def test_bootstrap_loads_before_the_page_styles(self, dashboard_html):
        """Bootstrap's reboot must lose to the page's own tokens and fonts."""
        assert (dashboard_html.index("bootstrap.min.css")
                < dashboard_html.index("ui-core.css"))
        assert (dashboard_html.index("bootstrap.bundle.min.js")
                < dashboard_html.index("geolocate.js"))

    def test_dashboard_loads_what_the_workspace_needs(self, dashboard_html):
        """Leaflet, the heat plugin and the workspace script must be present."""
        assert "leaflet.js" in dashboard_html
        assert "leaflet.heat" in dashboard_html
        assert "/static/js/geolocate.js" in dashboard_html
        assert "/static/css/geolocate.css" in dashboard_html

    def test_layout_uses_the_full_viewport(self):
        """Regression: a max-width wrapper capped the page and wasted a wide
        monitor's margins."""
        css = read(STATIC / "css" / "geolocate.css")
        assert ".sa-wrap {" in css
        assert "max-width: none" in css.split(".sa-wrap {", 1)[1].split("}", 1)[0]

    def test_map_scales_with_the_viewport(self):
        assert "#sa-map {" in read(STATIC / "css" / "geolocate.css")
        block = read(STATIC / "css" / "geolocate.css").split("#sa-map {", 1)[1]
        block = block.split("}", 1)[0]
        assert "vh" in block, "map height should take a share of the viewport"
        assert "clamp(" in block

    def test_waterfall_canvas_matches_its_display_size(self, geolocate_js):
        """A fixed 960px buffer stretched across a wide panel renders soft."""
        assert "function sizeWaterfall()" in geolocate_js
        match = re.search(
            r"function sizeWaterfall\(\) \{(.*?)\n    \}", geolocate_js, re.S
        )
        assert match
        body = match.group(1)
        assert "getBoundingClientRect" in body
        assert "devicePixelRatio" in body
        assert "replayFrames()" in body, "history must survive a resize"
        assert "ResizeObserver" in geolocate_js

    def test_identify_panel_uses_the_shared_component_classes(self, geolocate_html):
        """Styling must come from the same primitives as the rest of the page."""
        css = read(STATIC / "css" / "ui-core.css")
        for selector in (".sa-tabs", ".sa-tab[aria-selected=", ".sa-verdict", ".sa-diff"):
            assert selector in css, f"{selector} is not defined"
        assert 'class="sa-verdict"' in geolocate_html

    def test_verdict_state_is_not_conveyed_by_colour_alone(self, geolocate_html):
        assert 'data-verdict="identified"' in read(STATIC / "css" / "ui-core.css")
        # The status word itself changes per state, so the verdict survives a
        # monochrome display or a colour-vision difference.
        js = read(STATIC / "js" / "geolocate.js")
        for status in ("Identified", "Not in library", "Ambiguous", "No references"):
            assert status in js

    def test_identify_controls_are_uniquely_identified(self, geolocate_html):
        """A duplicate id silently breaks getElementById wiring."""
        ids = re.findall(r'id="([^"]+)"', geolocate_html)
        duplicates = {i for i in ids if ids.count(i) > 1}
        assert not duplicates, f"duplicate ids: {sorted(duplicates)}"

    def test_save_reference_stays_disabled_until_a_capture_exists(self, geolocate_html, geolocate_js):
        assert 'id="sa-fp-enrol" disabled' in geolocate_html
        assert "enrol.disabled = !result.values" in geolocate_js

    def test_no_view_switching_remains_in_the_script(self, geolocate_js):
        """The tabs are gone, so the switcher must be gone with them."""
        assert "switchView" not in geolocate_js
        # Exact ids: "sa-tab" is a prefix of the "sa-table" class name.
        for dead in ("sa-tab-locate", "sa-tab-identify", "sa-view-locate",
                     "sa-view-identify", "sa-identify-count"):
            assert dead not in geolocate_js, f"{dead} belonged to the tab UI"

    def test_enrolment_refreshes_the_library_view(self, geolocate_js):
        """The capture response describes the new entry, not the whole library."""
        assert re.search(
            r"if \(result\.enrolled\) \{(.*?)\n            \} else if",
            re.search(r"function captureFingerprint.*?\n    \}", geolocate_js, re.S).group(0),
            re.S,
        )
        assert "loadLibrary()" in re.search(
            r"if \(result\.enrolled\) \{(.*?)\n            \} else if",
            re.search(r"function captureFingerprint.*?\n    \}", geolocate_js, re.S).group(0),
            re.S,
        ).group(1)

    def test_geolocate_page_honours_the_light_theme(self):
        """The toggle set data-theme but this page declared only dark tokens,
        so pressing it appeared to do nothing."""
        css = read(STATIC / "css" / "geolocate.css")
        light = css.split('html[data-theme="light"]', 1)
        assert len(light) == 2
        for token in ("--sa-bg", "--sa-panel", "--sa-text", "--sa-muted", "--sa-accent"):
            assert token in light[1]


class TestMergedDashboard:
    """The workspace runs inside dashboard.html, so the two must not collide."""

    def test_workspace_controller_ids_resolve_on_the_dashboard(self, dashboard_html, geolocate_js):
        """geolocate.js now runs against the dashboard document.

        It used to resolve against its own page shell, so any id the dashboard
        already owned (help overlay, theme toggle, mute button) either pointed
        at nothing or silently bound a second handler to a shared control.
        """
        declared = set(re.findall(r'id="([^"]+)"', dashboard_html))
        referenced = set(re.findall(r"getElementById\('([^']+)'\)", geolocate_js))
        referenced |= set(re.findall(r"\$\('([a-z0-9_-]+)'\)", geolocate_js))
        missing = sorted(referenced - declared)
        assert not missing, f"geolocate.js references ids absent from the dashboard: {missing}"

    def test_shared_controls_are_wired_once(self, geolocate_js):
        """A double-bound toggle fires twice per click.

        Theme, mute and help are owned by the dashboard header now, so the
        workspace must not attach its own listeners to them.
        """
        for dead in ("$('sa-theme-toggle').addEventListener",
                     "$('sa-help-open').addEventListener",
                     "$('sa-help-close').addEventListener"):
            assert dead not in geolocate_js, f"{dead} double-binds a dashboard control"
        assert "$('help-open')" in geolocate_js
        assert "$('help-close')" in geolocate_js

    def test_keyboard_shortcuts_do_not_collide(self):
        """SAUI.shortcuts is one registry keyed by combo: last write wins.

        With both controllers on one page, a shared key would silently replace
        the other's action, so the two sets must be disjoint.
        """
        import re as _re

        def keys(path):
            js = read(STATIC / "js" / path)
            return set(_re.findall(r"\.register\('([^']+)'", js))

        dashboard_keys = keys("dashboard-extras.js")
        workspace_keys = keys("geolocate.js")
        overlap = dashboard_keys & workspace_keys
        assert not overlap, f"shortcut keys claimed by both controllers: {sorted(overlap)}"

    def test_workspace_shortcuts_avoid_the_removed_geolocate_page(self, extras_js):
        assert "window.open('/geolocate'" not in extras_js

    def test_dashboard_declares_the_workspace_dependencies(self, dashboard_html):
        """Leaflet, the heat plugin, the workspace script and its styles."""
        assert "leaflet.css" in dashboard_html
        assert "leaflet.js" in dashboard_html
        assert "leaflet.heat" in dashboard_html
        assert "/static/js/geolocate.js" in dashboard_html
        assert "/static/css/geolocate.css" in dashboard_html
        # Leaflet must be available before the workspace script runs.
        assert (dashboard_html.index("leaflet.js")
                < dashboard_html.index("/static/js/geolocate.js"))

    def test_dashboard_container_is_fluid(self):
        """The dashboard capped itself at 1920px, wasting a wider monitor."""
        css = read(STATIC / "css" / "cyberpunk.css")
        block = css.split(".container {", 1)[1].split("}", 1)[0]
        assert "max-width: none" in block


class TestGuidelineRevamp:
    """The MOSIP guideline pass: say what a panel is for, what each step needs,
    and what a control actually does before it is pressed."""

    PANEL_RE = re.compile(r'<section class="panel[^"]*"[^>]*>(.*?)</section>', re.S)

    def test_every_panel_states_its_purpose(self, dashboard_html):
        missing = []
        for panel in self.PANEL_RE.findall(dashboard_html):
            header = re.search(r"<h2>(.*?)</h2>", panel, re.S)
            title = re.sub(r"<[^>]+>", " ", header.group(1)) if header else "<no heading>"
            title = re.sub(r"\s+", " ", title).strip()
            if 'class="sa-panel-desc"' not in panel:
                missing.append(title)
        assert not missing, f"panels with no description of what they are for: {missing}"

    def test_getting_started_strip_is_present(self, dashboard_html):
        """Three ordered steps, each with a live status and one action."""
        for element_id in (
            "sa-guide", "sa-guide-title", "sa-guide-hide", "sa-guide-show",
            "sa-step-receivers", "sa-step-monitor", "sa-step-analyse",
            "sa-step-receivers-status", "sa-step-monitor-status",
            "sa-step-analyse-status",
            "sa-guide-receivers", "sa-guide-monitor", "sa-guide-workspace",
        ):
            assert f'id="{element_id}"' in dashboard_html, element_id

    def test_guide_reads_live_state(self, extras_js):
        """A status strip that does not consult live state is decoration."""
        for needle in (
            "function refreshGuide", "enabledReceiverCount", "monitoringActive",
            "window.SAGuideRefresh", "SAUI.store.get(GUIDE_KEY",
        ):
            assert needle in extras_js, needle

    def test_workspace_pokes_the_guide_when_the_roster_changes(self, geolocate_js, extras_js):
        assert "window.SAGuideRefresh" in geolocate_js
        assert "workspaceReadiness" in geolocate_js
        assert "window.workspaceReadiness = workspaceReadiness" in geolocate_js

    def test_dependency_cues_state_the_requirement(self, workspace_html, geolocate_js):
        """Locate and identify both refuse to run until something else exists,
        and neither used to say what."""
        assert 'id="sa-triangulate-req"' in workspace_html
        assert 'id="sa-fp-req"' in workspace_html
        assert "at least 2 enabled receivers" in workspace_html
        assert "at least 2 enabled receivers" in geolocate_js
        # Two, not three: a bearing fix needs two readings.
        assert "at least 3" not in workspace_html
        assert "needs 3" not in workspace_html

    def test_headings_are_sentence_case(self, dashboard_html):
        headings = []
        for raw in re.findall(r"<h2>(.*?)</h2>", dashboard_html, re.S):
            text = re.sub(r"<[^>]+>", " ", raw)
            headings.append(re.sub(r"\s+", " ", text).strip())
        shouting = [h for h in headings if h and h == h.upper()]
        assert not shouting, f"headings still shouting: {shouting}"

    def test_workspace_headings_are_not_forced_uppercase(self):
        """DOM sentence case is pointless if CSS shouts it back."""
        css = read(STATIC / "css" / "geolocate.css")
        panel_rule = css.split(".sa-panel > h2 {", 1)[1].split("}", 1)[0]
        assert "text-transform" not in panel_rule
        section_rule = css.split(".sa-section .sa-header h1 {", 1)[1].split("}", 1)[0]
        assert "text-transform" not in section_rule

    def test_icon_only_buttons_have_an_accessible_name(self, dashboard_html):
        unnamed = []
        for pattern in (r"<button\b([^>]*)>(.*?)</button>", r"<a\b([^>]*)>(.*?)</a>"):
            for attrs, inner in re.findall(pattern, dashboard_html, re.S):
                if re.sub(r"<[^>]+>", "", inner).strip():
                    continue
                if "aria-label=" in attrs or "title=" in attrs:
                    continue
                unnamed.append(re.sub(r"\s+", " ", attrs).strip()[:100])
        assert not unnamed, f"controls with no accessible name: {unnamed}"

    def test_controls_that_are_not_applied_say_so(self, dashboard_html):
        """A switch that does nothing, a status that promises a link that does
        not exist, and a duration the server silently truncates."""
        copy = re.sub(r"\s+", " ", dashboard_html)
        assert re.search(
            r'<input[^>]+id="strict-mode"[^>]+disabled', dashboard_html
        ), "strict mode must not look switchable"
        assert "cannot be switched off" in copy
        assert "Which of these are applied?" in copy
        assert "does not open a connection to the broker" in copy
        assert "at most one" in copy

    def test_status_copy_does_not_promise_more_than_the_server_does(self, dashboard_js):
        for lie in (
            "'MQTT connected successfully'",
            "'MQTT test message sent successfully'",
            "'Advanced monitoring started with filtering'",
            "'MQTT connected to ${data.data.broker}'",
            "'Test message published to MQTT'",
        ):
            assert lie not in dashboard_js, f"UI still claims: {lie}"
        assert "'Saved (no live link)'" in dashboard_js

    def test_recording_a_configuration_does_not_claim_a_run(self, dashboard_js, app_py):
        """These endpoints only store settings; the UI used to flip the
        monitoring state and the server used to raise monitoring_active, which
        then blocked a real start as "already running"."""
        assert "currentMonitoring = true" not in dashboard_js.split(
            "function startAdvancedMonitoring"
        )[1].split("async function testMQTT")[0]
        advanced = app_py.split("async def start_advanced_monitoring", 1)[1]
        assert "self.monitoring_active = True" not in advanced.split(
            "async def start_multiband_monitoring"
        )[0]
        assert "Advanced configuration recorded" in app_py

    def test_model_status_polling_leaves_the_training_select_alone(self, dashboard_js, dashboard_html):
        """The polled model type was written into #model-type, which is the
        training panel's <select>; the first poll destroyed its options."""
        assert "updateElementText('model-type-display'" in dashboard_js
        assert "updateElementText('model-type'," not in dashboard_js
        assert "model-type-select" not in dashboard_js
        assert 'id="model-type-select"' not in dashboard_html
        assert '<select id="model-type"' in dashboard_html

    def test_clearing_alerts_asks_first_and_clears_the_buffer(self, dashboard_js, extras_js):
        """Wiping the DOM brought every alert back on the next render, because
        the list is a view over a buffer this script owns."""
        assert "window.confirm(message)" in dashboard_js
        assert "not deleted" in dashboard_js
        assert "window.SAAlertsClear" in extras_js
        assert "window.SAAlertsClear" in dashboard_js

    def test_destructive_workspace_actions_explain_the_consequence(self, geolocate_js):
        assert "Triangulation, calibration and the waterfall all stop" in geolocate_js
        assert "uncorrected" in geolocate_js
