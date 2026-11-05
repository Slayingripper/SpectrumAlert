"""
Tests for JavaScript frontend functionality validation
These tests verify that the JavaScript functions are properly structured
and would work correctly when integrated with the web interface.
"""

import pytest
import re
from pathlib import Path


class TestJavaScriptFrontend:
    """Test suite for JavaScript frontend functionality"""

    @pytest.fixture
    def dashboard_js_path(self):
        """Path to dashboard.js file"""
        return Path(__file__).parent.parent.parent / "spectrum_alert" / "web" / "static" / "js" / "dashboard.js"

    @pytest.fixture
    def dashboard_js_content(self, dashboard_js_path):
        """Load dashboard.js content"""
        if dashboard_js_path.exists():
            return dashboard_js_path.read_text()
        return ""

    def test_dashboard_js_exists(self, dashboard_js_path):
        """Test that dashboard.js file exists"""
        assert dashboard_js_path.exists(), "dashboard.js file should exist"

    def test_required_functions_present(self, dashboard_js_content):
        """Test that all required JavaScript functions are present"""
        required_functions = [
            'startMonitoring',
            'stopMonitoring',
            'updateSystemStatus',
            'loadRecentAnomalies',
            'loadSpectrumAnalysis',
            'loadDataStats',
            # The function is named initializeWebSocket: it creates the socket
            # and wires the handlers. The test previously asked for
            # connectWebSocket, which never existed.
            'initializeWebSocket',
            'startAdvancedMonitoring',
            'startMultiBandMonitoring',
            'connectMQTT',
            'disconnectMQTT',
            'testMQTT',
            'trainModel',
            'testModel',
            'deployModel',
            'collectData'
        ]
        
        for function_name in required_functions:
            pattern = rf'(function\s+{function_name}|{function_name}\s*[=:]\s*function|\b{function_name}\s*\([^)]*\)\s*{{)'
            assert re.search(pattern, dashboard_js_content), f"Function '{function_name}' should be present in dashboard.js"

    def test_api_endpoints_referenced(self, dashboard_js_content):
        """Test that API endpoints are properly referenced in JavaScript"""
        api_endpoints = [
            '/api/monitoring/start',
            '/api/monitoring/stop',
            '/api/monitoring/status',
            '/api/monitoring/multiband',
            '/api/monitoring/advanced',
            # The dashboard reads /api/system/advanced-status, not
            # /api/system/status. Both endpoints exist on the server; the
            # frontend only calls the richer one, and the old assertion for the
            # unreferenced endpoint could never pass.
            '/api/system/advanced-status',
            '/api/anomalies/recent',
            '/api/spectrum/analysis',
            '/api/data/stats',
            '/api/data/collect',
            '/api/model/train',
            '/api/model/test',
            '/api/model/deploy',
            '/api/mqtt/connect',
            '/api/mqtt/disconnect',
            '/api/mqtt/test',
            '/api/mqtt/status'
        ]
        
        for endpoint in api_endpoints:
            assert endpoint in dashboard_js_content, f"API endpoint '{endpoint}' should be referenced in JavaScript"

    def test_websocket_connection_code(self, dashboard_js_content):
        """Test that WebSocket connection code is present"""
        websocket_patterns = [
            r'new\s+WebSocket',
            # The URL is derived from window.location.protocol and assembled as
            # `${wsProtocol}//${host}/ws`, so the literal strings "ws://" and
            # "wss://" never appear. Assert the derivation instead.
            r'wsProtocol.*wss:.*ws:',
            r'\.onmessage\s*=',
            r'\.onopen\s*=',
            r'\.onclose\s*=',
            r'\.onerror\s*='
        ]
        
        for pattern in websocket_patterns:
            assert re.search(pattern, dashboard_js_content), f"WebSocket pattern '{pattern}' should be present"

    def test_error_handling_present(self, dashboard_js_content):
        """Test that error handling is implemented"""
        error_patterns = [
            r'catch\s*\(',
            r'\.catch\(',
            r'try\s*{',
            r'throw\s+',
            r'console\.error',
            r'showError|displayError|handleError'
        ]
        
        error_count = sum(1 for pattern in error_patterns if re.search(pattern, dashboard_js_content))
        assert error_count >= 3, "JavaScript should have adequate error handling"

    def test_async_functions_present(self, dashboard_js_content):
        """Test that async functions are properly implemented"""
        async_patterns = [
            r'async\s+function',
            r'await\s+',
            r'\.then\(',
            r'Promise\.'
        ]
        
        async_count = sum(1 for pattern in async_patterns if re.search(pattern, dashboard_js_content))
        assert async_count >= 2, "JavaScript should use async patterns for API calls"

    def test_dom_manipulation_functions(self, dashboard_js_content):
        """Test that DOM manipulation functions are present"""
        dom_patterns = [
            r'document\.getElementById',
            r'document\.querySelector',
            r'\.innerHTML\s*=',
            r'\.textContent\s*=',
            r'\.style\.',
            r'\.classList\.',
            r'\.addEventListener'
        ]
        
        dom_count = sum(1 for pattern in dom_patterns if re.search(pattern, dashboard_js_content))
        assert dom_count >= 5, "JavaScript should have adequate DOM manipulation"

    def test_notification_system(self, dashboard_js_content):
        """Test that notification system is implemented"""
        notification_patterns = [
            r'showNotification|displayNotification',
            r'toast|alert',
            r'notification',
            r'success.*message|error.*message'
        ]
        
        notification_count = sum(1 for pattern in notification_patterns if re.search(pattern, dashboard_js_content, re.IGNORECASE))
        assert notification_count >= 1, "JavaScript should have notification system"

    def test_configuration_handling(self, dashboard_js_content):
        """Test that configuration handling is present"""
        config_patterns = [
            r'getConfig|saveConfig|loadConfig',
            r'localStorage',
            r'sessionStorage',
            r'JSON\.parse|JSON\.stringify'
        ]
        
        config_count = sum(1 for pattern in config_patterns if re.search(pattern, dashboard_js_content))
        assert config_count >= 2, "JavaScript should handle configuration persistence"

    def test_real_time_updates(self, dashboard_js_content):
        """Test that real-time update functionality is present"""
        realtime_patterns = [
            r'setInterval|setTimeout',
            r'WebSocket',
            r'updateStatus|refreshData',
            r'autoUpdate|periodicUpdate'
        ]
        
        realtime_count = sum(1 for pattern in realtime_patterns if re.search(pattern, dashboard_js_content))
        assert realtime_count >= 2, "JavaScript should support real-time updates"


class TestCSSFrontend:
    """Test suite for CSS frontend styling"""

    @pytest.fixture
    def css_path(self):
        """Path to CSS file"""
        return Path(__file__).parent.parent.parent / "spectrum_alert" / "web" / "static" / "css" / "cyberpunk.css"

    @pytest.fixture
    def css_content(self, css_path):
        """Load CSS content"""
        if css_path.exists():
            return css_path.read_text()
        return ""

    def test_css_file_exists(self, css_path):
        """Test that CSS file exists"""
        assert css_path.exists(), "cyberpunk.css file should exist"

    def test_responsive_design_classes(self, css_content):
        """Test that responsive design classes are present"""
        responsive_patterns = [
            r'@media\s*\(',
            r'grid|flex',
            r'responsive|mobile',
            r'max-width|min-width'
        ]
        
        responsive_count = sum(1 for pattern in responsive_patterns if re.search(pattern, css_content, re.IGNORECASE))
        assert responsive_count >= 2, "CSS should include responsive design elements"

    def test_cyberpunk_theme_elements(self, css_content):
        """Test that cyberpunk theme elements are present"""
        cyberpunk_patterns = [
            r'#[0-9a-fA-F]{3,6}',  # Hex colors
            r'neon|glow|shadow',
            r'gradient',
            r'transform',
            r'transition'
        ]
        
        cyberpunk_count = sum(1 for pattern in cyberpunk_patterns if re.search(pattern, css_content, re.IGNORECASE))
        assert cyberpunk_count >= 3, "CSS should have cyberpunk theme elements"

    def test_component_styles_present(self, css_content):
        """Test that component-specific styles are present"""
        component_patterns = [
            r'\.button|\.btn',
            r'\.card|\.panel',
            r'\.tab|\.nav',
            r'\.alert|\.notification',
            r'\.modal|\.dialog'
        ]
        
        component_count = sum(1 for pattern in component_patterns if re.search(pattern, css_content))
        assert component_count >= 3, "CSS should include component styles"


class TestHTMLTemplate:
    """Test suite for HTML template"""

    @pytest.fixture
    def template_path(self):
        """Path to main template file"""
        return Path(__file__).parent.parent.parent / "spectrum_alert" / "web" / "templates" / "dashboard.html"

    @pytest.fixture
    def template_content(self, template_path):
        """Load template content"""
        if template_path.exists():
            return template_path.read_text()
        return ""

    def test_template_file_exists(self, template_path):
        """Test that template file exists"""
        assert template_path.exists(), "dashboard.html template should exist"

    def test_required_html_elements(self, template_content):
        """Test that required HTML elements are present"""
        required_elements = [
            r'<html',
            r'<head',
            r'<body',
            r'<div',
            r'<script',
            r'<style|<link.*css'
        ]
        
        for pattern in required_elements:
            assert re.search(pattern, template_content, re.IGNORECASE), f"HTML element pattern '{pattern}' should be present"

    def test_form_elements_present(self, template_content):
        """Test that form elements for configuration are present"""
        form_patterns = [
            r'<form|<input|<select|<textarea|<button',
            r'type=["\'](?:text|number|range|checkbox|radio|submit)["\']',
            r'id=["\'][^"\']+["\']',
            r'class=["\'][^"\']+["\']'
        ]
        
        form_count = sum(1 for pattern in form_patterns if re.search(pattern, template_content, re.IGNORECASE))
        assert form_count >= 3, "HTML should include form elements for user interaction"

    def test_semantic_html_structure(self, template_content):
        """Test that semantic HTML structure is used"""
        semantic_patterns = [
            r'<header|<nav|<main|<section|<article|<aside|<footer',
            r'<h[1-6]',
            r'role=["\'][^"\']+["\']',
            r'aria-[a-z]+'
        ]
        
        semantic_count = sum(1 for pattern in semantic_patterns if re.search(pattern, template_content, re.IGNORECASE))
        assert semantic_count >= 2, "HTML should use semantic structure"

    def test_javascript_integration(self, template_content):
        """Test that JavaScript is properly integrated"""
        js_patterns = [
            r'<script.*src.*dashboard\.js',
            r'<script.*src.*websocket\.js',
            r'defer|async',
            r'DOMContentLoaded'
        ]
        
        js_count = sum(1 for pattern in js_patterns if re.search(pattern, template_content, re.IGNORECASE))
        assert js_count >= 1, "HTML should properly integrate JavaScript files"

    def test_css_integration(self, template_content):
        """Test that CSS is properly integrated"""
        css_patterns = [
            r'<link.*rel=["\']stylesheet["\']',
            r'href=["\'][^"\']*\.css["\']',
            r'<style'
        ]
        
        css_count = sum(1 for pattern in css_patterns if re.search(pattern, template_content, re.IGNORECASE))
        assert css_count >= 1, "HTML should properly integrate CSS files"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
