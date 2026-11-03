"""
Command Line Interface for SpectrumAlert v3.0
"""

import typer
import logging
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from typing import Optional
from spectrum_alert.core.domain.models import DetectionMode
import time

# Create the main app
app = typer.Typer(
    name="spectrum-alert",
    help="Advanced RF Spectrum Monitoring and Anomaly Detection System v1.1",
    add_completion=False
)

# Rich console for better output
console = Console()

# Sub-commands
monitor_app = typer.Typer(help="Spectrum monitoring commands")
train_app = typer.Typer(help="Model training commands")
config_app = typer.Typer(help="Configuration commands")
system_app = typer.Typer(help="System management commands")
geo_app = typer.Typer(help="Multi-receiver geolocation and calibration")

app.add_typer(monitor_app, name="monitor")
app.add_typer(train_app, name="train")
app.add_typer(config_app, name="config")
app.add_typer(system_app, name="system")
app.add_typer(geo_app, name="geo")


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path")
):
    """SpectrumAlert v1.1 - Advanced RF Spectrum Monitoring"""
    setup_logging(verbose)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


@monitor_app.command("start")
def start_monitoring(
    frequency_start: float = typer.Option(..., "--freq-start", "-fs", help="Start frequency in MHz"),
    frequency_end: float = typer.Option(..., "--freq-end", "-fe", help="End frequency in MHz"),
    mode: DetectionMode = typer.Option(DetectionMode.FULL, "--mode", "-m", help="Detection mode"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Duration in seconds"),
    session_name: Optional[str] = typer.Option(None, "--name", "-n", help="Session name"),
    sample_rate: float = typer.Option(1.024, "--sample-rate", "-sr", help="Sample rate in MHz"),
    gain: float = typer.Option(20.0, "--gain", "-g", help="Gain in dB"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Anomaly threshold (0..1)"),
    strict_threshold: bool = typer.Option(True, "--strict-threshold", help="Enable strict threshold mode for better filtering"),
    dc_exclude_hz: float = typer.Option(5000.0, "--dc-exclude-hz", help="Ignore +/- this Hz around center to avoid LO/DC spur"),
    edge_exclude_hz: float = typer.Option(10000.0, "--edge-exclude-hz", help="Ignore +/- this Hz near passband edges to avoid alias/rolloff"),
    novelty_enabled: bool = typer.Option(True, "--novelty", help="Enable persistence/novelty filter"),
    novelty_freq_tol_hz: float = typer.Option(5000.0, "--novelty-freq-tol-hz", help="Frequency tolerance for same-signal grouping"),
    novelty_cooldown_s: float = typer.Option(3.0, "--novelty-cooldown-s", help="Suppress re-alerts within this time window"),
    novelty_score_delta: float = typer.Option(0.15, "--novelty-score-delta", help="Minimum score rise to re-alert within cooldown"),
    mqtt_broker: Optional[str] = typer.Option(None, "--mqtt-broker", help="MQTT broker host (enable live publishing)")
):
    """Start spectrum monitoring"""
    console.print(f"[bold green]Starting spectrum monitoring[/bold green]")
    console.print(f"Frequency range: {frequency_start} - {frequency_end} MHz")
    console.print(f"Detection mode: {mode.value}")
    console.print(f"SDR: {sample_rate} MSps, gain {gain} dB, threshold {threshold}")
    
    # Convert MHz to Hz
    freq_start_hz = frequency_start * 1e6
    freq_end_hz = frequency_end * 1e6
    sample_rate_hz = sample_rate * 1e6
    # Use the center of the range for capture
    capture_center_hz = (freq_start_hz + freq_end_hz) / 2.0
    
    try:
        from spectrum_alert.infrastructure.sdr import RTLSDRInterface
        from spectrum_alert.infrastructure.storage import DataStorage
        from spectrum_alert.core.services.feature_extraction import FeatureExtractor
        from spectrum_alert.application.use_cases.spectrum_monitoring import SpectrumMonitoringUseCase
        try:
            from ..application.use_cases.anomaly_detection import AnomalyDetectionUseCase  # type: ignore
        except Exception:
            from spectrum_alert.application.use_cases.anomaly_detection import AnomalyDetectionUseCase  # type: ignore
        from spectrum_alert.infrastructure.messaging import MQTTManager
        from spectrum_alert.config.manager import ConfigurationManager
        
        sdr = RTLSDRInterface()
        storage = DataStorage()
        feature_extractor = FeatureExtractor(lite_mode=(mode == DetectionMode.LITE))
        monitoring_use_case = SpectrumMonitoringUseCase(sdr, storage, feature_extractor)
        anomaly_use_case = AnomalyDetectionUseCase(feature_extractor)
        anomaly_use_case.set_threshold(threshold)
        anomaly_use_case.set_strict_threshold_mode(strict_threshold)
        try:
            anomaly_use_case.set_dc_exclude_hz(dc_exclude_hz)
            anomaly_use_case.set_edge_exclude_hz(edge_exclude_hz)
            anomaly_use_case.configure_novelty(
                enabled=novelty_enabled,
                freq_tol_hz=novelty_freq_tol_hz,
                cooldown_s=novelty_cooldown_s,
                score_delta=novelty_score_delta,
            )
            anomaly_use_case.set_allowed_frequency_range(freq_start_hz, freq_end_hz)
            # Stronger gating for start to reduce false positives
            anomaly_use_case.configure_gating(min_snr_db=10.0, min_prominence_db=8.0, local_window_bins=5, neighbor_support_db=4.0)
        except Exception:
            pass
        
        # MQTT from config (CLI --mqtt-broker overrides host)
        mqtt_mgr = None
        try:
            cfg = ConfigurationManager()
            mqtt_enabled = cfg.get_setting('mqtt.enabled')
            if mqtt_broker or mqtt_enabled:
                broker_host = mqtt_broker or cfg.get_setting('mqtt.broker_host')
                mqtt_mgr = MQTTManager(
                    broker=broker_host,
                    port=cfg.get_setting('mqtt.broker_port'),
                    username=cfg.get_setting('mqtt.username'),
                    password=cfg.get_setting('mqtt.password'),
                    client_id=cfg.get_setting('mqtt.client_id'),
                    topic_prefix=cfg.get_setting('mqtt.topic_prefix'),
                    qos=cfg.get_setting('mqtt.qos'),
                    keepalive_seconds=cfg.get_setting('mqtt.keepalive_seconds'),
                    tls_enabled=cfg.get_setting('mqtt.tls_enabled'),
                )
                try:
                    if mqtt_mgr.connect():
                        console.print(f"[green]MQTT connected: {broker_host}:{cfg.get_setting('mqtt.broker_port')} (prefix: {cfg.get_setting('mqtt.topic_prefix')})[/green]")
                    else:
                        console.print("[yellow]MQTT connect timeout[/yellow]")
                        mqtt_mgr = None
                except Exception as me:
                    console.print(f"[yellow]MQTT connect failed: {me}[/yellow]")
                    mqtt_mgr = None
        except Exception as _cfg_err:
            pass
        
        session_name = session_name or f"monitoring_{int(freq_start_hz/1e6)}_{int(freq_end_hz/1e6)}"
        session = monitoring_use_case.start_monitoring_session(
            session_name=session_name,
            frequency_range=(freq_start_hz, freq_end_hz),
            detection_mode=mode
        )
        
        console.print(f"[green]Session started: {session.id}[/green]")
        
        import time
        with sdr.safe_operation():
            start_time = time.time()
            while True:
                try:
                    spectrum_data = monitoring_use_case.capture_spectrum_data(
                        frequency=capture_center_hz,
                        sample_rate=sample_rate_hz,
                        gain=gain,
                        duration=0.25
                    )
                    monitoring_use_case.extract_features(spectrum_data)
                    anomalies = anomaly_use_case.detect_anomalies(spectrum_data)
                    if anomalies:
                        for anomaly in anomalies:
                            try:
                                storage.save_anomaly(anomaly)
                            except Exception as sa_err:
                                console.print(f"[yellow]Failed to save anomaly: {sa_err}[/yellow]")
                            # Publish over MQTT if available
                            if mqtt_mgr:
                                mqtt_mgr.publish_anomaly(
                                    frequency=anomaly.frequency_hz,
                                    score=anomaly.confidence_score,
                                    details={
                                        "severity": anomaly.severity,
                                        "description": anomaly.description,
                                        "mode": anomaly.detection_mode.value,
                                    }
                                )
                        # Pinpoint exact peak frequency from anomaly
                        peaks = ", ".join(f"{a.frequency_hz/1e6:.6f} MHz" for a in anomalies)
                        console.print(f"[red]🚨 Detected {len(anomalies)} anomaly(ies) at: {peaks}[/red]")
                    
                    console.print(f"[dim]Captured data at {capture_center_hz/1e6:.6f} MHz[/dim]")
                    
                    if duration and (time.time() - start_time) > duration:
                        break
                    time.sleep(0.75)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Monitoring interrupted by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break
        
        monitoring_use_case.stop_monitoring_session()
        if mqtt_mgr:
            mqtt_mgr.disconnect()
        console.print("[green]Monitoring session completed[/green]")
    
    except Exception as e:
        console.print(f"[red]Failed to start monitoring: {e}[/red]")
        raise typer.Exit(1)

# Quick live verification command
@monitor_app.command("verify")
def verify_realtime_anomalies(
    frequency_mhz: float = typer.Option(100.0, "--freq", help="Center frequency MHz"),
    seconds: int = typer.Option(15, "--seconds", help="Run time in seconds"),
    sample_rate: float = typer.Option(1.024, "--sample-rate", "-sr", help="Sample rate in MHz"),
    gain: float = typer.Option(20.0, "--gain", "-g", help="Gain in dB"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Anomaly threshold (0..1)"),
    strict_threshold: bool = typer.Option(True, "--strict-threshold", help="Enable strict threshold mode for better filtering"),
    dc_exclude_hz: float = typer.Option(5000.0, "--dc-exclude-hz", help="Ignore +/- this Hz around center to avoid LO/DC spur"),
    edge_exclude_hz: float = typer.Option(10000.0, "--edge-exclude-hz", help="Ignore +/- this Hz near passband edges"),
    novelty_enabled: bool = typer.Option(True, "--novelty", help="Enable persistence/novelty filter"),
    novelty_freq_tol_hz: float = typer.Option(5000.0, "--novelty-freq-tol-hz", help="Frequency tolerance for same-signal grouping"),
    novelty_cooldown_s: float = typer.Option(3.0, "--novelty-cooldown-s", help="Suppress re-alerts within this time window"),
    novelty_score_delta: float = typer.Option(0.15, "--novelty-score-delta", help="Minimum score rise to re-alert within cooldown"),
    mqtt_broker: Optional[str] = typer.Option(None, "--mqtt-broker", help="MQTT broker host")
):
    """Run a short live verification with real-time anomaly detection and optional MQTT."""
    # Provide a valid range centered around the requested frequency using half the sample rate span
    half_span_mhz = max(sample_rate / 2.0, 0.001)
    start_mhz = frequency_mhz - half_span_mhz
    end_mhz = frequency_mhz + half_span_mhz
    if end_mhz <= start_mhz:
        end_mhz = start_mhz + 0.001
    return start_monitoring(
        frequency_start=start_mhz,
        frequency_end=end_mhz,
        mode=DetectionMode.LITE,
        duration=seconds,
        session_name=f"verify_{int(frequency_mhz)}",
        sample_rate=sample_rate,
        gain=gain,
        threshold=threshold,
        strict_threshold=strict_threshold,
        dc_exclude_hz=dc_exclude_hz,
        edge_exclude_hz=edge_exclude_hz,
        novelty_enabled=novelty_enabled,
        novelty_freq_tol_hz=novelty_freq_tol_hz,
        novelty_cooldown_s=novelty_cooldown_s,
        novelty_score_delta=novelty_score_delta,
        mqtt_broker=mqtt_broker,
    )


@monitor_app.command("status")
def monitor_status():
    """Show monitoring status"""
    console.print("[bold blue]Monitoring Status[/bold blue]")
    
    # Create status table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")
    
    try:
        # Check SDR availability. Prefer the direct hardware probe: it tells
        # "no dongle attached" apart from "driver/library broken", which the
        # pyrtlsdr error string alone cannot.
        from spectrum_alert.infrastructure.sdr.scanner import probe_local_sdr
        probe = probe_local_sdr(force=True)
        if probe["available"]:
            names = ", ".join(d["name"] for d in probe["devices"])
            table.add_row("SDR Device", "[green]Available[/green]", names or "RTL-SDR detected")
        elif probe["status"] == "no-device":
            table.add_row("SDR Device", "[yellow]No device[/yellow]", probe["detail"])
        elif probe["status"] == "tools-missing":
            table.add_row("SDR Device", "[red]No tools[/red]", probe["detail"])
        else:
            # Library problems (e.g. undefined symbol from a version
            # mismatch) land here with the probe's explanation attached.
            table.add_row("SDR Device", "[red]Error[/red]", probe["detail"])
        
        # Check system resources
        from spectrum_alert.infrastructure.monitoring import SystemMonitor
        monitor = SystemMonitor()
        health = monitor.check_system_health()
        
        status_color = {
            'healthy': 'green',
            'warning': 'yellow', 
            'critical': 'red',
            'error': 'red'
        }.get(health['overall'], 'red')
        
        table.add_row("System Health", f"[{status_color}]{health['overall'].title()}[/{status_color}]", 
                     f"CPU: {health['metrics'].get('cpu', {}).get('percent', 0):.1f}%")
        
        # Check data storage
        from spectrum_alert.infrastructure.storage import DataStorage
        storage = DataStorage()
        anomaly_count = storage.get_anomaly_count(days=1)
        table.add_row("Data Storage", "[green]Ready[/green]", f"{anomaly_count} anomalies today")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")


@monitor_app.command("autonomous")
def autonomous_monitoring(
    frequency_start: float = typer.Option(88.0, "--freq-start", "-fs", help="Start frequency in MHz"),
    frequency_end: float = typer.Option(108.0, "--freq-end", "-fe", help="End frequency in MHz"),
    data_collection_minutes: int = typer.Option(5, "--data-minutes", "-dm", help="Data collection duration in minutes (reduced default)"),
    training_interval_hours: int = typer.Option(1, "--training-hours", "-th", help="Training interval in hours"),
    monitoring_interval_minutes: int = typer.Option(10, "--monitor-minutes", "-mm", help="Monitoring interval in minutes (reduced default)"),
    mode: DetectionMode = typer.Option(DetectionMode.LITE, "--mode", "-m", help="Detection mode"),
    max_cycles: Optional[int] = typer.Option(None, "--max-cycles", "-mc", help="Maximum cycles (None = infinite)"),
    sample_rate: float = typer.Option(2.048, "--sample-rate", "-sr", help="Sample rate in MHz"),
    gain: float = typer.Option(30.0, "--gain", "-g", help="Gain in dB"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Anomaly threshold (0..1)"),
    strict_threshold: bool = typer.Option(True, "--strict-threshold", help="Enable strict threshold mode for better filtering"),
    dc_exclude_hz: float = typer.Option(5000.0, "--dc-exclude-hz", help="Ignore +/- this Hz around center to avoid LO/DC spur"),
    edge_exclude_hz: float = typer.Option(10000.0, "--edge-exclude-hz", help="Ignore +/- this Hz near passband edges to avoid alias/rolloff"),
    novelty_enabled: bool = typer.Option(True, "--novelty", help="Enable persistence/novelty filter"),
    novelty_freq_tol_hz: float = typer.Option(5000.0, "--novelty-freq-tol-hz", help="Frequency tolerance for same-signal grouping"),
    novelty_cooldown_s: float = typer.Option(3.0, "--novelty-cooldown-s", help="Suppress re-alerts within this time window"),
    novelty_score_delta: float = typer.Option(0.15, "--novelty-score-delta", help="Minimum score rise to re-alert within cooldown"),
    mqtt_broker: Optional[str] = typer.Option(None, "--mqtt-broker", help="MQTT broker host (overrides config)"),
    mqtt_port: int = typer.Option(1883, "--mqtt-port", help="MQTT broker port"),
    mqtt_topic_prefix: str = typer.Option("spectrum_alert", "--mqtt-topic-prefix", help="MQTT topic prefix"),
    mqtt_username: Optional[str] = typer.Option(None, "--mqtt-username", help="MQTT username"),
    mqtt_password: Optional[str] = typer.Option(None, "--mqtt-password", help="MQTT password"),
    mqtt_tls: bool = typer.Option(False, "--mqtt-tls", help="Enable TLS for MQTT"),
    continuous_learning: bool = typer.Option(False, "--continuous-learning", help="Train every cycle to adapt the model over time"),
):
    """Start autonomous continuous learning and monitoring"""
    console.print("[bold green]🚀 Starting Autonomous Mode[/bold green]")
    console.print(f"📻 Frequency range: {frequency_start} - {frequency_end} MHz")
    console.print(f"📊 Data collection: {data_collection_minutes} minutes per cycle")
    console.print(f"🧠 Training interval: {training_interval_hours} hours")
    console.print(f"📡 Monitoring interval: {monitoring_interval_minutes} minutes per cycle")
    console.print(f"🔧 Mode: {mode.value}, Gain: {gain} dB, Sample Rate: {sample_rate} MHz, Threshold: {threshold}")
    
    if max_cycles:
        console.print(f"🔄 Will run for {max_cycles} cycles")
    else:
        console.print("🔄 Will run continuously (Ctrl+C to stop)")
    
    # Convert MHz to Hz
    freq_start_hz = frequency_start * 1e6
    freq_end_hz = frequency_end * 1e6
    sample_rate_hz = sample_rate * 1e6
    
    try:
        # Import and initialize components
        from spectrum_alert.infrastructure.sdr import RTLSDRInterface
        from spectrum_alert.infrastructure.storage import DataStorage
        from spectrum_alert.core.services.feature_extraction import FeatureExtractor
        from spectrum_alert.application.use_cases.autonomous_learning import AutonomousLearningUseCase, AutonomousCycleConfig
        
        # Initialize components
        sdr = RTLSDRInterface()
        storage = DataStorage()
        feature_extractor = FeatureExtractor(lite_mode=(mode == DetectionMode.LITE))
        
        # Create configuration
        config = AutonomousCycleConfig(
            data_collection_minutes=data_collection_minutes,
            training_interval_hours=training_interval_hours,
            monitoring_interval_minutes=monitoring_interval_minutes,
            frequency_start=freq_start_hz,
            frequency_end=freq_end_hz,
            detection_mode=mode,
            sample_rate=sample_rate_hz,
            gain=gain,
            max_cycles=max_cycles,
            continuous_learning=continuous_learning,
        )
        
        # Status callback for real-time updates
        def status_callback(message: str, level: str = "info"):
            if level == "info":
                console.print(f"[blue]ℹ️  {message}[/blue]")
            elif level == "warning":
                console.print(f"[yellow]⚠️  {message}[/yellow]")
            elif level == "error":
                console.print(f"[red]❌ {message}[/red]")
        
        # Initialize autonomous learning use case
        autonomous_use_case = AutonomousLearningUseCase(
            sdr=sdr,
            storage=storage,
            feature_extractor=feature_extractor,
            config=config,
            status_callback=status_callback
        )
        # Apply threshold to anomaly detector
        try:
            autonomous_use_case.anomaly_use_case.set_threshold(threshold)
            autonomous_use_case.anomaly_use_case.set_strict_threshold_mode(strict_threshold)
            try:
                autonomous_use_case.anomaly_use_case.set_dc_exclude_hz(dc_exclude_hz)
                autonomous_use_case.anomaly_use_case.set_edge_exclude_hz(edge_exclude_hz)
                autonomous_use_case.anomaly_use_case.set_allowed_frequency_range(freq_start_hz, freq_end_hz)
                autonomous_use_case.anomaly_use_case.configure_gating(min_snr_db=10.0, min_prominence_db=8.0, local_window_bins=5, neighbor_support_db=4.0)
                autonomous_use_case.anomaly_use_case.configure_novelty(
                    enabled=novelty_enabled,
                    freq_tol_hz=novelty_freq_tol_hz,
                    cooldown_s=novelty_cooldown_s,
                    score_delta=novelty_score_delta,
                )
            except Exception:
                pass
        except Exception as _:
            pass
        
        # MQTT override from CLI (forces MQTT even if config disabled)
        if mqtt_broker:
            try:
                from spectrum_alert.infrastructure.messaging import MQTTManager
                mqtt_mgr = MQTTManager(
                    broker=mqtt_broker,
                    port=mqtt_port,
                    username=mqtt_username,
                    password=mqtt_password,
                    client_id=f"spectrum_alert_{int(time.time())}",
                    topic_prefix=mqtt_topic_prefix,
                    qos=1,
                    keepalive_seconds=60,
                    tls_enabled=mqtt_tls,
                )
                if mqtt_mgr.connect():
                    console.print(f"[green]MQTT connected: {mqtt_broker}:{mqtt_port} (prefix: {mqtt_topic_prefix})[/green]")
                    autonomous_use_case.mqtt_mgr = mqtt_mgr
                else:
                    console.print("[yellow]MQTT connect timeout[/yellow]")
            except Exception as me:
                console.print(f"[yellow]MQTT connect failed: {me}[/yellow]")
        
        console.print("\n[bold yellow]Starting autonomous learning cycle...[/bold yellow]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        # Start autonomous mode
        autonomous_use_case.start_autonomous_mode()
        
        # Show final statistics
        stats = autonomous_use_case.get_statistics()
        console.print("\n[bold green]📊 Autonomous Mode Statistics:[/bold green]")
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        
        stats_table.add_row("Cycles Completed", str(stats["cycles_completed"]))
        stats_table.add_row("Total Data Collected", str(stats["total_data_collected"]))
        stats_table.add_row("Last Training", stats["last_training"] or "Never")
        stats_table.add_row("Total Runtime", stats["runtime"] or "Unknown")
        stats_table.add_row("Frequency Range", stats["config"]["frequency_range_mhz"] + " MHz")
        
        console.print(stats_table)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️  Autonomous mode stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Autonomous mode failed: {e}[/red]")
        raise typer.Exit(1)


@monitor_app.command("autonomous-multiband")
def autonomous_multiband(
    bands: list[str] = typer.Option(..., "--band", "-b", help="Band range in MHz as START-END (e.g., 144-148). Repeat for multiple."),
    data_collection_minutes: int = typer.Option(5, "--data-minutes", "-dm", help="Data collection duration per band (minutes)"),
    training_interval_hours: int = typer.Option(1, "--training-hours", "-th", help="Training interval (hours)"),
    monitoring_interval_minutes: int = typer.Option(10, "--monitor-minutes", "-mm", help="Monitoring duration per band (minutes)"),
    mode: DetectionMode = typer.Option(DetectionMode.LITE, "--mode", "-m", help="Detection mode"),
    sample_rate: float = typer.Option(1.024, "--sample-rate", "-sr", help="Sample rate in MHz"),
    gain: float = typer.Option(25.0, "--gain", "-g", help="Gain in dB"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Anomaly threshold (0..1)"),
    strict_threshold: bool = typer.Option(True, "--strict-threshold", help="Enable strict threshold mode for better filtering"),
    dc_exclude_hz: float = typer.Option(8000.0, "--dc-exclude-hz", help="Ignore +/- this Hz around center to avoid LO/DC spur"),
    edge_exclude_hz: float = typer.Option(20000.0, "--edge-exclude-hz", help="Ignore +/- this Hz near passband edges"),
    novelty_enabled: bool = typer.Option(True, "--novelty", help="Enable persistence/novelty filter"),
    novelty_freq_tol_hz: float = typer.Option(5000.0, "--novelty-freq-tol-hz", help="Frequency tolerance for same-signal grouping"),
    novelty_cooldown_s: float = typer.Option(3.0, "--novelty-cooldown-s", help="Suppress re-alerts within this time window"),
    novelty_score_delta: float = typer.Option(0.15, "--novelty-score-delta", help="Minimum score rise to re-alert within cooldown"),
    mqtt_broker: Optional[str] = typer.Option(None, "--mqtt-broker", help="MQTT broker host (overrides config)"),
    mqtt_port: int = typer.Option(1883, "--mqtt-port", help="MQTT broker port"),
    mqtt_topic_prefix: str = typer.Option("spectrum_alert", "--mqtt-topic-prefix", help="MQTT topic prefix"),
    mqtt_username: Optional[str] = typer.Option(None, "--mqtt-username", help="MQTT username"),
    mqtt_password: Optional[str] = typer.Option(None, "--mqtt-password", help="MQTT password"),
    mqtt_tls: bool = typer.Option(False, "--mqtt-tls", help="Enable TLS for MQTT"),
    max_cycles_per_band: int = typer.Option(1, "--max-cycles-per-band", help="Number of autonomous cycles to run per band before switching"),
    rounds: Optional[int] = typer.Option(None, "--rounds", help="Number of rotations across all bands (None=infinite)"),
    continuous_learning: bool = typer.Option(False, "--continuous-learning", help="Train every cycle to adapt the model over time"),
):
    """Rotate autonomous training/monitoring across multiple bands (e.g., 2m and 70cm)."""
    # Helper to parse band strings like "144-148"
    def _parse_band(b: str) -> tuple[float, float]:
        sep = "-" if "-" in b else (":" if ":" in b else None)
        if not sep:
            raise typer.BadParameter(f"Invalid band format '{b}'. Use START-END in MHz, e.g., 144-148")
        parts = b.split(sep)
        if len(parts) != 2:
            raise typer.BadParameter(f"Invalid band format '{b}'. Use START-END in MHz, e.g., 144-148")
        try:
            a, c = float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            raise typer.BadParameter(f"Invalid numbers in band '{b}'")
        if c <= a:
            raise typer.BadParameter(f"Band end must be > start in '{b}'")
        return a, c

    band_ranges_mhz: list[tuple[float, float]] = []
    for b in bands:
        band_ranges_mhz.append(_parse_band(b))

    console.print("[bold green]🚀 Starting Autonomous Multi-Band Mode[/bold green]")
    console.print("Bands:")
    for a, c in band_ranges_mhz:
        console.print(f" • {a}-{c} MHz")

    # Rotate through bands
    rot = 0
    try:
        while rounds is None or rot < rounds:
            rot += 1
            console.print(f"\n[bold yellow]🔁 Rotation {rot}{'' if rounds is None else f'/{rounds}'}[/bold yellow]")
            for a, c in band_ranges_mhz:
                console.print(f"\n[bold cyan]▶ Band {a}-{c} MHz[/bold cyan]")
                try:
                    autonomous_monitoring(
                        frequency_start=a,
                        frequency_end=c,
                        data_collection_minutes=data_collection_minutes,
                        training_interval_hours=training_interval_hours,
                        monitoring_interval_minutes=monitoring_interval_minutes,
                        mode=mode,
                        max_cycles=max_cycles_per_band,
                        sample_rate=sample_rate,
                        gain=gain,
                        threshold=threshold,
                        strict_threshold=strict_threshold,
                        dc_exclude_hz=dc_exclude_hz,
                        edge_exclude_hz=edge_exclude_hz,
                        novelty_enabled=novelty_enabled,
                        novelty_freq_tol_hz=novelty_freq_tol_hz,
                        novelty_cooldown_s=novelty_cooldown_s,
                        novelty_score_delta=novelty_score_delta,
                        mqtt_broker=mqtt_broker,
                        mqtt_port=mqtt_port,
                        mqtt_topic_prefix=mqtt_topic_prefix,
                        mqtt_username=mqtt_username,
                        mqtt_password=mqtt_password,
                        mqtt_tls=mqtt_tls,
                        continuous_learning=continuous_learning,
                    )
                except typer.Exit:
                    # Respect inner command exit and continue to next band
                    continue
                except KeyboardInterrupt:
                    raise
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️  Multi-band mode stopped by user[/yellow]")


@train_app.command("models")
def train_models(
    mode: DetectionMode = typer.Option(DetectionMode.FULL, "--mode", "-m", help="Detection mode"),
    days: int = typer.Option(7, "--days", "-d", help="Days of training data to use")
):
    """Train machine learning models"""
    console.print(f"[bold green]Training models for {mode.value} mode[/bold green]")
    console.print(f"Using {days} days of training data")
    
    try:
        from spectrum_alert.infrastructure.storage import DataStorage
        
        storage = DataStorage()
        training_data = storage.load_training_data(mode=mode.value, days=days)
        
        if training_data is None or len(training_data) == 0:
            console.print("[red]No training data available[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]Loaded {len(training_data)} training samples[/green]")
        console.print("[yellow]Model training would be implemented here[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)


@system_app.command("status")
def system_status():
    """Show detailed system status"""
    try:
        from spectrum_alert.infrastructure.monitoring import SystemMonitor
        
        monitor = SystemMonitor()
        metrics = monitor.get_system_metrics()
        
        console.print("[bold blue]System Status[/bold blue]")
        
        # System metrics table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Resource")
        table.add_column("Usage")
        table.add_column("Available")
        table.add_column("Total")
        
        # CPU
        table.add_row(
            "CPU",
            f"{metrics['cpu']['percent']:.1f}%",
            f"{metrics['cpu']['count']} cores",
            f"{metrics['cpu']['count']} cores"
        )
        
        # Memory
        table.add_row(
            "Memory",
            f"{metrics['memory']['percent']:.1f}%",
            f"{metrics['memory']['available_gb']:.1f} GB",
            f"{metrics['memory']['total_gb']:.1f} GB"
        )
        
        # Disk
        table.add_row(
            "Disk",
            f"{metrics['disk']['percent']:.1f}%", 
            f"{metrics['disk']['free_gb']:.1f} GB",
            f"{metrics['disk']['total_gb']:.1f} GB"
        )
        
        console.print(table)
        
        # Uptime
        uptime = monitor.get_uptime_string()
        console.print(f"\n[bold]Uptime:[/bold] {uptime}")
        
    except Exception as e:
        console.print(f"[red]Error getting system status: {e}[/red]")


@app.command("version")
def version():
    """Show version information"""
    console.print("[bold green]SpectrumAlert v1.1[/bold green]")
    console.print("Advanced RF Spectrum Monitoring and Anomaly Detection System")


@monitor_app.command("live")
def live_monitor(
    frequency_mhz: float = typer.Option(..., "--freq", help="Center frequency MHz"),
    seconds: int = typer.Option(60, "--seconds", help="Run time in seconds"),
    sample_rate: float = typer.Option(1.024, "--sample-rate", "-sr", help="Sample rate in MHz"),
    gain: float = typer.Option(20.0, "--gain", "-g", help="Gain in dB"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Anomaly threshold (0..1)"),
    window_ms: float = typer.Option(20.0, "--window-ms", help="Analysis window in milliseconds (smaller = lower latency)"),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce console output"),
    dc_exclude_hz: float = typer.Option(5000.0, "--dc-exclude-hz", help="Ignore +/- this Hz around center to avoid LO/DC spur"),
    edge_exclude_hz: float = typer.Option(10000.0, "--edge-exclude-hz", help="Ignore +/- this Hz near passband edges to avoid alias/rolloff"),
    mqtt_broker: Optional[str] = typer.Option(None, "--mqtt-broker", help="MQTT broker host (overrides config)"),
    mqtt_port: int = typer.Option(1883, "--mqtt-port", help="MQTT broker port"),
    mqtt_topic_prefix: str = typer.Option("spectrum_alert", "--mqtt-topic-prefix", help="MQTT topic prefix"),
    mqtt_username: Optional[str] = typer.Option(None, "--mqtt-username", help="MQTT username"),
    mqtt_password: Optional[str] = typer.Option(None, "--mqtt-password", help="MQTT password"),
    mqtt_tls: bool = typer.Option(False, "--mqtt-tls", help="Enable TLS for MQTT")
):
    """Ultra low-latency live anomaly monitoring (no disk writes)."""
    console.print("[bold green]Starting LIVE monitoring[/bold green]")
    console.print(f"Center: {frequency_mhz:.6f} MHz | SR: {sample_rate} MSps | Gain: {gain} dB | Threshold: {threshold} | Window: {window_ms} ms")

    # Hz conversions and window size
    center_hz = frequency_mhz * 1e6
    sr_hz = sample_rate * 1e6
    win_seconds = max(window_ms, 2.0) / 1000.0 if sample_rate < 0.25 else window_ms / 1000.0
    num_samples = max(2048, int(sr_hz * win_seconds))

    try:
        from spectrum_alert.infrastructure.sdr import RTLSDRInterface
        # Avoid storage and full monitoring pipeline for speed
        from spectrum_alert.core.domain.models import SpectrumData, DetectionMode
        from spectrum_alert.core.services.feature_extraction import FeatureExtractor
        try:
            from ..application.use_cases.anomaly_detection import AnomalyDetectionUseCase  # type: ignore
        except Exception:
            from spectrum_alert.application.use_cases.anomaly_detection import AnomalyDetectionUseCase  # type: ignore
        from spectrum_alert.infrastructure.messaging import MQTTManager
        from spectrum_alert.config.manager import ConfigurationManager
        import time

        # MQTT via config
        mqtt_mgr = None
        try:
            from spectrum_alert.infrastructure.messaging import MQTTManager
            if mqtt_broker:
                mqtt_mgr = MQTTManager(
                    broker=mqtt_broker,
                    port=mqtt_port,
                    username=mqtt_username,
                    password=mqtt_password,
                    client_id=f"spectrum_alert_live_{int(time.time())}",
                    topic_prefix=mqtt_topic_prefix,
                    qos=1,
                    keepalive_seconds=60,
                    tls_enabled=mqtt_tls,
                )
                if mqtt_mgr.connect():
                    if not quiet:
                        console.print(f"[green]MQTT connected: {mqtt_broker}:{mqtt_port} (prefix: {mqtt_topic_prefix})[/green]")
                else:
                    if not quiet:
                        console.print("[yellow]MQTT connect timeout[/yellow]")
                    mqtt_mgr = None
            else:
                # Fallback to config when no CLI broker is provided
                cfg = ConfigurationManager()
                if cfg.get_setting('mqtt.enabled'):
                    mqtt_mgr = MQTTManager(
                        broker=cfg.get_setting('mqtt.broker_host'),
                        port=cfg.get_setting('mqtt.broker_port'),
                        username=cfg.get_setting('mqtt.username'),
                        password=cfg.get_setting('mqtt.password'),
                        client_id=cfg.get_setting('mqtt.client_id'),
                        topic_prefix=cfg.get_setting('mqtt.topic_prefix'),
                        qos=cfg.get_setting('mqtt.qos'),
                        keepalive_seconds=cfg.get_setting('mqtt.keepalive_seconds'),
                        tls_enabled=cfg.get_setting('mqtt.tls_enabled'),
                    )
                    if mqtt_mgr.connect():
                        if not quiet:
                            console.print(f"[green]MQTT connected: {cfg.get_setting('mqtt.broker_host')}:{cfg.get_setting('mqtt.broker_port')} (prefix: {cfg.get_setting('mqtt.topic_prefix')})[/green]")
                    else:
                        if not quiet:
                            console.print("[yellow]MQTT connect timeout[/yellow]")
                        mqtt_mgr = None
        except Exception:
            pass

        sdr = RTLSDRInterface()
        fe = FeatureExtractor(lite_mode=True)
        detector = AnomalyDetectionUseCase(fe)
        detector.set_threshold(threshold)
        try:
            detector.set_dc_exclude_hz(dc_exclude_hz)
            detector.set_edge_exclude_hz(edge_exclude_hz)
            # Live uses the center as absolute range by SR/2 to clamp outputs
            detector.set_allowed_frequency_range(center_hz - (sr_hz/2.0), center_hz + (sr_hz/2.0))
            detector.configure_gating(min_snr_db=9.0, min_prominence_db=7.0, local_window_bins=5, neighbor_support_db=3.5)
            detector.configure_novelty(enabled=True, freq_tol_hz=max(3000.0, dc_exclude_hz), cooldown_s=1.0, score_delta=0.1)
        except Exception:
            pass

        # Configure once
        with sdr.safe_operation():
            sdr.set_sample_rate(sr_hz)
            sdr.set_center_freq(center_hz)
            sdr.set_gain(gain)

            start = time.time()
            last_alert = 0.0
            alert_cooldown = 0.3  # seconds
            printed_cfg = False

            while (time.time() - start) < seconds:
                try:
                    samples = sdr.read_samples(num_samples)
                    # Minimal SpectrumData just for detector (no disk IO)
                    sd = SpectrumData(
                        frequency_hz=center_hz,
                        sample_rate_hz=sr_hz,
                        gain_db=gain,
                        samples=samples.tolist(),
                        power_spectrum=None,
                        duration_seconds=num_samples / sr_hz,
                    )
                    anomalies = detector.detect_anomalies(sd)
                    now = time.time()
                    if anomalies and (now - last_alert) > alert_cooldown:
                        peaks = ", ".join(f"{a.frequency_hz/1e6:.6f} MHz (score {a.confidence_score:.2f})" for a in anomalies)
                        console.print(f"[red]🚨 Live anomaly: {peaks}[/red]")
                        # Publish via MQTT
                        if mqtt_mgr:
                            for a in anomalies:
                                mqtt_mgr.publish_anomaly(
                                    frequency=a.frequency_hz,
                                    score=a.confidence_score,
                                    details={
                                        "severity": getattr(a, 'severity', 'unknown'),
                                        "description": getattr(a, 'description', ''),
                                        "mode": getattr(a, 'detection_mode', DetectionMode.LITE).value if hasattr(a, 'detection_mode') else DetectionMode.LITE.value,
                                    }
                                )
                        last_alert = now
                    elif not quiet and not printed_cfg:
                        console.print(f"[dim]LIVE running @ {frequency_mhz:.6f} MHz, window {num_samples} samples (~{window_ms:.1f} ms)[/dim]")
                        printed_cfg = True
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    if not quiet:
                        console.print(f"[yellow]Live loop error: {e}[/yellow]")
                    time.sleep(0.01)
                    continue

        if mqtt_mgr:
            mqtt_mgr.disconnect()
        if not quiet:
            console.print("[green]LIVE monitoring completed[/green]")
    except Exception as e:
        console.print(f"[red]Failed to start live monitoring: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def web(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run web interface on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind web interface to")
):
    """Start the web dashboard interface"""
    try:
        import uvicorn
        from spectrum_alert.web.app import SpectrumAlertWebApp
        
        console.print("[bold green]Starting SpectrumAlert Web Dashboard...[/bold green]")
        console.print(f"[cyan]Dashboard will be available at: http://localhost:{port}[/cyan]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        
        # Create web app
        web_app = SpectrumAlertWebApp()
        app_instance = web_app.app
        
        # Run the server
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level="info",
            access_log=False
        )
        
    except ImportError:
        console.print("[bold red]Error: Missing web dependencies![/bold red]")
        console.print("Install with: [cyan]pip install fastapi uvicorn jinja2 python-multipart[/cyan]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Web dashboard stopped[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error starting web dashboard: {e}[/bold red]")
        raise typer.Exit(1)


@geo_app.command("receivers")
def geo_receivers(
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory")
):
    """List configured receivers"""
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    import os

    store = ReceiverStore(os.path.join(data_dir, "receivers.json"))
    receivers = store.list()

    if not receivers:
        console.print("[yellow]No receivers configured.[/yellow]")
        console.print("Add one with: [cyan]spectrum-alert geo add-receiver --help[/cyan]")
        return

    table = Table(title=f"{len(receivers)} receiver(s)")
    table.add_column("Name")
    table.add_column("Mode")
    table.add_column("Latitude")
    table.add_column("Longitude")
    table.add_column("On")
    table.add_column("Tuning")

    for receiver in receivers:
        tuning = []
        if receiver.gain_db is not None:
            tuning.append(f"gain {receiver.gain_db} dB")
        if receiver.ppm_error:
            tuning.append(f"ppm {receiver.ppm_error}")
        if receiver.manual_rssi_db is not None:
            tuning.append(f"manual {receiver.manual_rssi_db:.1f} dBFS")
        table.add_row(
            receiver.name,
            receiver.mode.value,
            f"{receiver.lat:.6f}",
            f"{receiver.lon:.6f}",
            "yes" if receiver.enabled else "no",
            ", ".join(tuning) or "-",
        )
    console.print(table)


@geo_app.command("add-receiver")
def geo_add_receiver(
    name: str = typer.Argument(..., help="Receiver name"),
    lat: float = typer.Option(..., "--lat", help="Latitude in degrees"),
    lon: float = typer.Option(..., "--lon", help="Longitude in degrees"),
    mode: str = typer.Option("local", "--mode", help="local or rtltcp"),
    device_index: int = typer.Option(0, "--device", help="Local dongle index"),
    host: Optional[str] = typer.Option(None, "--host", help="rtl_tcp host"),
    port: int = typer.Option(1234, "--port", help="rtl_tcp port"),
    gain: Optional[float] = typer.Option(None, "--gain", help="Manual gain in dB"),
    ppm: Optional[float] = typer.Option(None, "--ppm", help="Crystal error in ppm"),
    manual_db: Optional[float] = typer.Option(None, "--manual-db", help="Pin to a fixed dBFS level"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
):
    """Add or update a receiver"""
    from spectrum_alert.core.domain.receivers import SDRReceiver
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    import os
    import uuid

    try:
        receiver = SDRReceiver(
            id=str(uuid.uuid4()),
            name=name,
            mode=mode,
            lat=lat,
            lon=lon,
            device_index=device_index if mode == "local" else None,
            host=host if mode == "rtltcp" else None,
            port=port if mode == "rtltcp" else None,
            gain_db=gain,
            ppm_error=ppm,
            manual_rssi_db=manual_db,
        )
        receiver.validate_transport()
    except Exception as exc:
        console.print(f"[bold red]Invalid receiver: {exc}[/bold red]")
        raise typer.Exit(1)

    store = ReceiverStore(os.path.join(data_dir, "receivers.json"))
    store.save(receiver)
    console.print(f"[green]Saved receiver {receiver.name} ({receiver.id})[/green]")


@geo_app.command("scan")
def geo_scan(
    frequency: float = typer.Argument(..., help="Centre frequency in Hz"),
    receiver: Optional[str] = typer.Option(None, "--receiver", "-r", help="Receiver id or name"),
    span: float = typer.Option(2_000_000.0, "--span", help="Span in Hz"),
    bins: int = typer.Option(64, "--bins", help="Number of output bins"),
    integration: float = typer.Option(1.0, "--integration", "-i", help="Integration seconds"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
):
    """Run a real spectrum scan and print the peaks"""
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner
    from spectrum_alert.core.services.spectrum_analysis import analyze_spectrum
    import os

    store = ReceiverStore(os.path.join(data_dir, "receivers.json"))
    receivers = store.enabled()
    if not receivers:
        console.print("[bold red]No enabled receivers.[/bold red]")
        raise typer.Exit(1)

    if receiver:
        match = next(
            (r for r in receivers
             if r.id == receiver or r.name.lower() == receiver.lower()),
            None,
        )
        if match is None:
            console.print(f"[bold red]Receiver '{receiver}' not found.[/bold red]")
            raise typer.Exit(1)
        targets = [match]
    else:
        targets = [receivers[0]]

    scanner = SpectrumScanner()
    for target in targets:
        console.print(f"Scanning [cyan]{target.name}[/cyan] at {frequency/1e6:.6f} MHz...")
        try:
            scan = scanner.scan(target, frequency, span, None, integration, bins)
        except Exception as exc:
            console.print(f"[bold red]{target.name}: {exc}[/bold red]")
            continue

        analysis = analyze_spectrum(
            [p["hz"] for p in scan.points],
            [p["dbm"] for p in scan.points],
            frequency,
        )
        table = Table(title=f"{target.name} ({scan.source})")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Peak", f"{scan.peak_dbfs:.1f} dBFS @ {(scan.peak_hz or 0)/1e6:.6f} MHz")
        table.add_row("Noise floor", f"{scan.noise_floor_dbfs:.1f} dBFS")
        table.add_row("At target", f"{analysis.target_dbfs:.1f} dBFS")
        table.add_row("Prominence", f"{analysis.prominence_db:.1f} dB")
        table.add_row("Reliability", f"{analysis.reliability:.2f}")
        console.print(table)
        if scan.warning:
            console.print(f"[yellow]note: {scan.warning}[/yellow]")


@geo_app.command("triangulate")
def geo_triangulate(
    frequency: float = typer.Argument(..., help="Frequency to locate, in Hz"),
    reference: Optional[float] = typer.Option(None, "--reference", "-r", help="Reference RSSI at 1 m, dBm"),
    exponent: Optional[float] = typer.Option(None, "--exponent", "-n", help="Path-loss exponent"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
):
    """Locate a frequency using every enabled receiver"""
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner
    from spectrum_alert.core.services.spectrum_analysis import analyze_spectrum
    from spectrum_alert.core.services.propagation import frequency_default_path_loss, rssi_to_distance_meters
    from spectrum_alert.core.services.triangulation import TriangulationReading, triangulate_position
    from spectrum_alert.core.services.calibration import CalibrationTable
    import os

    store = ReceiverStore(os.path.join(data_dir, "receivers.json"))
    receivers = store.enabled()
    if len(receivers) < 2:
        console.print("[bold red]At least 2 enabled receivers are required.[/bold red]")
        raise typer.Exit(1)

    calibration = CalibrationTable.load(os.path.join(data_dir, "calibration.json"))
    defaults = frequency_default_path_loss(frequency)
    n = exponent if exponent is not None else defaults.exponent
    if exponent is not None and not (1.5 <= exponent <= 4.5):
        console.print(
            f"[yellow]n={exponent} is outside the plausible range; using {defaults.exponent}[/yellow]"
        )
        n = defaults.exponent
    p0 = reference if reference is not None else defaults.reference_rssi_dbm

    scanner = SpectrumScanner()
    readings = []
    for target in receivers:
        try:
            scan = scanner.scan(target, frequency, 120_000.0, None, 0.5, 128)
        except Exception as exc:
            console.print(f"[yellow]{target.name}: {exc}[/yellow]")
            continue
        analysis = analyze_spectrum(
            [p["hz"] for p in scan.points], [p["dbm"] for p in scan.points], frequency
        )
        offset = calibration.offset_for(target.id, frequency)
        level = analysis.target_dbfs + offset
        modelled = rssi_to_distance_meters(level, p0, n)
        colour = "green" if analysis.reliability >= 0.5 else "yellow"
        console.print(
            f"  {target.name}: {level:.1f} dBFS -> {modelled:.0f} m "
            f"(prominence {analysis.prominence_db:.1f} dB, reliability "
            f"[{colour}]{analysis.reliability:.2f}[/{colour}])"
        )
        readings.append(
            TriangulationReading(
                receiver_id=target.id, name=target.name, lat=target.lat, lon=target.lon,
                rssi_db=level, source=scan.source, reliability=analysis.reliability,
            )
        )

    if len(readings) < 2:
        console.print("[bold red]Not enough receivers reported a level.[/bold red]")
        raise typer.Exit(1)

    estimate = triangulate_position(readings, p0, n)
    colour = {"good": "green", "fair": "yellow", "poor": "red"}.get(estimate.quality, "white")

    table = Table(title=f"Estimate at {frequency/1e6:.6f} MHz")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Latitude", f"{estimate.lat:.6f}")
    table.add_row("Longitude", f"{estimate.lon:.6f}")
    table.add_row("Uncertainty", f"+/- {estimate.uncertainty_meters:.0f} m")
    table.add_row("RMS residual", f"{estimate.rms_residual_meters:.0f} m ({estimate.rms_rssi_residual_db:.2f} dB)")
    table.add_row("Receiver spread", f"{estimate.receiver_spread_meters:.0f} m")
    table.add_row("Quality", f"[{colour}]{estimate.quality}[/{colour}]")
    table.add_row("Reasons", "; ".join(estimate.quality_reasons))
    console.print(table)

    residual_table = Table(title="Residuals")
    residual_table.add_column("Receiver")
    residual_table.add_column("Measured dBFS")
    residual_table.add_column("Predicted dBFS")
    residual_table.add_column("Residual dB")
    for detail in estimate.residuals:
        residual_table.add_row(
            detail.name,
            f"{detail.measured_rssi_db:.1f}",
            f"{detail.predicted_rssi_db:.1f}",
            f"{detail.residual_db:+.2f}",
        )
    console.print(residual_table)
    console.print("[dim]RSSI path-loss model; accuracy depends on calibration and geometry.[/dim]")


@geo_app.command("fit-ple")
def geo_fit_ple(
    apply: bool = typer.Option(False, "--apply", help="Store the fitted model"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
):
    """Fit the path-loss exponent from known transmitter positions"""
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner
    from spectrum_alert.core.services.spectrum_analysis import analyze_spectrum
    from spectrum_alert.core.services.calibration import (
        CalibrationTable, build_observations, fit_path_loss_exponent,
    )
    from spectrum_alert.web.geolocate import DEFAULT_TRANSMITTERS
    import os

    store = ReceiverStore(os.path.join(data_dir, "receivers.json"))
    receivers = store.enabled()
    if not receivers:
        console.print("[bold red]No enabled receivers.[/bold red]")
        raise typer.Exit(1)

    calibration = CalibrationTable.load(os.path.join(data_dir, "calibration.json"))
    scanner = SpectrumScanner()
    measurements = {}

    for transmitter in DEFAULT_TRANSMITTERS:
        frequency = transmitter["frequency_hz"]
        for target in receivers:
            if target.manual_rssi_db is not None:
                continue
            try:
                scan = scanner.scan(target, frequency, 160_000.0, None, 1.0, 128)
            except Exception as exc:
                console.print(f"[yellow]{target.name} @ {frequency/1e6:.1f} MHz: {exc}[/yellow]")
                continue
            analysis = analyze_spectrum(
                [p["hz"] for p in scan.points], [p["dbm"] for p in scan.points], frequency
            )
            if analysis.reliability >= 0.2:
                measurements[(target.id, frequency)] = analysis.target_dbfs

    observations = build_observations(
        DEFAULT_TRANSMITTERS, [r.to_dict() for r in receivers], measurements, calibration
    )
    result = fit_path_loss_exponent(observations, default_frequency_hz=DEFAULT_TRANSMITTERS[0]["frequency_hz"])

    table = Table(title="Path-loss fit")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("n", f"{result.path_loss_exponent:.3f}")
    table.add_row("P0", f"{result.reference_rssi_dbm:.2f} dBm")
    table.add_row("R2", f"{result.r_squared:.4f}")
    table.add_row("Points used", f"{sum(1 for p in result.points if p.used_in_fit)}/{len(result.points)}")
    if result.fallback_applied:
        table.add_row("Fallback", "yes (band defaults used)")
    console.print(table)

    if result.warning:
        console.print(f"[yellow]{result.warning}[/yellow]")
    for error in result.errors:
        console.print(f"[dim]{error}[/dim]")

    for point in result.points:
        marker = "Y" if point.used_in_fit else "N"
        console.print(
            f"  {point.transmitter[:24]:<24} {point.distance_meters/1000:7.2f} km "
            f"{point.adjusted_dbfs:7.1f} dBFS  resid {point.residual_db:+.2f}  in={marker}"
        )

    if apply and not result.fallback_applied:
        console.print(
            f"[green]Use: spectrum-alert geo triangulate --reference {result.reference_rssi_dbm:.2f} "
            f"--exponent {result.path_loss_exponent:.3f}[/green]"
        )
    elif apply:
        console.print("[yellow]Fit was not good enough to apply.[/yellow]")


@geo_app.command("calibrate")
def geo_calibrate(
    apply: bool = typer.Option(False, "--apply", help="Persist the offsets"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
):
    """Compute per-receiver calibration offsets from known transmitters"""
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner
    from spectrum_alert.core.services.spectrum_analysis import analyze_spectrum
    from spectrum_alert.core.services.calibration import (
        CalibrationTable, compute_relative_offsets, frequency_key,
    )
    from spectrum_alert.web.geolocate import DEFAULT_TRANSMITTERS
    import os

    store = ReceiverStore(os.path.join(data_dir, "receivers.json"))
    receivers = store.enabled()
    if not receivers:
        console.print("[bold red]No enabled receivers.[/bold red]")
        raise typer.Exit(1)

    calibration = CalibrationTable.load(os.path.join(data_dir, "calibration.json"))
    scanner = SpectrumScanner()
    preview = {}

    for transmitter in DEFAULT_TRANSMITTERS:
        frequency = transmitter["frequency_hz"]
        readings = []
        for target in receivers:
            if target.manual_rssi_db is not None:
                continue
            try:
                scan = scanner.scan(target, frequency, 160_000.0, None, 1.0, 128)
            except Exception as exc:
                console.print(f"[yellow]{target.name} @ {frequency/1e6:.1f} MHz: {exc}[/yellow]")
                continue
            analysis = analyze_spectrum(
                [p["hz"] for p in scan.points], [p["dbm"] for p in scan.points], frequency
            )
            readings.append({
                "receiver_id": target.id,
                "receiver_name": target.name,
                "rssi_dbfs": analysis.target_dbfs,
            })

        offsets, rejected = compute_relative_offsets(readings, frequency)
        rejected and [console.print(f"[yellow]rejected: {r}[/yellow]") for r in rejected]
        for receiver_id, entry in offsets.items():
            preview.setdefault(receiver_id, {})[frequency_key(frequency)] = entry

    table = Table(title="Calibration offsets (dB)")
    for receiver in receivers:
        entries = preview.get(receiver.id, {})
        if not entries:
            continue
        table.add_row(receiver.name, ", ".join(
            f"{parse_frequency_label(key)}: {entry['offset']:+.2f}"
            for key, entry in sorted(entries.items(), key=lambda kv: int(kv[0].lstrip('k')))
        ))

    if any(preview.values()):
        console.print(table)
    else:
        console.print("[yellow]No usable measurements; nothing to calibrate against.[/yellow]")

    if apply and preview:
        calibration.merge(preview)
        calibration.save()
        console.print("[green]Calibration offsets applied and saved.[/green]")
    elif preview:
        console.print("[dim]Re-run with --apply to store these offsets.[/dim]")


def parse_frequency_label(key: str) -> str:
    """Render a ``k446100`` calibration key as a readable frequency."""
    from spectrum_alert.core.services.calibration import parse_frequency_key

    return f"{parse_frequency_key(key)/1e6:.3f} MHz"


if __name__ == "__main__":
    app()
