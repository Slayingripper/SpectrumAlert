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

app.add_typer(monitor_app, name="monitor")
app.add_typer(train_app, name="train")
app.add_typer(config_app, name="config")
app.add_typer(system_app, name="system")


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
        # Check SDR availability
        from spectrum_alert.infrastructure.sdr import RTLSDRInterface
        try:
            sdr = RTLSDRInterface()
            sdr.open()
            sdr.close()
            table.add_row("SDR Device", "[green]Available[/green]", "RTL-SDR detected")
        except Exception as e:
            table.add_row("SDR Device", "[red]Error[/red]", str(e))
        
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


if __name__ == "__main__":
    app()
