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
    session_name: Optional[str] = typer.Option(None, "--name", "-n", help="Session name")
):
    """Start spectrum monitoring"""
    console.print(f"[bold green]Starting spectrum monitoring[/bold green]")
    console.print(f"Frequency range: {frequency_start} - {frequency_end} MHz")
    console.print(f"Detection mode: {mode.value}")
    
    # Convert MHz to Hz
    freq_start_hz = frequency_start * 1e6
    freq_end_hz = frequency_end * 1e6
    
    try:
        # Import and initialize components
        from spectrum_alert.infrastructure.sdr import RTLSDRInterface
        from spectrum_alert.infrastructure.storage import DataStorage
        from spectrum_alert.core.services.feature_extraction import FeatureExtractor
        from spectrum_alert.application.use_cases.spectrum_monitoring import SpectrumMonitoringUseCase
        
        # Initialize components
        sdr = RTLSDRInterface()
        storage = DataStorage()
        feature_extractor = FeatureExtractor(lite_mode=(mode == DetectionMode.LITE))
        monitoring_use_case = SpectrumMonitoringUseCase(sdr, storage, feature_extractor)
        
        # Start session
        session_name = session_name or f"monitoring_{int(freq_start_hz/1e6)}_{int(freq_end_hz/1e6)}"
        session = monitoring_use_case.start_monitoring_session(
            session_name=session_name,
            frequency_range=(freq_start_hz, freq_end_hz),
            detection_mode=mode
        )
        
        console.print(f"[green]Session started: {session.id}[/green]")
        
        # Start monitoring loop
        import time
        with sdr.safe_operation():
            start_time = time.time()
            while True:
                try:
                    # Capture data at start frequency (simplified for demo)
                    spectrum_data = monitoring_use_case.capture_spectrum_data(
                        frequency=freq_start_hz,
                        sample_rate=2.4e6,  # 2.4 MHz
                        gain=40,
                        duration=0.1
                    )
                    
                    # Extract features
                    monitoring_use_case.extract_features(spectrum_data)
                    
                    console.print(f"[dim]Captured data at {freq_start_hz/1e6:.1f} MHz[/dim]")
                    
                    # Check duration
                    if duration and (time.time() - start_time) > duration:
                        break
                        
                    time.sleep(1)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Monitoring interrupted by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break
        
        # Stop session
        monitoring_use_case.stop_monitoring_session()
        console.print("[green]Monitoring session completed[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to start monitoring: {e}[/red]")
        raise typer.Exit(1)


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


if __name__ == "__main__":
    app()
