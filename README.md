# SpectrumAlert v1.1

**SpectrumAlert** is an advanced RF spectrum monitoring and anomaly detection system designed for ham radio frequency monitoring, RF fingerprinting, and real-time spectrum analysis. Built with modern Python architecture and professional CLI interface.

## 🚀 Features

### Core Capabilities
- **Real-time Spectrum Monitoring**: Continuous scanning of configured frequency bands
- **Machine Learning Anomaly Detection**: Isolation Forest-based anomaly detection with adaptive thresholds
- **RF Fingerprinting**: Device identification and classification using unique signal characteristics
- **Exact Frequency Detection**: Pinpoint anomaly detection with 6-decimal precision frequency logging
- **Multi-mode Detection**: Lite mode (2 features) and Full mode (12 features) for different performance needs

### Modern Architecture
- **Professional CLI Interface**: Rich console output with tables, colors, and progress indicators
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Error Handling**: Robust exception hierarchy with recovery mechanisms
- **Configuration Management**: Structured configuration with validation

### Integration & Deployment
- **MQTT Integration**: Real-time publishing of anomalies and system status
- **Docker Support**: Complete containerization with autonomous service mode
- **Raspberry Pi Compatible**: Optimized for low-resource embedded deployments
- **Data Persistence**: Structured storage of spectrum data, features, and anomalies
- **System Monitoring**: Health checks and resource monitoring

## 📦 Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Slayingripper/SpectrumAlert.git
cd SpectrumAlert

# Install the package
pip install -e .

# Verify installation
spectrum-alert --help
```

### Requirements
- Python 3.8+
- RTL-SDR device
- Modern package dependencies (automatically installed)

## 🎯 Usage

### Command Line Interface

#### Basic Commands
```bash
# Show help
spectrum-alert --help

# Check system status
spectrum-alert monitor status

# View detailed system metrics
spectrum-alert system status

# Show version information
spectrum-alert version
```

#### Spectrum Monitoring
```bash
# Start monitoring FM radio band in full mode
spectrum-alert monitor start --freq-start 88 --freq-end 108 --mode full

# Monitor with specific duration
spectrum-alert monitor start --freq-start 144 --freq-end 148 --mode lite --duration 300

# Monitor with custom session name
spectrum-alert monitor start --freq-start 430 --freq-end 440 --mode full --name "UHF_Sweep"
```

#### Model Training
```bash
# Train models using last 7 days of data
spectrum-alert train models --mode full --days 7

# Train lite models
spectrum-alert train models --mode lite --days 3
```

### Configuration

The system uses structured configuration files in `config/config.ini`:

```ini
[GENERAL]
frequency_start = 144000000
frequency_end = 148000000
sample_rate = 2400000
gain = 40

[MONITORING]
step_size = 1000000
delay = 0.01
detection_mode = full

[MQTT]
broker = localhost
port = 1883
username = 
password = 
```

## 🏗️ Architecture

### Package Structure
```
spectrum_alert/
├── cli/                    # Modern CLI interface
├── core/                   # Business logic
│   ├── domain/            # Domain models (Pydantic)
│   ├── services/          # Core services
│   └── exceptions.py      # Custom exceptions
├── infrastructure/        # External integrations
│   ├── sdr/              # SDR interfaces
│   ├── messaging/        # MQTT communication
│   ├── storage/          # Data persistence
│   └── monitoring/       # System monitoring
└── application/           # Use cases and workflows
```

### Key Components

#### Domain Models
- `SpectrumData`: RF spectrum data representation
- `AnomalyDetection`: Anomaly detection results with confidence scores
- `MonitoringSession`: Session management and statistics
- `FeatureVector`: ML feature extraction results

#### Infrastructure Services
- `SDRInterface`: RTL-SDR device management with error recovery
- `MQTTManager`: Robust MQTT communication with reconnection
- `DataStorage`: Structured data persistence and retrieval
- `SystemMonitor`: Resource monitoring and health checks

## 🐳 Docker Deployment

### Basic Deployment
```bash
# Build and run
docker build -t spectrum-alert .
docker run --privileged --device=/dev/bus/usb spectrum-alert

# Using Docker Compose
docker-compose up
```

### Autonomous Service Mode
```bash
# 24-hour automated operation
docker-compose --profile autonomous up
```

## 📊 Monitoring Output

### System Status Dashboard
```
Monitoring Status
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Component     ┃ Status    ┃ Details           ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ SDR Device    │ Available │ RTL-SDR detected  │
│ System Health │ Healthy   │ CPU: 16.8%        │
│ Data Storage  │ Ready     │ 0 anomalies today │
└───────────────┴───────────┴───────────────────┘
```

### Resource Monitoring
```
System Status
┏━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Resource ┃ Usage ┃ Available ┃ Total    ┃
┡━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ CPU      │ 18.6% │ 24 cores  │ 24 cores │
│ Memory   │ 27.1% │ 45.7 GB   │ 62.7 GB  │
│ Disk     │ 67.4% │ 285.1 GB  │ 915.3 GB │
└──────────┴───────┴───────────┴──────────┘
```

## 🔧 Advanced Features

### Anomaly Detection Modes
- **Lite Mode**: 2 features (mean_amplitude, std_amplitude) - Fast, low resource
- **Full Mode**: 12 features including spectral, statistical, and cyclostationary features

### Exact Frequency Logging
Anomalies are logged with precise frequency information:
```
2025-08-08 12:30:45,123 - ANOMALY DETECTED at 146.520000 MHz (confidence: 0.87)
```

### MQTT Integration
Real-time publishing of:
- Anomaly detections with metadata
- System health status
- Monitoring session statistics
- Error and warning conditions

## 📈 Performance

### Optimized Scanning
- Fast scanning: 1MHz steps with 0.01s delays
- Configurable contamination rates for different RF environments
- Adaptive thresholds based on historical data

### Resource Efficiency
- Lite mode for Raspberry Pi deployment
- Memory-efficient data structures
- Optimized feature extraction algorithms

## 🛠️ Development

### Modern Python Practices
- Type hints throughout
- Pydantic data validation
- Clean Architecture principles
- Comprehensive error handling
- Rich CLI interface

### Extensibility
- Plugin-ready architecture
- Interface-based design
- Dependency injection ready
- Easy to add new SDR backends

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the existing architecture
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing documentation
- Review the CLI help system: `spectrum-alert --help`

3. Connect your RTL-SDR device to your system.

4. Configure the `config.ini` file located in the `Trainer/` folder. Define ham bands, sample rates, and MQTT settings.

## Usage

The software provides a menu-driven interface that allows you to choose between data gathering, model training, real-time spectrum monitoring, or an automated end-to-end workflow.

### Menu Options:

1. **Gather Data**: Collect RF data from the ham bands for further processing or training.
2. **Train Model**: Train RF fingerprinting and anomaly detection models using previously gathered data.
3. **Monitor Spectrum**: Real-time monitoring of spectrum activity, identifying anomalies and reporting them via MQTT.
4. **Automated Workflow**: Gathers data, trains models, and starts the monitor in a single automated flow.
5. **Check for Existing Data or Models**: Automates the process by skipping steps if data or models already exist.
6. **Start from Scratch**: Deletes all existing datasets and models, allowing you to begin fresh.

### Use Cases:
- **RF Spectrum Surveillance**: Continuously monitor ham bands and automatically detect anomalies that could indicate illegal transmissions.
- **Direction Finding**: Deploy multiple devices in different locations to triangulate the source of illegal transmissions by correlating anomalies across devices.
- **RF Fingerprinting**: Use the software to build models that identify specific devices based on their unique RF fingerprints. This can help track and identify persistent offenders in the spectrum.
- **Automated Monitoring and Detection**: Run the software in a fully automated mode to gather data, train models, and start monitoring without manual intervention.

### Example Scenarios:

#### Scenario 1: Simple Spectrum Monitoring
1. **Step 1**: Run the software and select `Gather Data`.
2. **Step 2**: After gathering data, train the model by selecting `Train Model`.
3. **Step 3**: Start real-time spectrum monitoring by selecting `Monitor Spectrum`.
4. **Step 4**: Detected anomalies will be logged and published to an MQTT broker for remote monitoring.

#### Scenario 2: Automated Process
1. **Step 1**: Select the `Automated Workflow` option.
2. **Step 2**: The software automatically gathers data, trains models, and begins real-time spectrum monitoring.
3. **Step 3**: Anomalies are detected in real-time, and data is published to an MQTT broker.

#### Scenario 3: Multi-device Direction Finding
1. **Step 1**: Deploy multiple devices running **Spectrum Alert** in different locations.
2. **Step 2**: Configure each device to monitor the same ham bands.
3. **Step 3**: Each device reports detected anomalies via MQTT.
4. **Step 4**: Correlate the detected anomalies across multiple devices to triangulate the source of the transmission. Use this to identify and track illegal transmissions or unauthorized users.

## Anomaly Detection and Heatmap

One of the core features of **Spectrum Alert** is anomaly detection. The software uses machine learning models to detect anomalies in the signal's characteristics, which could indicate unauthorized or illegal transmissions.

### Heatmap of Suspected Devices and Anomalies:
- **Heatmap**: Detected anomalies and suspected devices can be used to generate a heatmap showing the geographical concentration of anomalies. When multiple devices are deployed, they can collectively contribute data to enhance the accuracy of anomaly detection and location triangulation.
- **Direction Finding**: By analyzing the anomalies detected by multiple geographically distributed receivers, the system can triangulate the position of unauthorized transmissions, helping you locate the source of illegal or suspicious activity in the RF spectrum.

This feature makes **Spectrum Alert** a powerful tool for authorities or ham radio enthusiasts who want to identify and take action against illegal spectrum users.

### ToDo
[x] Create menu
[] Add support for multiple types of SDRs 

### License

The software is released under the GNU General Public License v3.0. You are free to use, modify, and distribute the software under the terms of the license.