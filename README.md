# SpectrumAlert v1.1

**SpectrumAlert** is an advanced RF spectrum monitoring and anomaly detection system designed for ham radio frequency monitoring, RF fingerprinting, and real-time spectrum analysis. Built with modern Python architecture and professional CLI interface.

## 🚀 Features

### Core Capabilities
- **Real-time Spectrum Monitoring**: Continuous band sweeping with overlapping captures
- **Anomaly Detection**: Hann-windowed periodogram with robust noise-floor estimation, DC/edge rejection and novelty suppression
- **Transmitter Geolocation**: RSSI triangulation across multiple receivers, with confidence ellipse and fit-quality grading
- **Receiver Calibration**: Per-receiver, per-frequency dB offsets and a fitted path-loss exponent
- **Multi-transport Scanning**: Local `rtl_power`, remote `rtl_tcp` (native IQ client), or pinned manual levels
- **Exact Frequency Detection**: Sub-bin parabolic interpolation on the detected peak
- **Multi-mode Detection**: Lite mode (2 features) and Full mode (12 features) for different performance needs
- **Band Plans**: PMR446, EU/US ISM, 2 m and 70 cm amateur, air band and FM broadcast

### Modern Architecture
- **Professional CLI Interface**: Rich console output with tables, colors, and progress indicators
- **Clean Architecture**: Domain-driven design with declared ports in `core/interfaces/ports.py`
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Device Safety**: Serialised access to each SDR, in-process and cross-process, with hard timeouts
- **Crash-safe Storage**: Atomic writes and legacy-format reads
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

# Reduce false positives with artifact filtering
spectrum-alert monitor start --freq-start 144 --freq-end 148 --dc-exclude-hz 8000 --edge-exclude-hz 20000 \
  --threshold 0.8 --strict-threshold --novelty --novelty-cooldown-s 5.0
```

#### Multi-band Autonomous Monitoring
```bash
# Rotate across 2m and 70cm bands, one cycle each and repeat forever
spectrum-alert monitor autonomous-multiband -b 144-148 -b 430-440 --data-minutes 5 --monitor-minutes 15 --max-cycles-per-band 1

# Same with MQTT and finite rotations
spectrum-alert monitor autonomous-multiband -b 144-148 -b 430-440 --data-minutes 5 --monitor-minutes 15 --max-cycles-per-band 1 --rounds 3 \
  --sample-rate 2.048 --gain 30 --threshold 0.7 --mqtt-broker YOUR_BROKER --mqtt-topic-prefix spectrum_alert

# With artifact suppression to reduce false positives
spectrum-alert monitor autonomous-multiband -b 144-148 -b 430-440 --dc-exclude-hz 10000 --edge-exclude-hz 25000 \
  --threshold 0.8 --strict-threshold --continuous-learning
```

#### Continuous Learning
```bash
# Enable training every cycle to adapt the model over time
spectrum-alert monitor autonomous --freq-start 144 --freq-end 148 --continuous-learning

# Combine with multi-band rotation
spectrum-alert monitor autonomous-multiband -b 144-148 -b 430-440 --continuous-learning

# With advanced filtering to minimize false positives
spectrum-alert monitor autonomous --freq-start 144 --freq-end 148 --continuous-learning \
  --dc-exclude-hz 12000 --edge-exclude-hz 30000 --threshold 0.75 --strict-threshold --novelty-freq-tol-hz 3000
```

#### Model Training
```bash
# Train models using last 7 days of data
spectrum-alert train models --mode full --days 7

# Train lite models
spectrum-alert train models --mode lite --days 3
```

### Geolocation and Calibration

Geolocation needs at least two enabled receivers. Add them with coordinates:

```bash
# A local dongle
spectrum-alert geo add-receiver Home --lat 35.130 --lon 33.310 --gain 30 --ppm 20

# A remote host running rtl_tcp
spectrum-alert geo add-receiver Hill --lat 35.080 --lon 33.380 \
  --mode rtltcp --host 10.0.0.5 --port 1234

spectrum-alert geo receivers
```

```bash
# One real scan, with peak, noise floor, prominence and reliability
spectrum-alert geo scan 446100000 --span 200000 -i 0.5

# Locate an emitter from every enabled receiver
spectrum-alert geo triangulate 446100000

# Equalise receivers against known broadcast transmitters (preview, then apply)
spectrum-alert geo calibrate
spectrum-alert geo calibrate --apply

# Fit the path-loss exponent from transmitter geometry
spectrum-alert geo fit-ple
```

The same functionality is available in the web interface, in the **analysis
workspace** section of the dashboard: a live spectrum waterfall, a map with
receiver positions and the estimated emitter with its confidence ellipse,
receiver management, calibration, and a PMR446 channel scanner.

The workspace used to be a separate page at `/geolocate`. It is now part of the
dashboard so that locating something does not take the operator away from the
system panels and alerts. `/geolocate` redirects to `/#workspace`, so existing
bookmarks still land in the right place.

### RF Fingerprinting (device identification)

Identification answers "what is this transmitter?" where geolocation answers
"where is it?". Both run off one capture, so the dashboard's analysis workspace
shows them **on one screen**, side by side under a shared waterfall, receiver
roster and target frequency. The map panel locates the target; the signature
panel identifies the transmitter; neither is hidden behind a tab. The layout is Bootstrap 5.3.3
(vendored under `static/vendor/bootstrap`, so no internet access is needed) on a
fluid grid that fills the viewport at any width.

A signature is a 12-value, gain-invariant feature vector: envelope statistics,
normalised higher-order moments of the complex signal, envelope rise/fall rates
and lag-1 correlation, and three spectral-shape measures. Gain invariance is the
point -- a signature captured on one receiver still matches on another, because
nothing in the vector restates absolute power.

```bash
# Capture from a receiver and identify it against the library
curl -X POST localhost:8081/api/fingerprints/capture \
  -H 'Content-Type: application/json' \
  -d '{"sdrId":"hilltop","frequencyHz":145500000,"captures":4}'

# Enrol a known device so future captures can be identified
curl -X POST localhost:8081/api/fingerprints/capture \
  -H 'Content-Type: application/json' \
  -d '{"sdrId":"hilltop","frequencyHz":145500000,"label":"Base station",
       "vendor":"Acme","model":"TX-1"}'

curl localhost:8081/api/fingerprints   # the reference library
```

Each capture averages several short bursts, because one capture carries the
receiver's own noise realisation and would otherwise not match its own
reference. Results are deliberately conservative:

| Status | Meaning |
| --- | --- |
| `identified` | One candidate close enough and clearly separated from the rest |
| `ambiguous` | Two references too close to separate; no ID is claimed |
| `indeterminate` | The library holds no distinct references to compare against |
| `no-match` | The closest reference is beyond the accept threshold |
| `unknown` | The library is empty |

An empty library reports `unknown` rather than guessing, and a pinned
`manualRssiDb` receiver is refused outright because a fixed level carries no
modulation to fingerprint.

### Web Interface

```bash
# Dashboard
spectrum-alert web --port 8000

# Or directly, for reload/worker control
uvicorn spectrum_alert.web.run:create_app --factory --port 8000
```

Pages:

| Path | Purpose |
| --- | --- |
| `/` | Dashboard: system status, anomalies, model training, data management |
| `/` | System panels, alerts, model, **analysis workspace** (waterfall, receivers, calibration, map, Identify, PMR446) |
| `/geolocate` | Redirects to `/#workspace` |

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
├── cli/                    # CLI interface (main + the `geo` commands)
├── core/                   # Business logic, no infrastructure imports
│   ├── domain/            # Pydantic models: SpectrumData, SDRReceiver, ...
│   ├── interfaces/        # Ports (Protocols) for the hexagonal boundary
│   ├── services/          # propagation, spectrum_analysis, calibration,
│   │                      # triangulation, band_plans, feature_extraction
│   └── exceptions.py      # Custom exceptions
├── infrastructure/        # Adapters
│   ├── sdr/               # RTL-SDR, rtl_tcp, scanner, device locks
│   ├── messaging/         # MQTT communication
│   ├── storage/           # DataStorage, ReceiverStore
│   └── monitoring/        # System monitoring
├── application/           # Use cases and workflows
└── web/                    # FastAPI app, geolocation router, UI assets
```

### Key Components

#### Domain Models
- `SpectrumData`: RF spectrum data representation
- `AnomalyDetection`: Anomaly detection results with confidence scores
- `MonitoringSession`: Session management and statistics
- `FeatureVector`: ML feature extraction results

#### Infrastructure Services
- `SDRInterface`: RTL-SDR device management; the only backend is `RTLSDRInterface`
  (`pyrtlsdr`)
- `SpectrumScanner`: unified scanner over `rtl_power`, `rtl_tcp` and manual levels
- `RtltcpClient`: native `rtl_tcp` protocol client for remote dongles
- `hold_device`: in-process mutex plus an advisory `flock`, so a dongle is never
  retuned mid-transfer or shared with another process
- `MQTTManager`: Robust MQTT communication with reconnection
- `DataStorage`: Structured data persistence; atomic writes, schema-versioned,
  reads both current and archived file formats
- `ReceiverStore`: receiver roster with atomic JSON persistence
- `SystemMonitor`: Resource monitoring and health checks

### Signal Levels: dBFS, Not dBm

An RTL-SDR has no calibrated reference. Everything it reports is **dBFS** (dB
relative to full scale) until a per-receiver calibration offset is applied, and
even then the result is only as good as that calibration. The dashboard, the
`geo` commands and the anomaly metadata all label levels as dBFS for this
reason. The path-loss model additionally assumes a reference power at 1 m
(`referenceRssiDbm`) and an exponent (`pathLossExponent`); both are band
defaults until you fit them with `geo fit-ple`.

### Path-Loss Model

RSSI is converted to a distance with::

    d = 10 ** ((P0 - RSSI) / (10 * n))

then a Huber-robust weighted least-squares fit finds the emitter position that
best matches every receiver's distance circle. Fit quality is graded `good` /
`fair` / `poor` from the RSSI residual, the per-receiver outliers, the
normalised geometric residual and the confidence ellipse, and the reasons are
reported alongside it. Two receivers are usable but underconstrained; three or
more give a meaningful fix.

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t spectrum-alert .

# Run with CLI entrypoint (uses non-root user). Override command as needed.
docker run --rm --privileged --device=/dev/bus/usb -v "$(pwd)/data:/app/data" spectrum-alert monitor status

# Start autonomous mode (single band) with enhanced filtering
docker run --rm --privileged --device=/dev/bus/usb -v "$(pwd)/data:/app/data" spectrum-alert monitor autonomous \
  --freq-start 144 --freq-end 148 --data-minutes 5 --monitor-minutes 15 --continuous-learning \
  --threshold 0.8 --strict-threshold --dc-exclude-hz 12000 --edge-exclude-hz 30000 --novelty-freq-tol-hz 5000

# Multi-band rotation in Docker with comprehensive filtering
docker run --rm --privileged --device=/dev/bus/usb -v "$(pwd)/data:/app/data" spectrum-alert monitor autonomous-multiband \
  -b 144-148 -b 430-440 --data-minutes 5 --monitor-minutes 15 --max-cycles-per-band 1 \
  --threshold 0.8 --strict-threshold --dc-exclude-hz 10000 --edge-exclude-hz 25000 --novelty --novelty-freq-tol-hz 5000

# With false positive reduction
docker run --rm --privileged --device=/dev/bus/usb -v "$(pwd)/data:/app/data" spectrum-alert monitor autonomous-multiband \
  -b 144-148 -b 430-440 --dc-exclude-hz 10000 --edge-exclude-hz 25000 --threshold 0.8 --strict-threshold --continuous-learning

# Production-ready autonomous monitoring with maximum filtering
docker run --rm --privileged --device=/dev/bus/usb -v "$(pwd)/data:/app/data" spectrum-alert monitor autonomous \
  --freq-start 144 --freq-end 148 --continuous-learning --threshold 0.85 --strict-threshold \
  --dc-exclude-hz 15000 --edge-exclude-hz 35000 --novelty --novelty-freq-tol-hz 3000 --novelty-cooldown-s 10.0 \
  --sample-rate 2.048 --gain 30 --data-minutes 3 --monitor-minutes 12
```

### Compose (example)
```yaml
services:
  spectrum-alert:
    build: .
    image: spectrum-alert:latest
    devices:
      - "/dev/bus/usb:/dev/bus/usb"
    environment:
      - MQTT_BROKER=your-broker
    command: [
      "monitor", "autonomous", 
      "--freq-start", "144", "--freq-end", "148", 
      "--continuous-learning", "--threshold", "0.8", "--strict-threshold",
      "--dc-exclude-hz", "12000", "--edge-exclude-hz", "30000",
      "--novelty", "--novelty-freq-tol-hz", "5000"
    ]
    volumes:
      - "./data:/app/data"
    restart: unless-stopped
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

### Geolocation
```
Estimate at 446.100000 MHz
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Field            ┃ Value               ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Latitude         │ 35.100000           │
│ Longitude        │ 33.350000           │
│ Uncertainty      │ +/- 50 m            │
│ RMS residual     │ 0 m (0.00 dB)       │
│ Receiver spread  │ 9597 m              │
│ Quality          │ good                │
│ Reasons          │ consistent fit      │
└──────────────────┴─────────────────────┘
Residuals
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Receiver ┃ Measured dBFS ┃ Predicted dBFS ┃ Residual dB ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ A        │ -174.3        │ -174.3         │ +0.00       │
└──────────┴───────────────┴───────────────┴─────────────┘
```

## 🔧 Advanced Features

### New in v1.1+
- Multi-band autonomous rotation: monitor multiple bands sequentially with `monitor autonomous-multiband` and repeated `--band` options.
- Continuous learning: enable `--continuous-learning` to retrain every cycle for adaptive models.
- Stronger detector gating and artifact suppression: DC/edge exclusion, novelty filter, frequency-range clamp.

### False Positive Reduction
- **DC Exclusion**: `--dc-exclude-hz 8000` ignores ±8kHz around center frequency to avoid LO/DC spur artifacts
- **Edge Exclusion**: `--edge-exclude-hz 20000` ignores ±20kHz near passband edges to avoid alias/rolloff artifacts  
- **Novelty Filter**: `--novelty --novelty-cooldown-s 5.0` suppresses repeated alerts from stable carriers
- **Threshold Tuning**: `--threshold 0.8` raises detection threshold to reduce noise-triggered alerts
- **Strict Threshold Mode**: `--strict-threshold` (default: enabled) requires borderline scores to have higher SNR
- **Frequency Tolerance**: `--novelty-freq-tol-hz 3000` groups nearby frequencies as same signal source

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
- **RF Fingerprinting**: Identify a device from its RF signature and tell whether an emission comes from a transmitter seen before. Implemented as the Identify view on `/geolocate` (see above); a reference library of known devices is built by capturing them once.
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

#### Scenario 3: Multi-receiver Direction Finding
1. **Step 1**: Deploy nodes running SpectrumAlert in different locations, each with a
   receiver carrying coordinates (`geo add-receiver`).
2. **Step 2**: Run `geo calibrate --apply` to equalise the receivers against known
   transmitters, then `geo fit-ple` for the propagation model.
3. **Step 3**: Scan the frequency from every receiver (`geo scan`) or locate it
   directly (`geo triangulate`). Remote nodes can be `rtl_tcp` servers.
4. **Step 4**: Use the fix together with its confidence ellipse and fit-quality grade.
   Use `/geolocate` for the map and likelihood surface.

Accuracy is bounded by geometry and by multipath. Two receivers are
underconstrained; three or more are needed for a meaningful fix, and a receiver
under deep fade is reported as such rather than quietly skewing the result.

## Anomaly Detection and Heatmap

One of the core features of **Spectrum Alert** is anomaly detection. The software uses machine learning models to detect anomalies in the signal's characteristics, which could indicate unauthorized or illegal transmissions.

### Heatmap of Suspected Devices and Anomalies:
- **Heatmap**: Detected anomalies and suspected devices can be used to generate a heatmap showing the geographical concentration of anomalies. When multiple devices are deployed, they can collectively contribute data to enhance the accuracy of anomaly detection and location triangulation.
- **Direction Finding**: By analyzing the anomalies detected by multiple geographically distributed receivers, the system can triangulate the position of unauthorized transmissions, helping you locate the source of illegal or suspicious activity in the RF spectrum.

This feature makes **Spectrum Alert** a powerful tool for authorities or ham radio enthusiasts who want to identify and take action against illegal spectrum users.

### ToDo
[x] Create menu
[x] Add support for multiple SDR transports (local, rtl_tcp, manual override)
[x] Multi-receiver geolocation with calibration and path-loss fitting
[x] Band plans (PMR446, ISM, amateur, air band)
[] Support for additional tuner families (LimeSDR, HackRF, RSP1A)

### License

The software is released under the GNU General Public License v3.0. You are free to use, modify, and distribute the software under the terms of the license.