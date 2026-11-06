# Configuration

SpectrumAlert has three configuration surfaces:

1. `config/config.ini` — persistent CLI and service configuration
2. Dashboard **Config** — browser and server settings
3. Environment variables — container service configuration

## `config/config.ini`

```ini
[HAM_BANDS]
bands = 144e6-146e6,420e6-440e6

[GENERAL]
freq_step = 100000
sample_rate = 2.048e6
runs_per_freq = 3
sdr_type = rtlsdr
lite_mode = false

[MONITORING]
freq_step = 1000000
scan_delay = 0.01
runs_per_freq = 1
use_wideband_scanning = false

[RECEIVER]
latitude = 35.130
longitude = 33.310

[MQTT]
broker = localhost
port = 1883
topic_anomalies = hamradio/anomalies
topic_modulation = hamradio/modulation
topic_signal_strength = hamradio/signal_strength
topic_coordinates = hamradio/coordinates

[DATA]
data_directory = data
max_file_size_mb = 100
cleanup_after_days = 30

[MODELS]
model_directory = models
auto_retrain_threshold = 1000
contamination_rate = 0.05

[LOGGING]
log_level = INFO
log_directory = logs
max_log_size_mb = 50
backup_count = 5
```

Unknown keys are ignored. Invalid numeric values are reported on startup.

## Dashboard configuration

### Basic

- **Detection threshold** — stored in the browser for the dashboard session
- **Frequency range** — used by advanced and data-collection paths
- **Scan interval** — seconds between live scans
- **Strict mode** — built into the detector and cannot be disabled

### Advanced

Advanced and multi-band settings are recorded with the server. Some fields are
stored for later builds rather than applied by the current live detector. The
dashboard disclosure states which values are active.

### MQTT

Broker, port and topic prefix can be recorded from the dashboard. The current
build records the settings and exposes a test event; it does not open a broker
connection from the dashboard process.

## Environment variables

The container stack reads:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPECTRUM_ALERT_WEB_PORT` | `8088` | Host port mapped to dashboard port 8000 |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `COLLECTION_HOURS` | `24` | Autonomous collection window |
| `RETRAIN_INTERVAL_HOURS` | `168` | Autonomous retraining interval |
| `MIN_TRAINING_SAMPLES` | `1000` | Autonomous training floor |
| `SERVICE_ALERT_THRESHOLD` | `0.7` | Service-mode threshold |
| `MQTT_BROKER` | `localhost` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `spectrum_alert` | MQTT topic prefix |

See `.env.example` for the full template.

## Data and model directories

Both paths are relative to the working directory:

```text
data/     captures, features, anomalies, logs
models/   trained joblib and pkl artifacts
```

In Docker, these are mounted from the checkout so data survives container
rebuilds.

## Related

- [Data and Training](Data-and-Training.md)
- [Docker](Docker.md)
- [Troubleshooting](Troubleshooting.md)
