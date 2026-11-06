# CLI Reference

The `spectrum-alert` console entry point remains fully supported. It uses the
same configuration, detector and data directory as the web dashboard.

```bash
spectrum-alert --help
spectrum-alert <command> --help
```

## Command groups

| Command | Purpose |
| --- | --- |
| `version` | Show version information |
| `web` | Start the web dashboard |
| `monitor` | Start, inspect and automate spectrum monitoring |
| `train` | Train a model from stored feature data |
| `system` | Host metrics and health |
| `geo` | Receiver roster, scanning, calibration and triangulation |
| `config` | Configuration command group |

## Monitoring

```bash
# Current receiver and system state
spectrum-alert monitor status

# Live scan
spectrum-alert monitor start --freq-start 144 --freq-end 148 --mode full

# Timed scan
spectrum-alert monitor start --freq-start 430 --freq-end 440 \
  --duration 300 --sample-rate 2.048 --gain 20

# Detection tuning
spectrum-alert monitor start --freq-start 144 --freq-end 148 \
  --threshold 0.8 --strict-threshold \
  --dc-exclude-hz 8000 --edge-exclude-hz 20000 \
  --novelty --novelty-cooldown-s 5
```

Common options:

| Option | Meaning |
| --- | --- |
| `--freq-start`, `--freq-end` | Scan range in MHz |
| `--mode full\|lite` | Detection feature set |
| `--duration` | Stop after this many seconds |
| `--sample-rate` | Receiver sample rate in MS/s |
| `--gain` | Tuner gain in dB |
| `--threshold` | Anomaly threshold, 0 to 1 |
| `--strict-threshold` | Require stronger SNR for borderline scores |
| `--dc-exclude-hz` | Ignore the LO/DC region |
| `--edge-exclude-hz` | Ignore Nyquist-edge artefacts |
| `--novelty` | Suppress repeat alerts for stable signals |

## Autonomous operation

```bash
# Continuous single band
spectrum-alert monitor autonomous --freq-start 144 --freq-end 148 \
  --continuous-learning

# Rotate between bands
spectrum-alert monitor autonomous-multiband \
  -b 144-148 -b 430-440 \
  --data-minutes 5 --monitor-minutes 15

# Finite number of rotations
spectrum-alert monitor autonomous-multiband \
  -b 144-148 -b 430-440 --rounds 3
```

## Training

```bash
# Train a full-feature model using the last seven days
spectrum-alert train models --mode full --days 7

# Train a lite model
spectrum-alert train models --mode lite --days 3
```

Training reads `data/features/*_<mode>.csv`, requires at least ten usable rows,
and writes model artifacts under `models/`. The web **Training** panel exposes
the same operation with additional model types and deployment controls.

## System

```bash
spectrum-alert system status
```

Reports CPU, memory, disk and receiver availability.

## Geolocation

```bash
# Receiver roster
spectrum-alert geo receivers

# Add a local receiver
spectrum-alert geo add-receiver Home --lat 35.130 --lon 33.310 \
  --gain 30 --ppm 20

# Add a remote rtl_tcp receiver
spectrum-alert geo add-receiver Hill --lat 35.080 --lon 33.380 \
  --mode rtltcp --host 10.0.0.5 --port 1234

# One scan
spectrum-alert geo scan 446100000 --span 200000 -i 0.5

# Locate an emitter
spectrum-alert geo triangulate 446100000

# Calibration
spectrum-alert geo calibrate
spectrum-alert geo calibrate --apply

# Fit path-loss exponent
spectrum-alert geo fit-ple
```

Triangulation needs at least two enabled receivers with coordinates. Two
receivers produce an underconstrained fix; three or more give a usable estimate.

## Web

```bash
spectrum-alert web --host 0.0.0.0 --port 8000
```

The default dashboard port is 8000. The compose file maps host port 8088 to it.

## Terminal and GUI together

The two interfaces can share data. A typical workflow:

1. Collect captures from the CLI.
2. Open the dashboard and review them under **Config → Data Management**.
3. Train or evaluate a model under **Training**.
4. Use the workspace for locating and identification.

A physical dongle cannot be open in two processes at once. Stop dashboard
monitoring before starting a CLI scan on the same device.

## Related

- [Quick Start](Quick-Start.md)
- [Web Dashboard](Web-Dashboard.md)
- [Docker](Docker.md)
- [Troubleshooting](Troubleshooting.md)
