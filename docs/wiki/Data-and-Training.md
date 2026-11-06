# Data and Training

## Storage layout

SpectrumAlert stores all runtime data under one directory:

```text
data/
├── spectrum/          one JSON file per capture
├── features/          feature CSV files, split by detection mode
├── anomalies/         daily anomaly history
├── logs/              runtime logs
├── calibration.json   per-receiver frequency offsets
└── receivers.json     receiver roster
```

The default directory is `data` relative to the working directory. Docker maps
the checkout's `data` directory into `/app/data`.

### Spectrum files

Each capture carries I/Q samples and metadata:

```json
{
  "schema_version": 2,
  "id": "...",
  "timestamp": "...",
  "frequency_hz": 145000000.0,
  "sample_rate_hz": 2048000.0,
  "gain_db": 20.0,
  "sample_count": 8192,
  "i_samples": [],
  "q_samples": [],
  "duration_seconds": 0.004
}
```

Writes are atomic. The reader also accepts the legacy `samples_real` /
`samples_imag` schema, so archived captures remain usable.

### Feature files

Features are split by mode because the two modes have different columns:

```text
features_YYYYMMDD_lite.csv
features_YYYYMMDD_full.csv
```

Mixing modes would introduce NaNs in the full-only columns.

## Levels are dBFS

The RTL-SDR has no absolute power reference. Every level is reported in dBFS
until a per-receiver calibration offset is applied. Calibration only makes the
result as good as the reference transmitter used to derive it.

## Detection modes

| Mode | Features | Use |
| --- | --- | --- |
| `lite` | 2 | Fast, low-power devices such as a Raspberry Pi |
| `full` | 12 | Spectral shape, circular phase statistics and higher-order moments |

## Train from the dashboard

1. Open **Training**.
2. Choose a model type and dataset.
3. Set contamination and epochs where applicable.
4. Press **Train**.
5. Review model status, test the model, then deploy.

Training requires at least ten usable feature rows. **All data** uses every
feature file for the selected mode; **Recent** limits the range to 24 hours.

## Train from the CLI

```bash
spectrum-alert train models --mode full --days 7
spectrum-alert train models --mode lite --days 3
```

The command writes model artifacts to `models/` and reports the generated file.

## Synthetic IQ fixtures

`scripts/generate_iq_samples.py` creates a deterministic dataset without a
radio:

```bash
python scripts/generate_iq_samples.py \
  --data-dir /tmp/spectrumalert-demo/data \
  --raw-dir  /tmp/spectrumalert-demo/raw
```

The default set includes:

- 30 clean noise captures
- CW, two-tone, FSK and ASK signals
- One wideband negative control
- Full and lite feature rows for every capture
- Four anomaly records

The generator exits non-zero if a clean capture alerts or an expected signal is
missed.

## Testing

```bash
pytest
```

The hardware-dependent collection test requires an actual RTL-SDR. All storage,
detection, feature, web and UI-contract tests run without one.

## Related

- [Web Dashboard](Web-Dashboard.md)
- [CLI Reference](CLI-Reference.md)
- [Configuration](Configuration.md)
