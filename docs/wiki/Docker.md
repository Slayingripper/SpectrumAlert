# Docker

The image runs as the non-root `spectrum` user. USB access is granted with
`devices:` and `group_add: plugdev`; privileged mode is not required.

## Start the dashboard

```bash
docker compose up -d
```

Open <http://localhost:8088>.

The container listens on port 8000 internally. Port 8088 is used on the host
because 8000 is commonly occupied by other services. Override it with:

```bash
SPECTRUM_ALERT_WEB_PORT=9000 docker compose up -d
```

Check status:

```bash
docker compose ps
docker compose logs -f spectrumalert
```

## Run a CLI command in a container

The default image command is the dashboard, but the entry point is still the
`spectrum-alert` CLI:

```bash
docker compose run --rm spectrumalert monitor status
docker compose run --rm spectrumalert system status
docker compose run --rm spectrumalert geo receivers
docker compose run --rm spectrumalert monitor start \
  --freq-start 144 --freq-end 148 --duration 30
```

Use a separate profile for long-running headless operation:

```bash
docker compose --profile service up spectrumalert-service
docker compose --profile autonomous up spectrumalert-autonomous
```

## Services

| Service | Purpose |
| --- | --- |
| `spectrumalert` | Web dashboard (default) |
| `spectrumalert-service` | Continuous monitoring profile |
| `spectrumalert-autonomous` | Collection + training + monitoring profile |

## USB access

The compose file maps the USB bus and adds the container user to `plugdev`:

```yaml
devices:
  - /dev/bus/usb:/dev/bus/usb
group_add:
  - plugdev
```

Verify the container sees the dongle:

```bash
docker compose exec spectrumalert rtl_test -t
```

If the host cannot see the dongle either, see
[Troubleshooting](Troubleshooting.md).

## Persistent data

The stack mounts the checkout's data, model, log and configuration directories:

```text
./data    -> /app/data
./models  -> /app/models
./logs    -> /app/logs
./config  -> /app/config
```

Training artifacts created in the container are therefore visible on the host.

## Synthetic IQ inside the container

```bash
docker compose run --rm --no-deps --entrypoint python spectrumalert \
  scripts/generate_iq_samples.py \
  --data-dir /tmp/iq/data \
  --raw-dir /tmp/iq/raw
```

For persistent fixtures, point `--data-dir` at `/app/data` and `--raw-dir` at a
mounted directory.

## Health check

The dashboard service checks the HTTP endpoint every 30 seconds. A healthy state
is shown by:

```bash
docker compose ps
```

Example:

```text
spectrum-alert   Up 2 minutes (healthy)   0.0.0.0:8088->8000/tcp
```

## Rebuild after changes

```bash
docker compose build
docker compose up -d
```

## Related

- [Installation](Installation.md)
- [CLI Reference](CLI-Reference.md)
- [Troubleshooting](Troubleshooting.md)
