"""
Base data storage classes and interfaces
"""

import os
import json
import csv
import logging
import tempfile
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from spectrum_alert.core.exceptions import StorageError
from spectrum_alert.core.domain.models import SpectrumData, AnomalyDetection, FeatureVector

logger = logging.getLogger(__name__)

#: Filename date layouts to recognise, longest/most specific first.
_DATE_PATTERNS = ("%Y%m%d_%H%M%S", "%Y%m%d")


def _parse_date_from_filename(name: str) -> Optional[datetime]:
    """Extract the embedded date from ``spectrum_20250809_101621_ab12cd34.json``.

    Prefers the filename date over mtime so retention survives copying the data
    directory between machines.
    """
    stem = os.path.splitext(name)[0]
    for pattern in _DATE_PATTERNS:
        for token in stem.split("_"):
            if len(token) < len(pattern.replace("%Y", "0000").replace("%m", "00").replace("%d", "00").replace("_", "")):
                continue
            try:
                return datetime.strptime(token, pattern)
            except ValueError:
                continue
    return None


#: Samples persisted per spectrum file. Beyond this the JSON cost grows faster
#: than the value: the detector only ever looks at a window around the peak.
DEFAULT_MAX_STORED_SAMPLES = 1000

#: Schema version written into every spectrum file.
STORAGE_SCHEMA_VERSION = 2


class DataStorage:
    """Handles data storage operations for SpectrumAlert.

    Writes are atomic (temp file in the destination directory, then
    ``os.replace``) and guarded by a process-wide lock. The previous
    implementation wrote in place, so a crash mid-write left unparsable JSON, and
    the feature-CSV path truncated a file whose own append handle was still open.
    """

    def __init__(self, data_dir: str = "data", max_stored_samples: int = DEFAULT_MAX_STORED_SAMPLES):
        self.data_dir = data_dir
        self.max_stored_samples = int(max_stored_samples)
        self._write_lock = threading.RLock()
        self._dirs_ready = False
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure data directories exist (once per instance)."""
        if self._dirs_ready:
            return
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "spectrum"),
            os.path.join(self.data_dir, "anomalies"),
            os.path.join(self.data_dir, "features"),
            os.path.join(self.data_dir, "logs")
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        self._dirs_ready = True

    def _atomic_write_json(self, filepath: str, payload: Dict[str, Any]) -> None:
        """Write JSON so readers never observe a partial file."""
        directory = os.path.dirname(os.path.abspath(filepath)) or "."
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".tmp-", suffix=".json", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, filepath)
        except OSError as e:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise StorageError(f"Failed to write {filepath}: {e}") from e

    def save_spectrum_data(self, spectrum_data: SpectrumData) -> str:
        """Save spectrum data to file with optimized storage"""
        try:
            filename = f"spectrum_{spectrum_data.timestamp.strftime('%Y%m%d_%H%M%S')}_{spectrum_data.id[:8]}.json"
            filepath = os.path.join(self.data_dir, "spectrum", filename)

            limit = self.max_stored_samples
            stored = spectrum_data.samples[:limit]
            # Record how many samples were kept and what the true capture length
            # was. Previously duration_seconds was recomputed on load from the
            # truncated count, which silently reported a 2 ms capture as 0.5 ms.
            data = {
                'schema_version': STORAGE_SCHEMA_VERSION,
                'id': spectrum_data.id,
                'timestamp': spectrum_data.timestamp.isoformat(),
                'frequency_hz': spectrum_data.frequency_hz,
                'sample_rate_hz': spectrum_data.sample_rate_hz,
                'gain_db': spectrum_data.gain_db,
                'sample_count': len(spectrum_data.samples),
                'stored_sample_count': len(stored),
                'duration_seconds': spectrum_data.duration_seconds,
                'i_samples': [float(x.real) for x in stored],
                'q_samples': [float(x.imag) for x in stored],
            }
            if spectrum_data.power_spectrum:
                # Bounded: a full-resolution power spectrum can be hundreds of
                # thousands of floats and would dwarf the stored IQ.
                data['power_spectrum'] = [float(v) for v in spectrum_data.power_spectrum[:2048]]

            with self._write_lock:
                self._atomic_write_json(filepath, data)

            logger.debug(f"Saved spectrum data to {filepath}")
            return filepath
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error saving spectrum data: {e}")
            raise StorageError(f"Failed to save spectrum data: {e}")

    def load_spectrum_data(self, filepath: str) -> SpectrumData:
        """Load spectrum data from file.

        Accepts both the current schema (``i_samples``/``q_samples``) and the
        legacy one (``samples_real``/``samples_imag``). The previous reader only
        understood the current keys, which made the 706 archived files in
        ``data/spectrum`` unreadable.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'i_samples' in data:
                i_samples = data['i_samples']
                q_samples = data['q_samples']
            elif 'samples_real' in data:
                # Legacy schema written by the earlier version.
                i_samples = data['samples_real']
                q_samples = data['samples_imag']
            else:
                raise StorageError(
                    f"Unrecognised spectrum schema in {filepath}; "
                    f"keys={sorted(data.keys())}"
                )

            iq_samples = [complex(i, q) for i, q in zip(i_samples, q_samples)]

            sample_rate = data['sample_rate_hz']
            # Prefer the recorded duration; fall back to the true sample count,
            # then to the stored count for legacy files that recorded neither.
            duration = data.get('duration_seconds')
            if not duration:
                total_samples = data.get('sample_count') or len(iq_samples)
                duration = total_samples / sample_rate

            return SpectrumData(
                id=data['id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                frequency_hz=data['frequency_hz'],
                sample_rate_hz=sample_rate,
                gain_db=data['gain_db'],
                samples=iq_samples,
                power_spectrum=data.get('power_spectrum'),
                duration_seconds=duration
            )
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error loading spectrum data from {filepath}: {e}")
            raise StorageError(f"Failed to load spectrum data: {e}")
    
    def save_anomaly(self, anomaly: AnomalyDetection) -> str:
        """Save anomaly detection result to CSV file"""
        try:
            # Create daily anomaly file
            date_str = anomaly.timestamp.strftime('%Y%m%d')
            filename = f"anomalies_{date_str}.csv"
            filepath = os.path.join(self.data_dir, "anomalies", filename)
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(filepath)
            
            with self._write_lock:
                with open(filepath, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'id', 'timestamp', 'frequency_hz', 'anomaly_type',
                        'confidence_score', 'severity', 'description',
                        'detection_mode', 'spectrum_data_id', 'metadata'
                    ])

                    if not file_exists:
                        writer.writeheader()

                    writer.writerow({
                        'id': anomaly.id,
                        'timestamp': anomaly.timestamp.isoformat(),
                        'frequency_hz': anomaly.frequency_hz,
                        'anomaly_type': anomaly.anomaly_type.value,
                        'confidence_score': anomaly.confidence_score,
                        'severity': anomaly.severity,
                        'description': anomaly.description,
                        'detection_mode': anomaly.detection_mode.value,
                        'spectrum_data_id': anomaly.spectrum_data_id,
                        'metadata': json.dumps(anomaly.metadata)
                    })
            
            logger.debug(f"Saved anomaly to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving anomaly: {e}")
            raise StorageError(f"Failed to save anomaly: {e}")
    
    def get_anomalies_since(self, since: datetime, until: Optional[datetime] = None) -> List[AnomalyDetection]:
        """Get anomalies since a specific datetime.

        Uses the stdlib ``csv`` module rather than pandas: this runs on every
        dashboard refresh, and per-row parsing is fast enough while avoiding a
        heavyweight dependency on the read path.
        """
        from spectrum_alert.core.domain.models import AnomalyType, DetectionMode

        try:
            anomalies: List[AnomalyDetection] = []
            anomalies_dir = os.path.join(self.data_dir, "anomalies")

            if not os.path.isdir(anomalies_dir):
                return anomalies

            # Newest files first so an early cutoff can stop the scan.
            filenames = sorted(
                (n for n in os.listdir(anomalies_dir)
                 if n.startswith("anomalies_") and n.endswith(".csv")),
                reverse=True,
            )

            for filename in filenames:
                filepath = os.path.join(anomalies_dir, filename)
                try:
                    with open(filepath, "r", newline="", encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            raw_ts = row.get("timestamp")
                            if not raw_ts:
                                continue
                            try:
                                timestamp = datetime.fromisoformat(raw_ts)
                            except ValueError:
                                continue
                            if timestamp < since:
                                continue
                            if until is not None and timestamp > until:
                                continue
                            raw_meta = row.get("metadata")
                            try:
                                metadata = json.loads(raw_meta) if raw_meta else {}
                            except ValueError:
                                metadata = {}
                            anomalies.append(
                                AnomalyDetection(
                                    id=row.get("id"),
                                    timestamp=timestamp,
                                    frequency_hz=float(row["frequency_hz"]),
                                    anomaly_type=AnomalyType(row["anomaly_type"]),
                                    confidence_score=float(row["confidence_score"]),
                                    severity=row["severity"],
                                    description=row.get("description", ""),
                                    detection_mode=DetectionMode(row["detection_mode"]),
                                    spectrum_data_id=row.get("spectrum_data_id"),
                                    metadata=metadata,
                                )
                            )
                except (OSError, KeyError, ValueError) as e:
                    logger.warning("Error reading anomaly file %s: %s", filepath, e)
                    continue

            anomalies.sort(key=lambda x: x.timestamp, reverse=True)
            return anomalies

        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            raise StorageError(f"Failed to get anomalies: {e}")
    
    def save_feature_vector(self, feature_vector: FeatureVector) -> str:
        """Save feature vector to a per-day, per-mode CSV file.

        Split by mode deliberately: LITE produces 2 features and FULL produces 12.
        Writing both into one file produced sparse columns that silently
        contaminated the training matrix with NaNs, because LITE rows simply had
        no value for the FULL-only columns.

        The whole read-modify-write runs under the instance lock and the rewrite
        goes through a temp file. The original code held an append handle open
        while truncating and rewriting the same path, which could interleave
        buffered data into the middle of the new content.
        """
        try:
            date_str = feature_vector.timestamp.strftime('%Y%m%d')
            mode = feature_vector.mode.value if hasattr(feature_vector.mode, 'value') else str(feature_vector.mode)
            filename = f"features_{date_str}_{mode}.csv"
            filepath = os.path.join(self.data_dir, "features", filename)

            row = {
                'id': feature_vector.id,
                'timestamp': feature_vector.timestamp.isoformat(),
                'spectrum_data_id': feature_vector.spectrum_data_id,
                'mode': mode,
                **feature_vector.features
            }

            with self._write_lock:
                existing_fieldnames: List[str] = []
                existing_rows: List[Dict[str, Any]] = []

                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r', newline='', encoding='utf-8') as read_f:
                            reader = csv.DictReader(read_f)
                            existing_fieldnames = list(reader.fieldnames or [])
                            existing_rows = [dict(r) for r in reader]
                    except (OSError, csv.Error) as e:
                        logger.warning("Could not read %s (%s); starting a new file", filepath, e)
                        existing_fieldnames = []
                        existing_rows = []

                if not existing_fieldnames:
                    # Fresh file: append under the lock so a concurrent first
                    # writer cannot interleave a header into the middle.
                    with open(filepath, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()
                        writer.writerow(row)
                else:
                    all_fieldnames = list(dict.fromkeys(existing_fieldnames + list(row.keys())))
                    if list(row.keys()) == existing_fieldnames:
                        # Same schema: plain append, no rewrite needed.
                        with open(filepath, 'a', newline='', encoding='utf-8') as f:
                            csv.DictWriter(f, fieldnames=all_fieldnames).writerow(row)
                    else:
                        # Schema changed (a feature set grew). Rewrite atomically.
                        existing_rows.append(row)
                        temp_path = f"{filepath}.tmp.{os.getpid()}"
                        try:
                            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                                writer.writeheader()
                                for existing in existing_rows:
                                    writer.writerow({
                                        k: existing.get(k, '') for k in all_fieldnames
                                    })
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(temp_path, filepath)
                        except OSError:
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass
                            raise

            logger.debug(f"Saved feature vector to {filepath}")
            return filepath
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error saving feature vector: {e}")
            raise StorageError(f"Failed to save feature vector: {e}")

    def save_features(self, feature_vector: FeatureVector) -> str:
        """Alias for :meth:`save_feature_vector`.

        The use-case layer calls ``save_features`` while the storage layer named
        it ``save_feature_vector``; that mismatch raised ``AttributeError`` and
        broke the entire monitoring loop. Keeping both names avoids breaking any
        caller still using either.
        """
        return self.save_feature_vector(feature_vector)

    def load_training_data(
        self,
        mode: Optional[str] = None,
        days: Optional[int] = None,
        max_rows: Optional[int] = None,
        data_dir: Optional[str] = None,
    ) -> Any:
        """Load feature rows as a DataFrame for training.

        Only same-mode files are read. Mixing LITE and FULL rows produced NaNs in
        the FULL-only columns, which then propagated through ``StandardScaler``
        into the fitted model.

        ``data_dir`` defaults to this instance's ``features`` directory.
        """
        import pandas as pd

        features_dir = data_dir or os.path.join(self.data_dir, "features")
        if not os.path.isdir(features_dir):
            logger.warning("features directory %s does not exist", features_dir)
            return pd.DataFrame()

        suffix = f"_{mode}" if mode else ""
        paths = sorted(
            os.path.join(features_dir, name)
            for name in os.listdir(features_dir)
            if name.startswith("features_")
            and name.endswith(".csv")
            and (not suffix or name.endswith(f"{suffix}.csv"))
        )
        if days is not None:
            cutoff = datetime.now() - timedelta(days=int(days))
            kept = []
            for path in paths:
                stamp = _parse_date_from_filename(os.path.basename(path))
                if stamp is None or stamp >= cutoff:
                    kept.append(path)
            paths = kept

        if not paths:
            return pd.DataFrame()

        frames = []
        for path in paths:
            try:
                frame = pd.read_csv(path)
            except Exception as e:
                logger.warning("Skipping unreadable feature file %s: %s", path, e)
                continue
            if not frame.empty:
                frames.append(frame)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        # Drop all-empty columns, which is how a partially-populated historical
        # file shows up after a mode switch.
        combined = combined.dropna(axis=1, how="all")
        combined = combined.dropna(axis=0, how="any")
        if max_rows is not None and len(combined) > int(max_rows):
            combined = combined.tail(int(max_rows))
        return combined
    
    def get_feature_files(self) -> List[str]:
        """Get all feature CSV files"""
        try:
            features_dir = os.path.join(self.data_dir, "features")
            if not os.path.exists(features_dir):
                return []
            
            files = []
            for filename in os.listdir(features_dir):
                if filename.startswith("features_") and filename.endswith(".csv"):
                    files.append(os.path.join(features_dir, filename))
            
            return sorted(files)
        except Exception as e:
            logger.error(f"Error getting feature files: {e}")
            raise StorageError(f"Failed to get feature files: {e}")
    
    def get_spectrum_files(self) -> List[str]:
        """Get all spectrum JSON files"""
        try:
            spectrum_dir = os.path.join(self.data_dir, "spectrum")
            if not os.path.exists(spectrum_dir):
                return []
            
            files = []
            for filename in os.listdir(spectrum_dir):
                if filename.startswith("spectrum_") and filename.endswith(".json"):
                    files.append(os.path.join(spectrum_dir, filename))
            
            return sorted(files)
        except Exception as e:
            logger.error(f"Error getting spectrum files: {e}")
            raise StorageError(f"Failed to get spectrum files: {e}")
    
    def get_recent_anomalies(self, limit: int = 50) -> List[AnomalyDetection]:
        """Most recent anomalies, newest first."""
        anomalies = self.get_anomalies_since(datetime.min)
        return anomalies[: max(1, int(limit))]

    def get_anomaly_count(self, days: int = 1) -> int:
        """Number of anomalies recorded in the last ``days`` days."""
        try:
            return len(self.get_anomalies_since(datetime.now() - timedelta(days=int(days))))
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error counting anomalies: {e}")
            raise StorageError(f"Failed to count anomalies: {e}")

    def get_anomaly_stats(self, days: int = 7) -> Dict[str, Any]:
        """Aggregate anomaly counts by severity, type and detection mode."""
        anomalies = self.get_anomalies_since(datetime.now() - timedelta(days=int(days)))
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_mode: Dict[str, int] = {}
        for anomaly in anomalies:
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1
            type_value = getattr(anomaly.anomaly_type, "value", str(anomaly.anomaly_type))
            by_type[type_value] = by_type.get(type_value, 0) + 1
            mode_value = getattr(anomaly.detection_mode, "value", str(anomaly.detection_mode))
            by_mode[mode_value] = by_mode.get(mode_value, 0) + 1
        return {
            "days": int(days),
            "total": len(anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "by_mode": by_mode,
        }

    def cleanup_old_data(self, days_to_keep: int = 30, dry_run: bool = False) -> Dict[str, Any]:
        """Delete data files older than ``days_to_keep`` days.

        Age comes from the timestamp embedded in each filename, not from
        ``st_mtime``. A clone, rsync or container copy rewrites mtimes, which made
        the previous implementation either delete everything or nothing at a
        stroke.

        ``anomalies/`` is now included; it was never cleaned, so it grew without
        bound even though it is the only output the dashboard reads.

        Returns a summary of what was removed, rather than ``None``, which the
        caller previously formatted into the log as the string "None".
        """
        try:
            cutoff = datetime.now() - timedelta(days=int(days_to_keep))
            removed: List[str] = []
            kept = 0

            for directory, matcher in (
                (os.path.join(self.data_dir, "spectrum"), lambda n: n.startswith("spectrum_") and n.endswith(".json")),
                (os.path.join(self.data_dir, "features"), lambda n: n.startswith("features_") and n.endswith(".csv")),
                (os.path.join(self.data_dir, "anomalies"), lambda n: n.startswith("anomalies_") and n.endswith(".csv")),
            ):
                if not os.path.isdir(directory):
                    continue
                for name in os.listdir(directory):
                    if not matcher(name):
                        continue
                    path = os.path.join(directory, name)
                    stamp = _parse_date_from_filename(name)
                    # Unknown date: fall back to mtime rather than deleting blind.
                    if stamp is None:
                        try:
                            stamp = datetime.fromtimestamp(os.stat(path).st_mtime)
                        except OSError:
                            continue
                    if stamp >= cutoff:
                        kept += 1
                        continue
                    if dry_run:
                        removed.append(path)
                        continue
                    try:
                        os.remove(path)
                        removed.append(path)
                    except OSError as e:
                        logger.warning("Could not remove %s: %s", path, e)

            summary = {
                "removed": len(removed),
                "kept": kept,
                "removed_paths": removed,
                "days_to_keep": int(days_to_keep),
                "dry_run": bool(dry_run),
            }
            logger.info(
                "Cleanup %s: removed %d file(s), kept %d (cutoff %s)",
                "preview" if dry_run else "complete",
                len(removed),
                kept,
                cutoff.isoformat(),
            )
            return summary
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise StorageError(f"Failed to cleanup old data: {e}")
