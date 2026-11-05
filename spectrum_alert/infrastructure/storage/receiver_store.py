"""JSON-backed persistence for the receiver roster.

Atomic writes and a process-wide lock are the point of this module. The original
in-memory-plus-``json.dump`` approach lost the entire roster if the process died
mid-write, and two concurrent saves could interleave into a corrupt file that
then failed to parse on restart.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import ValidationError

from spectrum_alert.core.domain.receivers import SDRReceiver

logger = logging.getLogger(__name__)

__all__ = ["ReceiverStore", "JsonFileStore"]

#: Bumped when the on-disk shape changes incompatibly.
STORE_VERSION = 2


class JsonFileStore:
    """Small helper for atomic JSON reads/writes under a lock."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()

    def read(self) -> Dict:
        """Load the document, returning ``{}`` when absent or unreadable.

        A corrupt file is quarantined rather than deleted: losing the receiver
        roster silently would be worse than leaving evidence behind.
        """
        if not os.path.exists(self.path):
            return {}
        with self._lock:
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data if isinstance(data, dict) else {}
            except (OSError, ValueError) as exc:
                logger.error("could not read %s: %s", self.path, exc)
                self._quarantine()
                return {}

    def write(self, payload: Dict) -> None:
        """Write ``payload`` atomically.

        Temp file in the same directory, fsync, then ``os.replace``. A reader
        therefore sees either the old document or the new one, never a partial
        write.
        """
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=directory,
                prefix=".receivers-",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    json.dump(payload, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(handle.name, self.path)
            except OSError as exc:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
                raise OSError(f"failed to persist {self.path}: {exc}") from exc

    def _quarantine(self) -> None:
        try:
            os.replace(self.path, f"{self.path}.corrupt")
            logger.warning("moved corrupt file to %s.corrupt", self.path)
        except OSError:
            pass


class ReceiverStore:
    """The receiver roster on disk.

    Satisfies :class:`~spectrum_alert.core.interfaces.ports.ReceiverStorePort`.
    """

    def __init__(self, path: str):
        self._store = JsonFileStore(path)
        self._receivers: Dict[str, SDRReceiver] = {}
        self.load()

    @property
    def path(self) -> str:
        return self._store.path

    def load(self) -> None:
        """(Re)read the roster from disk, keeping entries that still validate."""
        document = self._store.read()
        raw_list = document.get("receivers")
        receivers: Dict[str, SDRReceiver] = {}
        if isinstance(raw_list, list):
            for entry in raw_list:
                if not isinstance(entry, dict):
                    continue
                try:
                    receiver = SDRReceiver.from_dict(entry)
                except (ValidationError, ValueError, TypeError) as exc:
                    logger.warning("dropping invalid receiver entry %r: %s", entry, exc)
                    continue
                receivers[receiver.id] = receiver
        self._receivers = receivers
        logger.info("loaded %d receiver(s) from %s", len(receivers), self._store.path)

    def _persist(self) -> None:
        self._store.write(
            {
                "version": STORE_VERSION,
                "updated_at": datetime.utcnow().isoformat(),
                "receivers": [r.to_dict() for r in self._receivers.values()],
            }
        )

    def list(self) -> List[SDRReceiver]:
        return list(self._receivers.values())

    def enabled(self) -> List[SDRReceiver]:
        return [r for r in self._receivers.values() if r.enabled]

    def get(self, receiver_id: str) -> Optional[SDRReceiver]:
        return self._receivers.get(str(receiver_id))

    def save(self, receiver: SDRReceiver) -> SDRReceiver:
        """Insert or update. Assigns an id and creation time when missing."""
        if not receiver.id:
            receiver.id = str(uuid.uuid4())
        if receiver.created_at is None:
            receiver = receiver.model_copy(update={"created_at": datetime.utcnow()})
        self._receivers[receiver.id] = receiver
        self._persist()
        return receiver

    def delete(self, receiver_id: str) -> bool:
        removed = self._receivers.pop(str(receiver_id), None) is not None
        if removed:
            self._persist()
        return removed

    def clear(self) -> None:
        self._receivers.clear()
        self._persist()

    def __len__(self) -> int:
        return len(self._receivers)