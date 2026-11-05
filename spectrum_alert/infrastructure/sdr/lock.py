"""Serialisation access to SDR hardware.

An RTL-SDR is a single USB device that cannot be retuned while a transfer is in
flight. Two threads or two processes touching the same dongle produce garbled
spectra or ``LIBUSB_ERROR_ACCESS`` at best, and silently wrong data at worst.

Two mechanisms are provided, because they solve different halves of the problem:

* :class:`DeviceLock` is an **in-process** mutex keyed by device identity. It
  serialises threads inside one Python process. It is an ``RLock`` reentrant per
  key, so a high-level routine can call a lower-level one that also locks the
  same device without deadlocking.

* :func:`device_file_lock` is a **cross-process** advisory lock (flock on
  POSIX) held on a lock file. ``flock`` is released automatically when the file
  descriptor closes, including on ``SIGKILL``, so a crashed process cannot leave
  a dongle permanently wedged.

Use both together when several processes may monitor the same receiver.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceLock",
    "DeviceLockRegistry",
    "device_file_lock",
    "LOCK_DIR_ENV",
]

LOCK_DIR_ENV = "SPECTRUM_ALERT_LOCK_DIR"

#: Directory holding cross-process lock files.
DEFAULT_LOCK_DIR = "/tmp/spectrum_alert_locks"

_registry_lock = threading.Lock()
_registries: Dict[str, "DeviceLockRegistry"] = {}

# Per-process state for cross-process file locks.
#
# flock is per open-file-description, not per process: opening the same lock file
# twice in one process yields two descriptions, and the second flock(LOCK_EX)
# blocks against the first, deadlocking any nested acquisition. Depth tracking
# here makes the file lock reentrant within a process while preserving
# cross-process mutual exclusion (which is the whole point of using flock).
_file_lock_guard = threading.Lock()
_file_lock_state: Dict[str, "_FileLockState"] = {}


class _FileLockState:
    """Holds one process's claim on a lock file."""

    def __init__(self, handle, key: str):
        self.handle = handle
        self.key = key
        self.depth = 0


class DeviceLock:
    """Reentrant mutex for one device key."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
        self._depth = 0
        self._owner: Optional[int] = None

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire, honouring reentrancy for the owning thread."""
        thread_id = threading.get_ident()
        with self._lock:
            if self._depth > 0 and self._owner == thread_id:
                self._depth += 1
                return True

        acquired = self._lock.acquire(timeout=timeout) if timeout is not None else self._lock.acquire()
        if not acquired:
            return False
        self._depth = 1
        self._owner = thread_id
        return True

    def release(self) -> None:
        # Only the outermost release() may drop the underlying RLock; an inner
        # one just decrements the depth. Releasing the RLock unconditionally
        # would unbalance the recursion count and let a second thread in.
        release_underlying = False
        with self._lock:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                release_underlying = True
        if release_underlying:
            self._lock.release()

    def __enter__(self) -> "DeviceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


class DeviceLockRegistry:
    """Hands out one :class:`DeviceLock` per device key."""

    def __init__(self) -> None:
        self._locks: Dict[str, DeviceLock] = {}
        self._guard = threading.Lock()

    def for_key(self, key: str) -> DeviceLock:
        normalized = str(key)
        with self._guard:
            lock = self._locks.get(normalized)
            if lock is None:
                lock = DeviceLock(normalized)
                self._locks[normalized] = lock
            return lock

    def lock_for(self, key: str, timeout: Optional[float] = None) -> DeviceLock:
        """Return an acquired lock. Prefer using the registry as a context manager."""
        lock = self.for_key(key)
        if not lock.acquire(timeout=timeout):
            raise TimeoutError(f"timed out acquiring SDR lock for {key!r}")
        return lock


def get_registry(name: str = "default") -> DeviceLockRegistry:
    """Fetch (or create) a named registry.

    Sharing one registry across a process is the point: two call sites asking for
    ``"usb:0"`` get the same lock object.
    """
    with _registry_lock:
        registry = _registries.get(name)
        if registry is None:
            registry = DeviceLockRegistry()
            _registries[name] = registry
        return registry


def lock_dir() -> str:
    """Directory for cross-process lock files."""
    return os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR


@contextmanager
def device_file_lock(
    key: str, timeout: Optional[float] = 30.0, poll_interval: float = 0.1
) -> Iterator[None]:
    """Hold an advisory cross-process lock for ``key``.

    Reentrant within a process: a nested acquisition of the same key reuses this
    process's existing claim instead of trying to ``flock`` it again (which would
    deadlock, because flock is per open-file-description rather than per
    process). Cross-process exclusion is unaffected.

    Yields once the lock is held; raises ``TimeoutError`` if ``timeout`` elapses.

    A degraded fallback matters here: some filesystems do not support ``flock``,
    and losing cross-process safety is much better than losing the ability to
    scan at all. The warning says so explicitly.
    """
    normalized = str(key)

    # Claim an existing in-process hold, if there is one. The guard must NOT be
    # held across the yield: a nested acquisition would then block on the very
    # guard this call is holding, deadlocking.
    with _file_lock_guard:
        state = _file_lock_state.get(normalized)
        if state is not None:
            state.depth += 1

    if state is not None:
        try:
            yield
        finally:
            with _file_lock_guard:
                state.depth -= 1
        return

    directory = lock_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "lock dir %s unavailable (%s); continuing without cross-process lock",
            directory, exc,
        )
        yield
        return

    safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in normalized)[:120]
    path = os.path.join(directory, f"{safe_key}.lock")

    try:
        handle = open(path, "a+")
    except OSError as exc:
        logger.warning(
            "cannot open lock file %s (%s); continuing without cross-process lock", path, exc
        )
        yield
        return

    try:
        import fcntl

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        locked = True
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    logger.warning(
                        "flock unsupported on %s (%s); continuing unlocked", path, exc
                    )
                    locked = False
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for cross-process lock on {normalized!r}"
                    ) from exc
                time.sleep(poll_interval)

        state = _FileLockState(handle=handle, key=normalized)
        state.depth = 1
        with _file_lock_guard:
            _file_lock_state[normalized] = state

        try:
            yield
        finally:
            with _file_lock_guard:
                if _file_lock_state.get(normalized) is state:
                    del _file_lock_state[normalized]
            if locked:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


def hold_device(
    key: str,
    registry: Optional[DeviceLockRegistry] = None,
    file_lock: bool = True,
    timeout: Optional[float] = 30.0,
):
    """Combined in-process + cross-process lock context manager."""
    registry = registry or get_registry()
    return _CombinedDeviceLock(key, registry, file_lock, timeout)


class _CombinedDeviceLock:
    """Context manager pairing a :class:`DeviceLock` with an optional file lock."""

    def __init__(
        self,
        key: str,
        registry: DeviceLockRegistry,
        file_lock: bool,
        timeout: Optional[float],
    ) -> None:
        self.key = key
        self.registry = registry
        self.file_lock = file_lock
        self.timeout = timeout
        self._device_lock: Optional[DeviceLock] = None
        self._file_ctx = None

    def __enter__(self) -> "DeviceLock":
        self._device_lock = self.registry.lock_for(self.key, timeout=self.timeout)
        if self.file_lock:
            self._file_ctx = device_file_lock(self.key, timeout=self.timeout)
            self._file_ctx.__enter__()
        return self._device_lock

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file_ctx is not None:
            self._file_ctx.__exit__(exc_type, exc, tb)
            self._file_ctx = None
        if self._device_lock is not None:
            self._device_lock.release()
            self._device_lock = None