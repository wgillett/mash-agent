"""Simple on-disk JSON cache keyed by a hash of the request."""

import contextlib
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class DiskCache:
    """Stores JSON-serializable values as files under ``directory`` with a TTL."""

    def __init__(self, directory: Path, ttl_seconds: float = 24 * 3600) -> None:
        self.directory = directory
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if time.time() - payload["stored_at"] > self.ttl_seconds:
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        return payload["value"]

    def set(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"stored_at": time.time(), "value": value})
        # Unique temp file per writer so concurrent sets of one key cannot clobber each other.
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=f"{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp_name, self._path(key))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
