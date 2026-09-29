"""Simple on-disk JSON cache keyed by a hash of the request."""

import hashlib
import json
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
            return None
        return payload["value"]

    def set(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps({"stored_at": time.time(), "value": value}))
        tmp.replace(self._path(key))
