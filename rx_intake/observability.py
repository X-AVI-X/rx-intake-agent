"""Structured JSON logs with a trace id per request.

Patient text is never logged. Only a hash and length are recorded, so the
logs can be shipped to any log platform without leaking PHI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("rx_intake")


def configure(level: str = "INFO") -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def event(name: str, trace_id: str, **fields: object) -> None:
    logger.info(json.dumps({"event": name, "trace_id": trace_id, "ts": round(time.time(), 3), **fields}, default=str))


@contextmanager
def timed() -> Iterator[dict]:
    box = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = round((time.perf_counter() - start) * 1000, 1)
