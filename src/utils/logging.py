"""Structured (JSON-lines) logging with an optional run_id context."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from pathlib import Path

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "run_id", None) or _run_id.get()
        if rid:
            payload["run_id"] = rid
        extra = getattr(record, "data", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = getattr(record, "run_id", None) or _run_id.get()
        extra = getattr(record, "data", None)
        tail = " " + " ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
        prefix = f"[{rid}] " if rid else ""
        return f"{time.strftime('%H:%M:%S')} {record.levelname[0]} {prefix}{record.getMessage()}{tail}"


_configured = False


def setup_logging(level: str = "INFO", json_file: str | Path | None = None) -> None:
    global _configured
    root = logging.getLogger("entiq")
    root.setLevel(level)
    if not _configured:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(HumanFormatter())
        root.addHandler(h)
        _configured = True
    if json_file:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(json_file, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"entiq.{name}")


def log_event(logger: logging.Logger, msg: str, **data) -> None:
    logger.info(msg, extra={"data": data})


def set_run_id(run_id: str | None):
    return _run_id.set(run_id)
