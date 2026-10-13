"""JSON log + gizli bilgi maskeleme (R7)."""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|(?i:api[_-]?key)\s*[=:]\s*\S+)")


def mask(text: str) -> str:
    return _SECRET_RE.sub("***", text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": mask(record.getMessage()),
        }
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            data.update({k: mask(str(v)) if isinstance(v, str) else v for k, v in extra.items()})
        if record.exc_info:
            data["exc"] = mask(self.formatException(record.exc_info))
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_question(logger: logging.Logger, question: str | None) -> None:
    """Kullanıcı sorusu yalnızca DEBUG seviyesinde yazılır (R7)."""
    if question and logger.isEnabledFor(logging.DEBUG):
        logger.debug("question", extra={"extra_data": {"question": question}})
