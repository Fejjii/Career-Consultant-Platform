"""Structured logging configuration using ``structlog``."""

from __future__ import annotations

import logging
import sys

import structlog

from career_intel.security.hardening import redact_log_event


def setup_logging(log_level: str = "INFO", json_output: bool | None = None) -> None:
    """Configure structlog + stdlib logging for the whole application.

    Parameters
    ----------
    log_level:
        Root log level (DEBUG, INFO, WARNING, ERROR).
    json_output:
        If *True*, emit JSON lines.  If *None*, auto-detect: JSON in
        non-development environments, console otherwise.
    """
    if json_output is None:
        import os

        json_output = os.getenv("ENVIRONMENT", "development") != "development"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redact_log_event,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level.upper())

    # Quiet noisy libraries
    for lib in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
