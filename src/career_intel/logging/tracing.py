"""LangSmith tracing configuration and helpers."""

from __future__ import annotations

import os

import structlog

from career_intel.config import get_settings

logger = structlog.get_logger()


def configure_langsmith() -> bool:
    """Set LangSmith environment variables from settings.

    Returns True if tracing was enabled, False otherwise.
    """
    settings = get_settings()

    if not settings.langchain_tracing_v2:
        logger.info("langsmith_disabled")
        return False

    if not settings.langchain_api_key:
        logger.warning("langsmith_no_api_key")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    logger.info("langsmith_enabled", project=settings.langchain_project)
    return True
