"""CV text extraction, cleaning, and token-safe truncation.

Supports PDF, DOCX, and plain-text uploads.  Extracted text is normalized
and truncated to stay within a configurable token budget so it can safely
be injected into the LLM context window.

Privacy:
  Raw CV content MUST NEVER be logged.  Only metadata (filename, byte
  length, token count, risk score) is safe to emit in log events.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import structlog
import tiktoken

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
_DEFAULT_MAX_TOKENS = 3000
_DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
_ENCODING_NAME = "cl100k_base"


class CVProcessingError(Exception):
    """Raised when CV extraction or validation fails."""


def validate_cv_upload(
    data: bytes,
    filename: str,
    *,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> None:
    """Pre-flight validation before expensive parsing.

    Raises ``CVProcessingError`` if the file is invalid.
    """
    if not filename or not filename.strip():
        raise CVProcessingError("Filename is required.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise CVProcessingError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not data:
        raise CVProcessingError("Uploaded file is empty (0 bytes).")

    if len(data) > max_file_bytes:
        size_mb = round(len(data) / (1024 * 1024), 1)
        limit_mb = round(max_file_bytes / (1024 * 1024), 1)
        raise CVProcessingError(
            f"File too large ({size_mb} MB). Maximum allowed is {limit_mb} MB."
        )


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from an uploaded file's raw bytes.

    Parameters
    ----------
    data:
        Raw file bytes (must already pass ``validate_cv_upload``).
    filename:
        Original filename (used to determine the parser).

    Raises
    ------
    CVProcessingError
        If the file type is unsupported or parsing fails.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise CVProcessingError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if ext == ".pdf":
            return _extract_pdf(data)
        elif ext == ".docx":
            return _extract_docx(data)
        else:
            return data.decode("utf-8", errors="replace")
    except CVProcessingError:
        raise
    except Exception as exc:
        logger.error("cv_extraction_failed", filename=filename, error=str(exc)[:200])
        raise CVProcessingError(f"Failed to extract text from '{filename}': {exc}") from exc


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    if not pages:
        raise CVProcessingError("PDF contains no extractable text (possibly scanned/image-only).")
    return "\n\n".join(pages)


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise CVProcessingError("DOCX contains no text content.")
    return "\n".join(paragraphs)


def clean_cv_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x20-\x7E\n\t\u00A0-\uFFFF]", "", text)
    return text.strip()


def truncate_to_token_limit(text: str, max_tokens: int = _DEFAULT_MAX_TOKENS) -> str:
    """Truncate text to fit within a token budget.

    Uses tiktoken's cl100k_base encoding (used by GPT-4/GPT-4o).
    Adds a truncation notice if content is shortened.
    """
    enc = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = enc.decode(tokens[:max_tokens])
    logger.info("cv_truncated", original_tokens=len(tokens), max_tokens=max_tokens)
    return truncated + "\n\n[CV truncated to fit context window]"


def process_cv(
    data: bytes,
    filename: str,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
) -> str:
    """Full pipeline: validate -> extract -> clean -> truncate.

    Returns the processed CV text ready for prompt injection (after sanitization).
    Raw CV content is never logged — only safe metadata.
    """
    validate_cv_upload(data, filename, max_file_bytes=max_file_bytes)

    raw = extract_text_from_bytes(data, filename)
    cleaned = clean_cv_text(raw)
    if not cleaned:
        raise CVProcessingError("CV appears to be empty after text extraction.")
    truncated = truncate_to_token_limit(cleaned, max_tokens=max_tokens)

    logger.info(
        "cv_processed",
        filename=filename,
        raw_bytes=len(data),
        extracted_chars=len(cleaned),
        final_chars=len(truncated),
    )
    return truncated
