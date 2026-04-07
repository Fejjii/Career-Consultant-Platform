"""Backend CV upload and processing endpoint.

Moves canonical CV processing to the server so the backend controls
validation, size limits, extraction quality, and risk scoring.
The frontend sends the raw file; the backend returns processed text
and a risk assessment.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import Field

from career_intel.api.deps import SettingsDep
from career_intel.schemas.domain import CVRiskScore
from career_intel.security.sanitize import score_cv_risk
from career_intel.services.cv_processor import (
    CVProcessingError,
    process_cv,
)

router = APIRouter(prefix="/cv", tags=["cv"])
logger = structlog.get_logger()


class CVProcessResponse(CVRiskScore):
    """Response from the CV processing endpoint.

    Extends ``CVRiskScore`` with the processed text and metadata.
    """

    cv_text: str
    filename: str
    warnings: list[str] = Field(default_factory=list)


@router.post(
    "/process",
    response_model=CVProcessResponse,
    responses={
        400: {"description": "Invalid file or processing error"},
        413: {"description": "File too large"},
    },
)
async def process_cv_upload(
    file: UploadFile,
    settings: SettingsDep,
) -> CVProcessResponse:
    """Accept a CV file upload, process it server-side, and return the result.

    The response includes the cleaned text, a risk score, and any warnings.
    The caller can then include ``cv_text`` in subsequent chat requests.
    """
    filename = file.filename or "unknown"
    logger.info("cv_upload_received", filename=filename, content_type=file.content_type)

    data = await file.read()

    try:
        cv_text = process_cv(
            data,
            filename,
            max_file_bytes=settings.max_cv_file_bytes,
        )
    except CVProcessingError as exc:
        status = 413 if "too large" in str(exc).lower() else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    risk = score_cv_risk(cv_text)

    warnings: list[str] = []
    if risk.flagged:
        warnings.append(
            f"CV scored {risk.score:.2f} risk — "
            f"matched: {', '.join(risk.matched_patterns)}. "
            "Potentially suspicious content detected."
        )
        logger.warning(
            "cv_risk_flagged_at_upload",
            filename=filename,
            risk_score=risk.score,
            patterns=risk.matched_patterns,
        )

    logger.info(
        "cv_upload_processed",
        filename=filename,
        text_chars=len(cv_text),
        risk_score=risk.score,
    )

    return CVProcessResponse(
        cv_text=cv_text,
        filename=filename,
        score=risk.score,
        matched_patterns=risk.matched_patterns,
        flagged=risk.flagged,
        warnings=warnings,
    )
