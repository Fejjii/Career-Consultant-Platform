FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system -e ".[dev]"

COPY src/ src/
COPY streamlit_app/ streamlit_app/

EXPOSE 8000

CMD ["uvicorn", "career_intel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
