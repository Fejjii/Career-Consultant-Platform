# AI Career Intelligence Assistant

A production-grade career copilot that provides **retrieval-grounded**, **CV-aware** career guidance using trusted labor-market and skills data. Built with intent-first routing, RAG, tool calling, streaming, and multi-layer security.

---

## Architecture

```mermaid
flowchart TB
    subgraph client ["Client Layer"]
        UI["Streamlit UI<br/>(chat + CV upload + sources panel)"]
    end

    subgraph api ["API Layer (FastAPI)"]
        Chat["POST /chat"]
        Stream["POST /chat/stream (SSE)"]
        CV["POST /cv/process"]
        Health["GET /health/*"]
        Guards["Input Guards + Rate Limiting"]
    end

    subgraph orchestration ["Orchestration"]
        Router["Intent Router (LLM)"]
        Chain["Chain / Stream Engine"]
        Synth["Synthesizer"]
        CTX["Context Builder"]
        Tools["Tool Registry"]
    end

    subgraph rag ["RAG Layer"]
        Rewrite["Query Rewriter"]
        Retriever["Multi-query Retriever"]
        Chunks["Chunk Builder + Citation Map"]
    end

    subgraph data ["Data Layer"]
        Qdrant[("Qdrant")]
        Postgres[("Postgres")]
        Redis[("Redis")]
    end

    subgraph providers ["External Providers"]
        OpenAI["OpenAI API<br/>(Chat + Embeddings + Moderation)"]
    end

    UI -->|"HTTP/JSON + SSE"| api
    Chat --> Guards --> Chain
    Stream --> Guards --> Chain
    CV --> Guards

    Chain --> Router
    Router -->|"small_talk / direct_answer"| Synth
    Router -->|"retrieval_required / tool_required"| Rewrite
    Rewrite --> Retriever --> Chunks --> CTX --> Synth
    Router -->|"tool_required"| Tools --> Synth

    Retriever --> Qdrant
    Chain --> OpenAI
    Guards --> Redis
```

## Key capabilities

| Feature | Description |
|---------|-------------|
| **Intent-first routing** | LLM classifies intent (small_talk, direct_answer, retrieval_required, tool_required) before deciding actions |
| **Advanced RAG** | Query rewriting, multi-query retrieval, metadata filtering, weak-evidence abstention, citation-grounded answers |
| **CV-aware assistance** | Secure CV upload, token-safe processing, risk scoring, CV context included only when relevant |
| **Tool calling** | Skill gap analyzer, role comparison, learning plan generator |
| **End-to-end streaming** | SSE streaming with fast-path for conversational intents, status events for retrieval |
| **Multi-layer security** | Heuristic guards, encoded-attack detection, OpenAI moderation, structural sanitization, randomized boundaries, rate limiting |
| **Evaluation framework** | Golden dataset, routing accuracy checks, citation integrity, retrieval hit metrics |

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose (for Qdrant, Postgres, Redis)

### 1. Clone and set up environment

```bash
git clone <repo-url> && cd Career_Consultant_Platform
cp .env.example .env    # fill in OPENAI_API_KEY and other secrets
```

### 2. Start infrastructure

```bash
docker compose up -d    # starts Qdrant, Postgres, Redis
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Run the API

```bash
uv run uvicorn career_intel.api.main:app --reload
```

### 5. Run the Streamlit UI

```bash
uv run streamlit run streamlit_app/app.py
```

### 6. Run tests

```bash
uv run python -m pytest tests/ --ignore=tests/integration -v
```

## Project structure

```
src/career_intel/
  config/          # Pydantic BaseSettings, env loading
  api/             # FastAPI routers (chat, cv, health, ingest, feedback, evaluation, metrics)
  orchestration/   # Chain, stream engine, context builder, prompts, synthesizer
  rag/             # Ingestion, chunking, embeddings, retrieval, citation mapping
  tools/           # Intent-first router, skill gap, role compare, learning plan
  security/        # Multi-layer guards, sanitization, risk scoring, rate limiting
  services/        # CV processor (extract, clean, truncate)
  storage/         # Postgres, Redis, Qdrant client wrappers
  llm/             # Centralized LLM/embedding client factory with retry/backoff
  logging/         # Structured logging setup
  schemas/         # Shared Pydantic models (API + domain + routing)
  evaluation/      # Golden datasets, eval runner, routing accuracy, metrics
streamlit_app/     # Streamlit frontend
tests/             # Unit, orchestration, RAG, security, API, tool tests
docs/              # Architecture, security, evaluation, RAG pipeline docs
```

## Request lifecycle

```mermaid
sequenceDiagram
    participant U as User (Streamlit)
    participant A as FastAPI
    participant G as Input Guards
    participant R as Router (LLM)
    participant RAG as RAG Pipeline
    participant T as Tools
    participant S as Synthesizer
    participant L as LLM (OpenAI)

    U->>A: POST /chat/stream
    A->>G: validate_input_deep
    G-->>A: ok
    A->>R: route_query(query)
    R->>L: classify intent
    L-->>R: {intent, tool, use_cv}
    R-->>A: RouterDecision

    alt intent = small_talk / direct_answer
        A->>L: stream direct response
        L-->>U: SSE tokens (fast path)
    else intent = retrieval_required / tool_required
        A->>RAG: rewrite + retrieve
        RAG-->>A: chunks
        opt intent = tool_required
            A->>T: execute_tool
            T-->>A: ToolCallResult
        end
        A->>S: synthesize_answer(chunks, tools, cv)
        S->>L: stream grounded response
        L-->>U: SSE tokens + citations
    end
```

## CV upload flow

```mermaid
sequenceDiagram
    participant U as User
    participant ST as Streamlit
    participant API as POST /cv/process
    participant P as CV Processor
    participant RS as Risk Scorer

    U->>ST: upload PDF/DOCX/TXT
    ST->>API: multipart file upload
    API->>P: validate + extract + clean + truncate
    P-->>API: cv_text
    API->>RS: score_cv_risk(cv_text)
    RS-->>API: CVRiskScore
    API-->>ST: {cv_text, risk_score, warnings}
    ST->>ST: store cv_text in session

    Note over ST,API: On subsequent chat, cv_text is included<br/>in request body only when router says use_cv=true
```

## Documentation

- [Architecture](docs/architecture.md)
- [Security](docs/security.md)
- [RAG Pipeline](docs/rag_pipeline.md)
- [Evaluation](docs/evaluation.md)
- [Workflows](docs/workflows.md)

## Known limitations

- The router relies on a single LLM call; misclassification is possible for ambiguous queries
- Multilingual injection detection depends on OpenAI's moderation API coverage
- CV parsing supports PDF/DOCX/TXT only; scanned/image-only PDFs will fail
- No user authentication; sessions are browser-local
- Retrieval quality depends on the ingested knowledge base

## License

MIT
