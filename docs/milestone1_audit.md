# Milestone 1 Readiness Audit

Performed: 2026-03-31

---

## Milestone 1 definition

A working vertical slice where the app can:

1. Ingest pilot data
2. Answer career questions with grounded citations
3. Abstain when evidence is weak
4. Call at least one tool correctly
5. Stream the answer in the UI
6. Apply security guards and rate limiting
7. Produce logs/traces
8. Pass tests end-to-end

---

## Audit verdicts

| # | Capability | Verdict | Finding |
|---|-----------|---------|---------|
| 1 | **Ingest pilot data** | **BLOCKED** | `qdrant_store.upsert_vectors` passes 16-char hex `chunk_id` strings as Qdrant point IDs. Qdrant requires either an integer or a valid UUID string. A 16-char hex like `a7b3c9d2e1f04a82` is neither — the upsert will fail at runtime with a Qdrant validation error. Ingestion has never been run against a live Qdrant instance. |
| 2 | **Answer with citations** | **Partially implemented** | Synthesis chain, citation builder, and prompt contract are all wired. Cannot be validated end-to-end because ingestion (item 1) is broken, so there are no vectors to retrieve. |
| 3 | **Abstain on weak evidence** | **Fully implemented (code-level)** | `synthesize.py` and `stream.py` both check `WEAK_EVIDENCE_THRESHOLD = 0.35` and return the abstain message. Logic is correct. Not yet validated against a real corpus because of item 1. |
| 4 | **Call at least one tool** | **Partially implemented** | Tool registry, Pydantic schemas, and all 3 tool implementations exist. Tool router makes an LLM call to decide which tool to invoke. Cannot be validated end-to-end because retrieval (item 1) is broken. |
| 5 | **Stream answer in UI** | **BLOCKED — 2 bugs** | **(a)** Rate-limit middleware matches `request.url.path in ("/chat", ...)` but the streaming endpoint is mounted at `/chat/stream` (with router prefix `/chat` + route `/stream`). The path seen by middleware is `/chat/stream`, which does NOT match `"/chat"`. The streaming endpoint completely bypasses rate limiting. **(b)** `stream.py` line 48 imports `validate_input` (Layer 1 only) instead of `validate_input_deep` (multi-layer). The streaming path has weaker injection protection than the non-streaming path. |
| 6 | **Security guards and rate limiting** | **Partially implemented** | Guards, sanitizer, classifier, and rate limiter all exist and are tested in isolation. But rate limiting only fires on exact path matches `/chat`, `/ingest`, `/feedback` — it misses `/chat/stream` and uses string equality not prefix matching. |
| 7 | **Produce logs/traces** | **Fully implemented (code-level)** | Structured logging via structlog throughout. LangSmith wiring in `tracing.py` sets env vars on startup. Not validated with a real LangSmith key, but the mechanism is sound. |
| 8 | **Pass tests end-to-end** | **Passing (83/83)** | All current tests pass. However, there are NO integration tests that exercise ingest -> retrieve -> answer on a real (or test-container) Qdrant instance. The test suite validates components in isolation only. |

---

## Detailed findings

### BUG 1 (Critical): Qdrant point ID type mismatch

**File:** `src/career_intel/rag/chunking.py` line 29 / `src/career_intel/storage/qdrant_store.py` line 54

`RawChunk.__post_init__` generates `chunk_id = hashlib.sha256(...).hexdigest()[:16]` (e.g. `"a7b3c9d2e1f04a82"`). This 16-char hex string is passed to `PointStruct(id=uid, ...)`.

Qdrant's `PointStruct.id` accepts:
- An unsigned 64-bit integer
- A UUID-format string (e.g. `"550e8400-e29b-41d4-a716-446655440000"`)

A 16-char hex string is neither. The upsert call will raise a Qdrant validation error.

**Fix:** Convert chunk IDs to valid UUIDs using `uuid.uuid5` with the text hash as the name, or convert the hex to an integer.

### BUG 2 (High): Streaming endpoint bypasses rate limiting

**File:** `src/career_intel/api/main.py` line 58

```python
if request.url.path in ("/chat", "/ingest", "/feedback"):
```

The streaming endpoint is at `/chat/stream`. This exact-match check misses it entirely. An attacker can send unlimited requests to `/chat/stream`.

**Fix:** Use `request.url.path.startswith("/chat")` or add `"/chat/stream"` to the set.

### BUG 3 (High): Streaming path uses Layer-1-only input validation

**File:** `src/career_intel/orchestration/stream.py` line 48

```python
from career_intel.security.guards import validate_input  # Layer 1 only!
```

The non-streaming path (`chain.py` line 36) correctly uses `validate_input_deep` (all 3 layers). The streaming path silently downgrades to heuristic-only checks.

**Fix:** Change to `validate_input_deep` and make the call `await validate_input_deep(...)`.

### ISSUE 4 (Medium): Ingestion never validated end-to-end

The `ingest_pipeline.py` imports from `storage/db.py` which uses SQLAlchemy async models, but `init_db()` (which creates tables) runs inside `run_ingestion()`. This means:
- First ingestion call creates tables on the fly (fragile in production, fine for MVP).
- The pipeline has never been run against a real stack (Qdrant + Postgres + OpenAI) because of Bug 1.

### ISSUE 5 (Low): `conftest.py` caches Settings with a fake API key

`tests/conftest.py` sets `OPENAI_API_KEY=sk-test-key-not-real` and the `Settings` class uses `@lru_cache`. If any test path tries to clear the cache and reload with a real key (e.g. for an integration test), the cached instance persists. This is fine for unit tests but will trip up future integration tests.

### ISSUE 6 (Low): No test for streaming endpoint format

There are no tests that verify `POST /chat/stream` returns valid SSE-formatted events. The Streamlit app parses `data: {...}\n\n` lines; if the format is wrong the UI silently fails.

---

## Gap analysis summary

| Gap | Severity | Blocks Milestone 1? |
|-----|----------|---------------------|
| Qdrant point ID type mismatch | **Critical** | YES — ingestion crashes, nothing downstream works |
| Streaming endpoint bypasses rate limiting | **High** | YES — security requirement not met |
| Streaming path uses weak validation | **High** | YES — security requirement not met for streaming |
| No end-to-end integration test | **Medium** | Partially — we can't prove the vertical slice works |
| No streaming format test | **Low** | No — can be verified manually |
| Settings cache in tests | **Low** | No |

---

## Prioritized fix list

| Priority | Fix | Effort | Files |
|----------|-----|--------|-------|
| **P0** | Fix Qdrant point ID generation: use `uuid.uuid5(uuid.NAMESPACE_OID, sha256_hex)` in `chunking.py` | 10 min | `rag/chunking.py` |
| **P1** | Fix rate-limit middleware path matching: use `startswith` or explicit set including `/chat/stream` | 5 min | `api/main.py` |
| **P2** | Fix streaming path to use `validate_input_deep` | 5 min | `orchestration/stream.py` |
| **P3** | Add minimal streaming endpoint format test | 15 min | `tests/api/test_chat_stream.py` |
| **P4** | Run full ingest -> retrieve -> answer manual smoke test against Docker Compose stack | 30 min | Manual validation |

---

## The single best next coding task

**Fix Bug 1 + Bug 2 + Bug 3 in one commit.** These three changes are small (< 30 lines total), completely unblock the vertical slice, and remove the only critical/high blockers. Everything else in the codebase is sound and ready — these three bugs are the only things standing between the current code and a working Milestone 1 demo.
