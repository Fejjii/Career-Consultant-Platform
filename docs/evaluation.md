# Evaluation Strategy

## Evaluation pipeline

```mermaid
flowchart TD
  gold[Golden Dataset] --> retEval[Retrieval Eval]
  gold --> ansEval[Answer Eval]
  gold --> citeEval[Citation Integrity Eval]
  gold --> toolEval[Tool Schema Eval]
  gold --> safetyEval[Safety and Injection Eval]
  retEval --> report[Eval Report]
  ansEval --> report
  citeEval --> report
  toolEval --> report
  safetyEval --> report
  report --> ls[LangSmith Comparison — Optional]
```

## Evaluation dimensions

| Dimension | Metrics | Method |
|-----------|---------|--------|
| Retrieval quality | nDCG@k, MRR | Gold query-to-chunk_id labels |
| Answer quality | Rubric-based LLM judge | Deterministic + LLM-as-judge |
| Citation integrity | Precision/recall vs gold sources; cited_ids subset of retrieved_ids | Deterministic |
| Tool correctness | Schema validation, fixture-based expected outputs | Deterministic |
| Safety — injection | Heuristic, encoded, paraphrased, multilingual, base64 attack cases | Deterministic + moderation API |
| Safety — refusal | "must refuse salary guarantee" cases, out-of-scope queries | Deterministic |
| Streaming correctness | Full response matches non-streaming; SSE format valid | Integration tests |

## Golden dataset

Located at `src/career_intel/evaluation/datasets/golden_queries.json`.

Each entry contains:
- `query`: the user question
- `expected_chunk_ids`: chunk IDs that should appear in retrieval
- `expected_citations`: source IDs that should be cited
- `expected_behaviour`: e.g. "abstain", "use_tool:skill_gap", "cite_source"
- `tags`: category labels for filtering

## Early evaluation (Phase 3)

Introduced alongside the RAG pipeline:
- Retrieval smoke tests (do gold chunks appear in top-k?)
- Citation integrity checks (cited IDs are subset of retrieved IDs)
- Weak-evidence / abstain test cases
- Run via `pytest tests/rag/`

## Security evaluation test suite

Located at `tests/security/test_injection.py`:
- Direct injection (10+ heuristic patterns)
- Paraphrased/synonymous injection attempts
- Base64-encoded instruction injection
- Unicode BiDi character injection
- Zero-width character sequence detection
- Legitimate career queries that must NOT be flagged (false-positive regression)

## Evolving toward RAGAs / TruLens

The `evaluation/` module exposes stable interfaces. Adding RAGAs or TruLens later requires only an adapter — no change to the golden dataset format or eval runner contract.
