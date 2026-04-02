# Workflows

## Chat request sequence

```mermaid
sequenceDiagram
  participant U as User
  participant S as Streamlit
  participant A as FastAPI
  participant O as Orchestrator
  participant Q as Qdrant
  participant L as OpenAI
  U->>S: Send message
  S->>A: POST /chat
  A->>O: run_turn
  O->>O: input_guards
  O->>L: query_rewrite
  O->>Q: similarity_search
  Q-->>O: chunks
  O->>L: tool_calls (optional)
  O->>L: final_answer
  O-->>A: reply + citations + tools
  A-->>S: JSON response
  S-->>U: Render chat and sources
```

## Tool calling flow

```mermaid
flowchart TD
  intent[Assistant Intent] --> toolSelect[Tool Router]
  toolSelect --> t1[Skill Gap Tool]
  toolSelect --> t2[Role Compare Tool]
  toolSelect --> t3[Learning Plan Tool]
  t1 --> ragFetch1[RAG Fetch Context]
  t2 --> ragFetch2[RAG Fetch Context]
  t3 --> ragFetch3[RAG Fetch Context]
  ragFetch1 --> structured1[Pydantic Output]
  ragFetch2 --> structured2[Pydantic Output]
  ragFetch3 --> structured3[Pydantic Output]
  structured1 --> merge[Merge Tool Results]
  structured2 --> merge
  structured3 --> merge
  merge --> final[Final Narration with Citations]
```

## Tool dispatch policy

1. The orchestrator inspects the LLM's function-call decision.
2. Requested tool must be in the registered tool set.
3. Tool input is validated against its Pydantic schema.
4. Tool executes (may trigger RAG sub-retrieval).
5. Tool output is validated and merged into the final response.
6. If a tool fails, the orchestrator returns a graceful fallback message.
