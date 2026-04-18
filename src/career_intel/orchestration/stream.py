"""Streaming orchestration — yields tokens as Server-Sent Events for the chat endpoint.

Design:
  - The router runs FIRST to classify intent.
  - For small_talk / general_knowledge: tokens stream immediately (no retrieval).
  - For retrieval / tool intents: retrieval + tool calling happen before
    streaming, then the synthesis is streamed token-by-token.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from career_intel.llm import get_chat_llm
from career_intel.llm.token_usage import merge_token_usages, usage_from_langchain_message
from career_intel.orchestration.synthesize import (
    CONVERSATIONAL_PROMPT,
    FALLBACK_SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    answer_length_system_suffix,
    synthesize_answer,
)
from career_intel.schemas.api import AnswerLengthMode, ChatMessage, ToolCallResult
from career_intel.security.guards import sanitize_model_output
from career_intel.security.sanitize import wrap_cv_content

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from career_intel.config import Settings

logger = structlog.get_logger()

_SKIP_RETRIEVAL_INTENTS = frozenset({"small_talk", "general_knowledge"})
_RAG_BLOCKED_NO_EVIDENCE_MESSAGE = "RAG blocked due to no evidence"


async def stream_turn(
    *,
    messages: list[ChatMessage],
    session_id: str,
    use_tools: bool,
    filters: dict[str, Any] | None,
    settings: Settings,
    trace_id: str,
    cv_text: str | None = None,
    user_timezone: str | None = None,
    answer_length: AnswerLengthMode = "balanced",
) -> AsyncGenerator[str, None]:
    """Execute a turn and yield SSE-formatted chunks.

    Event types:
      ``data: {"type": "intent", "data": "..."}``
      ``data: {"type": "token", "content": "..."}``
      ``data: {"type": "citations", "data": [...]}``
      ``data: {"type": "tool_calls", "data": [...]}``
      ``data: {"type": "usage", "data": {"prompt_tokens": int, ...}}`` (when available)
      ``data: {"type": "done"}``
      ``data: {"type": "error", "detail": "..."}``
    """
    from career_intel.api.routers.metrics import inc
    from career_intel.security.guards import validate_input_deep
    from career_intel.tools.registry import canonicalize_intent, execute_tool, route_query

    inc("chat_requests_total")
    user_query = messages[-1].content
    t_start = time.monotonic()

    try:
        await validate_input_deep(
            user_query,
            max_length=settings.max_input_length,
            settings=settings,
        )

        from career_intel.services.source_inventory import (
            build_source_inventory_summary,
            format_source_inventory_answer,
            is_source_inventory_query,
        )

        if is_source_inventory_query(user_query):
            summary = build_source_inventory_summary()
            reply = format_source_inventory_answer(summary)
            logger.info(
                "source_inventory_answered",
                session_id=session_id,
                answer_source="source_inventory",
                sources_count=summary.total_source_groups,
                esco_ingestion_status=summary.esco_ingestion_status,
            )
            yield _sse({"type": "intent", "data": "source_inventory"})
            yield _sse({"type": "status", "detail": "Loading source inventory..."})
            yield _sse({"type": "token", "content": reply})
            yield _sse({"type": "debug", "data": {
                "intent": "source_inventory",
                "confidence": 1.0,
                "retrieval_invoked": False,
                "tool_invoked": False,
                "answer_source": "source_inventory",
                "answer_mode": "SOURCE_INVENTORY",
                "runtime_utility_used": None,
                "sources_count": summary.total_source_groups,
                "use_cv": False,
                "answer_length": answer_length,
                "route_latency_ms": 0.0,
                "total_latency_ms": round((time.monotonic() - t_start) * 1000, 1),
            }})
            yield _sse({"type": "done"})
            return

        # --- Route FIRST ---
        t_route_start = time.monotonic()
        decision, router_usage = await route_query(
            user_query,
            cv_available=cv_text is not None,
            settings=settings,
        )
        intent = canonicalize_intent(decision.intent)
        route_latency_ms = round((time.monotonic() - t_route_start) * 1000, 1)
        runtime_utility_used: str | None = None
        yield _sse({"type": "intent", "data": intent})

        logger.info(
            "intent_classified",
            classified_intent=intent,
            confidence=decision.confidence,
            reason=decision.reason,
        )

        if intent == "dynamic_runtime":
            from career_intel.services.runtime_utility import resolve_preferred_timezone, resolve_runtime_query

            runtime_timezone = resolve_preferred_timezone(
                user_timezone=user_timezone,
                configured_timezone=getattr(settings, "runtime_default_timezone", None),
            )

            runtime_result = resolve_runtime_query(
                user_query,
                timezone_name=runtime_timezone,
            )
            if runtime_result is None:
                reply = (
                    "I recognized this as a runtime/date-time request but could not resolve it "
                    "deterministically. Please rephrase with a clear time or calendar reference."
                )
                runtime_utility_used = "blocked_no_runtime_resolution"
            else:
                reply = runtime_result.answer_text
                runtime_utility_used = runtime_result.utility_name

            yield _sse({"type": "status", "detail": "Resolving runtime data..."})
            yield _sse({"type": "token", "content": reply})

            total_ms = round((time.monotonic() - t_start) * 1000, 1)
            logger.info(
                "stream_complete",
                session_id=session_id,
                intent=intent,
                retrieval_invoked=False,
                tool_invoked=False,
                answer_source="runtime",
                answer_mode="RUNTIME",
                runtime_utility_used=runtime_utility_used,
                sources_count=0,
                stream_mode=True,
                route_latency_ms=route_latency_ms,
                total_latency_ms=total_ms,
            )
            yield _sse({"type": "debug", "data": {
                "intent": intent,
                "confidence": decision.confidence,
                "retrieval_invoked": False,
                "tool_invoked": False,
                "answer_source": "runtime",
                "answer_mode": "RUNTIME",
                "runtime_utility_used": runtime_utility_used,
                "sources_count": 0,
                "use_cv": decision.use_cv,
                "answer_length": answer_length,
                "route_latency_ms": route_latency_ms,
                "total_latency_ms": total_ms,
            }})
            if router_usage is not None:
                yield _sse({"type": "usage", "data": router_usage.model_dump()})
            yield _sse({"type": "done"})
            return

        # --- Fast path: small_talk / general_knowledge ---
        if intent in _SKIP_RETRIEVAL_INTENTS:
            logger.info(
                "grounding_path_selected",
                path="FALLBACK",
                answer_source="llm_fallback",
                answer_mode="LLM",
                runtime_utility_used=runtime_utility_used,
                reason="intent_does_not_require_grounding",
            )
            llm = get_chat_llm(settings, temperature=0.7, streaming=True)
            raw_text = ""
            last_stream_chunk: Any = None
            conv_system = CONVERSATIONAL_PROMPT + answer_length_system_suffix(answer_length)
            async for chunk in llm.astream([
                {"role": "system", "content": conv_system},
                {"role": "user", "content": user_query},
            ]):
                last_stream_chunk = chunk
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    raw_text += token
            stream_usage = usage_from_langchain_message(last_stream_chunk)
            turn_usage = merge_token_usages(router_usage, stream_usage)
            safe_text = sanitize_model_output(raw_text)
            first_token = True
            for token in _chunk_text(safe_text):
                if first_token:
                    logger.info(
                        "stream_first_token",
                        session_id=session_id,
                        intent=intent,
                        first_token_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
                    )
                    first_token = False
                yield _sse({"type": "token", "content": token})

            total_ms = round((time.monotonic() - t_start) * 1000, 1)
            logger.info(
                "stream_complete",
                session_id=session_id,
                intent=intent,
                retrieval_invoked=False,
                tool_invoked=False,
                answer_source="llm_fallback",
                answer_mode="LLM",
                runtime_utility_used=runtime_utility_used,
                sources_count=0,
                stream_mode=True,
                route_latency_ms=route_latency_ms,
                total_latency_ms=total_ms,
            )
            yield _sse({"type": "debug", "data": {
                "intent": intent,
                "confidence": decision.confidence,
                "retrieval_invoked": False,
                "tool_invoked": False,
                "answer_source": "llm_fallback",
                "answer_mode": "LLM",
                "runtime_utility_used": runtime_utility_used,
                "sources_count": 0,
                "use_cv": decision.use_cv,
                "answer_length": answer_length,
                "route_latency_ms": route_latency_ms,
                "total_latency_ms": total_ms,
            }})
            if turn_usage is not None:
                yield _sse({"type": "usage", "data": turn_usage.model_dump()})
            yield _sse({"type": "done"})
            return

        # --- Normal path with strict grounding priority ---
        from career_intel.rag.query_preprocessor import normalize_query_for_retrieval
        from career_intel.rag.retriever import (
            assess_evidence_strength,
            assess_source_consistency,
            build_query_profile,
            detect_query_source,
            merge_query_profiles,
            retrieve_chunks,
            rewrite_query,
            should_force_rag,
        )

        rewritten = user_query
        chunks: list = []
        evidence_strength = "none"
        force_rag = False
        force_reason = "not_evaluated"
        path_reason = "intent_requires_tool"
        detected_source = None
        source_consistent = False
        source_consistency_reason = "not_evaluated"
        retrieved_sources: list[str] = []
        query_profile = None
        retriever_called = False
        if intent == "domain_specific":
            retriever_called = True
            logger.info(
                "retriever_invocation",
                called=True,
                intent=intent,
                reason="domain_specific_intent",
                query_preview=user_query[:80],
            )
            yield _sse({"type": "status", "detail": "Retrieving evidence..."})
            retrieval_query_context = await normalize_query_for_retrieval(
                user_query,
                settings=settings,
            )
            retrieval_query = retrieval_query_context.retrieval_query
            original_detected_source = (
                detect_query_source(retrieval_query) or detect_query_source(user_query)
            )
            original_query_profile = build_query_profile(
                retrieval_query,
                detected_source=original_detected_source,
            )
            rewritten = await rewrite_query(retrieval_query, settings=settings)
            rewritten_detected_source = detect_query_source(rewritten)
            detected_source = rewritten_detected_source or original_detected_source
            rewritten_query_profile = build_query_profile(
                rewritten,
                detected_source=detected_source,
            )
            query_profile = merge_query_profiles(
                original_query_profile,
                rewritten_query_profile,
            )
            logger.info(
                "grounding_query_profile",
                query_preview=user_query[:80],
                detected_source=detected_source,
                original_detected_source=original_detected_source,
                rewritten_detected_source=rewritten_detected_source,
                esco_relation_query=query_profile.esco_relation_query,
                taxonomy_query=query_profile.taxonomy_query,
                essential_optional_query=query_profile.essential_optional_query,
                salient_concepts=list(query_profile.salient_concepts),
            )
            chunks = await retrieve_chunks(
                query=rewritten,
                filters=filters,
                settings=settings,
                detected_source_override=detected_source,
                query_profile_override=query_profile,
            )
            source_consistent, source_consistency_reason, retrieved_sources = assess_source_consistency(
                chunks,
                detected_source=detected_source,
            )
            evidence_strength, path_reason = assess_evidence_strength(
                chunks,
                settings=settings,
                detected_source=detected_source,
            )
            force_rag, force_reason = should_force_rag(
                chunks,
                settings=settings,
                detected_source=detected_source,
            )
            logger.info(
                "retriever_result",
                called=True,
                retrieved_chunks=len(chunks),
                detected_source=detected_source,
                selected_esco_doc_types=[
                    chunk.metadata.esco_doc_type for chunk in chunks if chunk.metadata.esco_doc_type
                ],
            )
        elif intent == "tool_required":
            yield _sse({"type": "status", "detail": "Running structured analysis..."})
        else:
            logger.info(
                "retriever_invocation",
                called=False,
                intent=intent,
                reason="intent_not_domain_specific",
                query_preview=user_query[:80],
            )

        rag_blocked_reason = None
        if (
            intent == "domain_specific"
            and query_profile is not None
            and query_profile.esco_relation_query
            and evidence_strength == "none"
        ):
            rag_blocked_reason = "blocked_none_evidence_for_esco_relation_query"

        tool_results: list[ToolCallResult] = []
        if intent == "tool_required" and decision.tool_name and use_tools:
            try:
                result = await execute_tool(decision, settings)
                tool_results = [result]
                inc("tool_calls_total", 1)
                yield _sse({"type": "tool_calls", "data": [result.model_dump()]})
            except Exception as exc:
                logger.error("tool_execution_failed", tool=decision.tool_name, error=str(exc))
                tool_results = [ToolCallResult(
                    tool_name=decision.tool_name,
                    inputs=decision.params,
                    output={"error": str(exc)},
                    success=False,
                    error=str(exc),
                )]
                yield _sse({"type": "tool_calls", "data": [tool_results[0].model_dump()]})

        answer_source = "llm_fallback"
        answer_mode = "LLM"
        if intent == "domain_specific" and evidence_strength == "none":
            path_reason = rag_blocked_reason or "blocked_none_evidence"
        elif intent == "domain_specific" and evidence_strength in {"strong", "weak"}:
            answer_source = "rag"
            answer_mode = "RAG"
            path_reason = f"rag_{evidence_strength}"
        elif intent == "domain_specific" and force_rag and rag_blocked_reason is None:
            answer_source = "rag"
            answer_mode = "RAG"
            path_reason = f"rag_forced_{force_reason}"
        elif rag_blocked_reason is not None:
            path_reason = rag_blocked_reason
        elif intent == "tool_required" and any(result.success for result in tool_results):
            answer_source = "tool"
            answer_mode = "TOOL"
            path_reason = "tool_required_success"
        elif intent == "domain_specific" and any(result.success for result in tool_results):
            answer_source = "tool"
            answer_mode = "TOOL"
            path_reason = "weak_rag_promoted_to_tool"
        elif intent == "tool_required":
            path_reason = "tool_unavailable_or_failed"

        logger.info(
            "grounding_rag_evaluation",
            evidence_strength=evidence_strength,
            similarity_threshold=settings.rag_similarity_threshold,
            strong_similarity_threshold=settings.rag_strong_evidence_threshold,
            weak_similarity_threshold=settings.rag_weak_evidence_threshold,
            rerank_coherence_threshold=settings.rag_rerank_coherence_threshold,
            force_min_chunks=settings.rag_force_min_chunks,
            retrieved_chunks=len(chunks),
            retriever_called=retriever_called,
            detected_source=detected_source,
            esco_relation_query=(query_profile.esco_relation_query if query_profile is not None else False),
            taxonomy_query=(query_profile.taxonomy_query if query_profile is not None else False),
            essential_optional_query=(
                query_profile.essential_optional_query if query_profile is not None else False
            ),
            salient_concepts=(list(query_profile.salient_concepts) if query_profile is not None else []),
            source_consistent=source_consistent,
            source_consistency_reason=source_consistency_reason,
            retrieved_sources=retrieved_sources,
            force_rag_applied=force_rag and answer_source == "rag",
            force_reason=force_reason,
            rag_accepted=answer_source == "rag",
            rag_blocked_reason=rag_blocked_reason,
            similarity_scores=[round(chunk.score, 4) for chunk in chunks],
            rerank_scores=[round(chunk.rerank_score or chunk.score, 4) for chunk in chunks],
        )
        if rag_blocked_reason is not None:
            logger.info(
                "grounding_rag_blocked_none_evidence",
                query_preview=user_query[:80],
                message=_RAG_BLOCKED_NO_EVIDENCE_MESSAGE,
                evidence_strength=evidence_strength,
                reason=rag_blocked_reason,
                detected_source=detected_source,
                esco_relation_query=query_profile.esco_relation_query if query_profile is not None else False,
                taxonomy_query=query_profile.taxonomy_query if query_profile is not None else False,
            )
            if getattr(settings, "environment", None) == "development":
                print(f"[RAG] {_RAG_BLOCKED_NO_EVIDENCE_MESSAGE}")
        if intent == "domain_specific" and answer_source != "rag":
            logger.info(
                "grounding_rag_rejected",
                evidence_strength=evidence_strength,
                reason=path_reason,
                detected_source=detected_source,
                esco_relation_query=query_profile.esco_relation_query if query_profile is not None else False,
                taxonomy_query=query_profile.taxonomy_query if query_profile is not None else False,
                similarity_scores=[round(chunk.score, 4) for chunk in chunks],
                rerank_scores=[round(chunk.rerank_score or chunk.score, 4) for chunk in chunks],
            )

        logger.info(
            "grounding_path_selected",
            path=answer_source.upper() if answer_source != "llm_fallback" else "FALLBACK",
            answer_source=answer_source,
            answer_mode=answer_mode,
            runtime_utility_used=runtime_utility_used,
            evidence_strength=evidence_strength,
            reason=path_reason,
            similarity_scores=[round(chunk.score, 4) for chunk in chunks],
            rerank_scores=[round(chunk.rerank_score or chunk.score, 4) for chunk in chunks],
        )

        tool_block = ""
        if tool_results:
            parts = [f"### Tool: {tr.tool_name}\n```json\n{tr.output}\n```" for tr in tool_results]
            tool_block = "\n\n".join(parts)

        if answer_source == "rag":
            yield _sse({"type": "status", "detail": "Generating grounded response..."})
            reply_text, citations, synth_usage = await synthesize_answer(
                query=user_query,
                rewritten_query=rewritten,
                chunks=chunks,
                tool_results=tool_results,
                answer_source=answer_source,
                settings=settings,
                cv_text=cv_text,
                use_cv=decision.use_cv,
                answer_length=answer_length,
            )
            turn_usage = merge_token_usages(router_usage, synth_usage)

            first_token = True
            for token in _chunk_text(reply_text):
                if first_token:
                    logger.info(
                        "stream_first_token",
                        session_id=session_id,
                        intent=intent,
                        first_token_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
                    )
                    first_token = False
                yield _sse({"type": "token", "content": token})

            if citations:
                yield _sse({"type": "citations", "data": [c.model_dump() for c in citations]})

            total_ms = round((time.monotonic() - t_start) * 1000, 1)
            logger.info(
                "stream_complete",
                session_id=session_id,
                intent=intent,
                retrieval_invoked=True,
                tool_invoked=bool(tool_results),
                answer_source=answer_source,
                answer_mode=answer_mode,
                runtime_utility_used=runtime_utility_used,
                sources_count=len(citations),
                stream_mode=True,
                route_latency_ms=route_latency_ms,
                total_latency_ms=total_ms,
            )
            yield _sse({"type": "debug", "data": {
                "intent": intent,
                "confidence": decision.confidence,
                "retrieval_invoked": True,
                "tool_invoked": bool(tool_results),
                "answer_source": answer_source,
                "answer_mode": answer_mode,
                "runtime_utility_used": runtime_utility_used,
                "evidence_strength": evidence_strength,
                "path_reason": path_reason,
                "sources_count": len(citations),
                "use_cv": decision.use_cv,
                "answer_length": answer_length,
                "route_latency_ms": route_latency_ms,
                "total_latency_ms": total_ms,
            }})
            if turn_usage is not None:
                yield _sse({"type": "usage", "data": turn_usage.model_dump()})
            yield _sse({"type": "done"})
            return

        if answer_source == "tool":
            parts = [f"Question: {user_query}"]
            if cv_text and decision.use_cv:
                parts.append(wrap_cv_content(cv_text))
            parts.append(f"Tool results:\n{tool_block}")
            user_prompt = "\n\n".join(parts)
            system_prompt = TOOL_SYSTEM_PROMPT + answer_length_system_suffix(answer_length)
        else:
            parts = [f"Question: {user_query}"]
            if cv_text and decision.use_cv:
                parts.append(wrap_cv_content(cv_text))
            if tool_block:
                parts.append(f"Tool results attempted:\n{tool_block}")
            user_prompt = "\n\n".join(parts)
            system_prompt = FALLBACK_SYSTEM_PROMPT + answer_length_system_suffix(answer_length)

        llm = get_chat_llm(settings, temperature=0.2, streaming=True)
        yield _sse({"type": "status", "detail": "Generating grounded response..."})

        raw_text = ""
        last_stream_chunk = None
        async for chunk in llm.astream([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]):
            last_stream_chunk = chunk
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                raw_text += token

        stream_usage = usage_from_langchain_message(last_stream_chunk)
        turn_usage = merge_token_usages(router_usage, stream_usage)
        safe_text = sanitize_model_output(raw_text)
        first_token = True
        for token in _chunk_text(safe_text):
            if first_token:
                logger.info(
                    "stream_first_token",
                    session_id=session_id,
                    intent=intent,
                    first_token_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
                )
                first_token = False
            yield _sse({"type": "token", "content": token})

        citations = []
        if citations:
            yield _sse({"type": "citations", "data": [c.model_dump() for c in citations]})

        total_ms = round((time.monotonic() - t_start) * 1000, 1)
        logger.info(
            "stream_complete",
            session_id=session_id,
            intent=intent,
            retrieval_invoked=retriever_called,
            tool_invoked=bool(tool_results),
            answer_source=answer_source,
            answer_mode=answer_mode,
            runtime_utility_used=runtime_utility_used,
            sources_count=len(citations),
            stream_mode=True,
            route_latency_ms=route_latency_ms,
            total_latency_ms=total_ms,
        )
        yield _sse({"type": "debug", "data": {
            "intent": intent,
            "confidence": decision.confidence,
            "retrieval_invoked": retriever_called,
            "tool_invoked": bool(tool_results),
            "answer_source": answer_source,
            "answer_mode": answer_mode,
            "runtime_utility_used": runtime_utility_used,
            "evidence_strength": evidence_strength,
            "path_reason": path_reason,
            "sources_count": len(citations),
            "use_cv": decision.use_cv,
            "answer_length": answer_length,
            "route_latency_ms": route_latency_ms,
            "total_latency_ms": total_ms,
        }})
        if turn_usage is not None:
            yield _sse({"type": "usage", "data": turn_usage.model_dump()})
        yield _sse({"type": "done"})

    except Exception as exc:
        logger.error("stream_error", error=str(exc)[:200], session_id=session_id)
        yield _sse({"type": "error", "detail": str(exc)[:500]})


def _sse(payload: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_text(text: str, chunk_size: int = 240) -> list[str]:
    """Split buffered text into stable SSE token chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
