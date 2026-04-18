"""Main orchestration chain with strict grounding priority enforcement."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from career_intel.llm.token_usage import merge_token_usages
from career_intel.schemas.api import AnswerLengthMode, ChatMessage, ChatResponse, ToolCallResult

if TYPE_CHECKING:
    from career_intel.config import Settings

logger = structlog.get_logger()

_SKIP_RETRIEVAL_INTENTS = frozenset({"small_talk", "general_knowledge"})
_RAG_BLOCKED_NO_EVIDENCE_MESSAGE = "RAG blocked due to no evidence"


async def run_turn(
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
) -> ChatResponse:
    """Execute a single conversational turn through the full pipeline.

    Flow: input guard -> route -> conditional(rewrite + retrieve + tools) -> synthesize.
    """
    from career_intel.api.routers.metrics import inc

    inc("chat_requests_total")
    t_start = time.monotonic()

    user_query = messages[-1].content
    logger.info(
        "turn_start",
        session_id=session_id,
        query_preview=user_query[:80],
        query_length=len(user_query),
        has_cv=cv_text is not None,
        stream_mode=False,
    )

    # --- Step 1: Input guards ---
    from career_intel.security.guards import validate_input_deep

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
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            citations=[],
            tool_calls=[],
            answer_source="source_inventory",
            answer_mode="SOURCE_INVENTORY",
            runtime_utility_used=None,
            intent="source_inventory",
            answer_length=answer_length,
            trace_id=trace_id,
            usage=None,
        )

    # --- Step 2: Route FIRST (intent classification) ---
    from career_intel.tools.registry import canonicalize_intent, execute_tool, route_query

    decision, router_usage = await route_query(
        user_query,
        cv_available=cv_text is not None,
        settings=settings,
    )
    intent = canonicalize_intent(decision.intent)
    use_cv = decision.use_cv
    answer_mode = "LLM"
    runtime_utility_used: str | None = None

    logger.info(
        "intent_classified",
        classified_intent=intent,
        confidence=decision.confidence,
        reason=decision.reason,
    )

    # --- Step 2.5: Deterministic runtime path ---
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

        logger.info(
            "turn_complete",
            session_id=session_id,
            intent=intent,
            retrieval_invoked=False,
            tool_invoked=False,
            answer_source="runtime",
            answer_mode="RUNTIME",
            runtime_utility_used=runtime_utility_used,
            sources_count=0,
            total_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
        )
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            citations=[],
            tool_calls=[],
            answer_source="runtime",
            answer_mode="RUNTIME",
            runtime_utility_used=runtime_utility_used,
            intent=intent,
            answer_length=answer_length,
            trace_id=trace_id,
            usage=router_usage,
        )

    # --- Step 3: Fast path for small_talk / general_knowledge ---
    if intent in _SKIP_RETRIEVAL_INTENTS:
        from career_intel.orchestration.synthesize import generate_direct_response

        reply, direct_usage = await generate_direct_response(
            user_query,
            settings,
            answer_length=answer_length,
        )
        turn_usage = merge_token_usages(router_usage, direct_usage)
        answer_source = "llm_fallback"
        answer_mode = "LLM"
        logger.info(
            "grounding_path_selected",
            path="FALLBACK",
            answer_source=answer_source,
            reason="intent_does_not_require_grounding",
        )

        logger.info(
            "turn_complete",
            session_id=session_id,
            intent=intent,
            retrieval_invoked=False,
            tool_invoked=False,
            answer_source=answer_source,
            answer_mode=answer_mode,
            runtime_utility_used=runtime_utility_used,
            sources_count=0,
            total_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
        )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            citations=[],
            tool_calls=[],
            answer_source=answer_source,
            answer_mode=answer_mode,
            runtime_utility_used=runtime_utility_used,
            intent=intent,
            answer_length=answer_length,
            trace_id=trace_id,
            usage=turn_usage,
        )

    # --- Step 4: Rewrite + Retrieve (only when the router asks for RAG evidence) ---
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
        retrieval_query_context = await normalize_query_for_retrieval(
            user_query,
            settings=settings,
        )
        retrieval_query = retrieval_query_context.retrieval_query
        original_detected_source = detect_query_source(retrieval_query) or detect_query_source(user_query)
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

    if not retriever_called:
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

    # --- Step 5: Tool execution (only if router selected a tool) ---
    tool_results: list[ToolCallResult] = []
    if intent == "tool_required" and decision.tool_name and use_tools:
        try:
            result = await execute_tool(decision, settings)
            tool_results = [result]
            inc("tool_calls_total", 1)
        except Exception as exc:
            logger.error("tool_execution_failed", tool=decision.tool_name, error=str(exc))
            tool_results = [ToolCallResult(
                tool_name=decision.tool_name,
                inputs=decision.params,
                output={"error": str(exc)},
                success=False,
                error=str(exc),
            )]

    answer_source = "llm_fallback"
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

    # --- Step 6: Synthesize answer with citations ---
    from career_intel.orchestration.synthesize import synthesize_answer

    reply, citations, synth_usage = await synthesize_answer(
        query=user_query,
        rewritten_query=rewritten if answer_source == "rag" else user_query,
        chunks=chunks,
        tool_results=tool_results,
        answer_source=answer_source,
        settings=settings,
        cv_text=cv_text,
        use_cv=use_cv,
        answer_length=answer_length,
    )
    turn_usage = merge_token_usages(router_usage, synth_usage)

    logger.info(
        "turn_complete",
        session_id=session_id,
        intent=intent,
        retrieval_invoked=retriever_called,
        tool_invoked=bool(tool_results),
        answer_source=answer_source,
        answer_mode=answer_mode,
        runtime_utility_used=runtime_utility_used,
        sources_count=len(citations),
        total_latency_ms=round((time.monotonic() - t_start) * 1000, 1),
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        citations=citations,
        tool_calls=tool_results,
        answer_source=answer_source,
        answer_mode=answer_mode,
        runtime_utility_used=runtime_utility_used,
        intent=intent,
        answer_length=answer_length,
        trace_id=trace_id,
        usage=turn_usage,
    )
