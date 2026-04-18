"""Run quantitative RAGAS evaluation for the Career Intel assistant.

Usage:
    uv run python scripts/run_ragas.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from career_intel.config import Settings, get_settings
from career_intel.orchestration.chain import run_turn
from career_intel.rag.query_preprocessor import normalize_query_for_retrieval
from career_intel.rag.retriever import (
    build_query_profile,
    detect_query_source,
    merge_query_profiles,
    retrieve_chunks,
    rewrite_query,
)
from career_intel.schemas.api import ChatMessage
from career_intel.tools.registry import canonicalize_intent, route_query

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "career_intel"
    / "evaluation"
    / "datasets"
    / "ragas_queries.json"
)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "ragas"
METRIC_COLUMNS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass(frozen=True)
class RagasEvalItem:
    """One query + references used to score the assistant."""

    id: str
    category: str
    query: str
    expected_answer: str
    expected_contexts: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the curated dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the JSON dataset.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where JSON/Markdown reports are written.",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> list[RagasEvalItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RagasEvalItem(
            id=item["id"],
            category=item["category"],
            query=item["query"],
            expected_answer=item["expected_answer"],
            expected_contexts=list(item.get("expected_contexts", [])),
        )
        for item in payload
    ]


async def _collect_retrieved_contexts(query: str, settings: Settings) -> tuple[str, list[str]]:
    """Mirror retrieval flow to gather contexts for context metrics."""
    decision, _usage = await route_query(query, cv_available=False, settings=settings)
    intent = canonicalize_intent(decision.intent)
    if intent != "domain_specific":
        return intent, []

    retrieval_query_context = await normalize_query_for_retrieval(query, settings=settings)
    retrieval_query = retrieval_query_context.retrieval_query
    original_detected_source = detect_query_source(retrieval_query) or detect_query_source(query)
    original_query_profile = build_query_profile(
        retrieval_query,
        detected_source=original_detected_source,
    )

    rewritten = await rewrite_query(retrieval_query, settings=settings)
    rewritten_detected_source = detect_query_source(rewritten)
    detected_source = rewritten_detected_source or original_detected_source
    query_profile = merge_query_profiles(
        original_query_profile,
        build_query_profile(rewritten, detected_source=detected_source),
    )
    chunks = await retrieve_chunks(
        query=rewritten,
        filters=None,
        settings=settings,
        detected_source_override=detected_source,
        query_profile_override=query_profile,
    )
    return intent, [chunk.text for chunk in chunks]


async def _evaluate_samples(
    items: list[RagasEvalItem],
    settings: Settings,
) -> tuple[list[SingleTurnSample], list[dict[str, Any]]]:
    samples: list[SingleTurnSample] = []
    trace_rows: list[dict[str, Any]] = []

    for item in items:
        intent, retrieved_contexts = await _collect_retrieved_contexts(item.query, settings)
        chat_response = await run_turn(
            messages=[ChatMessage(role="user", content=item.query)],
            session_id=f"ragas-{item.id}",
            use_tools=True,
            filters=None,
            settings=settings,
            trace_id=f"ragas-{item.id}",
        )
        samples.append(
            SingleTurnSample(
                user_input=item.query,
                response=chat_response.reply,
                retrieved_contexts=retrieved_contexts,
                reference=item.expected_answer,
                reference_contexts=item.expected_contexts,
            )
        )
        trace_rows.append(
            {
                "id": item.id,
                "category": item.category,
                "query": item.query,
                "intent": intent,
                "answer_source": chat_response.answer_source,
                "retrieved_context_count": len(retrieved_contexts),
                "reference_context_count": len(item.expected_contexts),
                "response_preview": chat_response.reply[:180],
            }
        )

    return samples, trace_rows


def _safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return round(mean(clean), 4) if clean else None


def _build_summary_rows(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    averages: dict[str, float | None] = {}
    for metric in METRIC_COLUMNS:
        metric_values = [
            float(row[metric])
            for row in per_query_rows
            if isinstance(row.get(metric), (float, int)) and not math.isnan(float(row[metric]))
        ]
        averages[metric] = _safe_mean(metric_values)

    by_category: dict[str, dict[str, Any]] = {}
    for row in per_query_rows:
        bucket = by_category.setdefault(row["category"], {metric: [] for metric in METRIC_COLUMNS})
        for metric in METRIC_COLUMNS:
            value = row.get(metric)
            if isinstance(value, (float, int)) and not math.isnan(float(value)):
                bucket[metric].append(float(value))

    category_averages = {
        category: {metric: _safe_mean(values[metric]) for metric in METRIC_COLUMNS}
        for category, values in by_category.items()
    }

    weak_queries = [
        {
            "id": row["id"],
            "category": row["category"],
            "query": row["query"],
            "faithfulness": row.get("faithfulness"),
            "answer_relevancy": row.get("answer_relevancy"),
            "context_precision": row.get("context_precision"),
            "context_recall": row.get("context_recall"),
            "answer_source": row.get("answer_source"),
            "retrieved_context_count": row.get("retrieved_context_count"),
        }
        for row in per_query_rows
        if (
            isinstance(row.get("faithfulness"), (float, int))
            and float(row["faithfulness"]) < 0.7
        )
        or (
            isinstance(row.get("answer_relevancy"), (float, int))
            and float(row["answer_relevancy"]) < 0.7
        )
        or (
            isinstance(row.get("context_precision"), (float, int))
            and float(row["context_precision"]) < 0.6
        )
        or (
            isinstance(row.get("context_recall"), (float, int))
            and float(row["context_recall"]) < 0.6
        )
    ]

    return {
        "overall_averages": averages,
        "category_averages": category_averages,
        "weak_queries": weak_queries,
    }


def _write_reports(
    *,
    report_dir: Path,
    dataset_path: Path,
    per_query_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"ragas_report_{timestamp}.json"
    md_path = report_dir / f"ragas_report_{timestamp}.md"

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "query_count": len(per_query_rows),
        "metrics": list(METRIC_COLUMNS),
        "summary": summary,
        "per_query": per_query_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overall = summary["overall_averages"]
    markdown = "\n".join(
        [
            "# RAGAS Evaluation Report",
            "",
            f"- Generated at: `{payload['generated_at_utc']}`",
            f"- Dataset: `{dataset_path}`",
            f"- Queries evaluated: `{len(per_query_rows)}`",
            "",
            "## Average metrics",
            "",
            f"- Faithfulness: `{overall['faithfulness']}`",
            f"- Answer relevancy: `{overall['answer_relevancy']}`",
            f"- Context precision: `{overall['context_precision']}`",
            f"- Context recall: `{overall['context_recall']}`",
            "",
            "## Weak queries (threshold-based)",
            "",
            f"- Count: `{len(summary['weak_queries'])}`",
            "",
            "See JSON report for full per-query rows and category breakdown.",
        ]
    )
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


async def _async_main(args: argparse.Namespace) -> None:
    settings = get_settings()
    dataset_items = _load_dataset(args.dataset)
    samples, trace_rows = await _evaluate_samples(dataset_items, settings)

    eval_dataset = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    )
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )
    result_df = result.to_pandas()
    metric_rows = result_df.to_dict(orient="records")

    per_query_rows: list[dict[str, Any]] = []
    for trace_row, metric_row in zip(trace_rows, metric_rows, strict=True):
        row: dict[str, Any] = {**trace_row}
        for metric_name in METRIC_COLUMNS:
            value = metric_row.get(metric_name)
            if isinstance(value, (float, int)) and not math.isnan(float(value)):
                row[metric_name] = round(float(value), 4)
            else:
                row[metric_name] = None
        per_query_rows.append(row)

    summary = _build_summary_rows(per_query_rows)
    json_path, md_path = _write_reports(
        report_dir=args.report_dir,
        dataset_path=args.dataset,
        per_query_rows=per_query_rows,
        summary=summary,
    )

    print("\nRAGAS per-query metrics")
    for row in per_query_rows:
        print(
            f"- {row['id']} [{row['category']}] | "
            f"faithfulness={row['faithfulness']} "
            f"answer_relevancy={row['answer_relevancy']} "
            f"context_precision={row['context_precision']} "
            f"context_recall={row['context_recall']}"
        )

    overall = summary["overall_averages"]
    print("\nRAGAS averages")
    print(
        f"- faithfulness={overall['faithfulness']} "
        f"answer_relevancy={overall['answer_relevancy']} "
        f"context_precision={overall['context_precision']} "
        f"context_recall={overall['context_recall']}"
    )
    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
