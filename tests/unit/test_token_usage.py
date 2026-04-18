"""Tests for LangChain → API token usage extraction."""

from __future__ import annotations

from types import SimpleNamespace

from career_intel.llm.token_usage import merge_token_usages, usage_from_langchain_message
from career_intel.schemas.api import TokenUsage


def test_usage_from_usage_metadata_openai_shape() -> None:
    message = SimpleNamespace(
        content="hi",
        usage_metadata={"input_tokens": 3, "output_tokens": 7, "total_tokens": 10},
        response_metadata={},
    )
    usage = usage_from_langchain_message(message)
    assert usage == TokenUsage(prompt_tokens=3, completion_tokens=7, total_tokens=10)


def test_usage_from_response_metadata_token_usage() -> None:
    message = SimpleNamespace(
        content="x",
        usage_metadata={},
        response_metadata={
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        },
    )
    usage = usage_from_langchain_message(message)
    assert usage == TokenUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16)


def test_merge_token_usages_sums() -> None:
    a = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    b = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    merged = merge_token_usages(a, b)
    assert merged == TokenUsage(prompt_tokens=11, completion_tokens=22, total_tokens=33)


def test_merge_with_none() -> None:
    u = TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
    assert merge_token_usages(None, u) == u
    assert merge_token_usages(None, None) is None
