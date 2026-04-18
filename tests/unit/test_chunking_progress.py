from career_intel.rag.chunking import chunk_text_by_tokens


def test_chunk_text_by_tokens_makes_progress_when_overlap_covers_all_units() -> None:
    text = "Alpha. Beta. Gamma."
    metadata = {"source_id": "test-doc"}

    chunks = chunk_text_by_tokens(text, metadata, chunk_size=2, overlap=2)

    assert chunks
    assert len(chunks) < 10
    assert chunks[0].text
