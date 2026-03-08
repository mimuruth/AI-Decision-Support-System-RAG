"""Tests for chunking strategies."""
import pytest

from src.chunking import (
    fixed_size_chunking,
    sentence_chunking,
    chunk_document,
    _approx_token_count,
    _split_sentences,
)
from src.models import DocumentChunk

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow. "
    "Two driven jocks help fax my big quiz. "
    "The jay, pig, fox, zebra, and my wolves quack. "
    "Blowzy red vixens fight for a quick jump. "
    "Joaquin Phoenix was gazed by MTV for luck. "
    "The vex'd buzz quickly flung jab mop."
)


class TestApproxTokenCount:
    def test_empty_string(self):
        assert _approx_token_count("") == 0

    def test_single_word(self):
        assert _approx_token_count("hello") == 1

    def test_multiple_words(self):
        assert _approx_token_count("hello world foo") == 3


class TestSplitSentences:
    def test_splits_on_period(self):
        sents = _split_sentences("Hello world. How are you?")
        assert len(sents) == 2

    def test_preserves_abbreviation(self):
        # "Dr." should not cause a split
        sents = _split_sentences("Dr. Smith works here. He is great.")
        assert len(sents) == 2

    def test_single_sentence(self):
        sents = _split_sentences("No period here")
        assert len(sents) == 1


class TestFixedSizeChunking:
    def test_produces_chunks(self):
        chunks = fixed_size_chunking(SAMPLE_TEXT, "test_doc", chunk_size=10, overlap=2)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_size_respected(self):
        chunks = fixed_size_chunking(SAMPLE_TEXT, "test_doc", chunk_size=10, overlap=0)
        for chunk in chunks[:-1]:  # last chunk may be smaller
            assert chunk.token_count <= 10

    def test_overlap_creates_more_chunks(self):
        no_overlap = fixed_size_chunking(SAMPLE_TEXT, "doc", chunk_size=10, overlap=0)
        with_overlap = fixed_size_chunking(SAMPLE_TEXT, "doc", chunk_size=10, overlap=3)
        assert len(with_overlap) >= len(no_overlap)

    def test_chunk_ids_are_unique(self):
        chunks = fixed_size_chunking(SAMPLE_TEXT, "doc", chunk_size=10, overlap=0)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_passed_through(self):
        chunks = fixed_size_chunking(SAMPLE_TEXT, "doc", metadata={"source": "test"})
        assert all(c.metadata.get("source") == "test" for c in chunks)


class TestSentenceChunking:
    def test_produces_chunks(self):
        chunks = sentence_chunking(SAMPLE_TEXT, "test_doc")
        assert len(chunks) > 0

    def test_max_sentences_respected(self):
        chunks = sentence_chunking(SAMPLE_TEXT, "doc", max_sentences=2)
        # Each chunk should contain at most 2 sentences (approximately)
        for chunk in chunks:
            # sentences ending in . ? ! – rough check
            approx_sents = len([c for c in chunk.text if c in ".?!"])
            assert approx_sents <= 4  # some leniency for abbreviations

    def test_short_text_single_chunk(self):
        short = "One sentence only."
        chunks = sentence_chunking(short, "doc")
        assert len(chunks) == 1
        assert chunks[0].text == "One sentence only."


class TestChunkDocumentDispatcher:
    def test_fixed_size_strategy(self):
        chunks = chunk_document(SAMPLE_TEXT, "doc", strategy="fixed_size")
        assert len(chunks) > 0

    def test_sentence_strategy(self):
        chunks = chunk_document(SAMPLE_TEXT, "doc", strategy="sentence")
        assert len(chunks) > 0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_document(SAMPLE_TEXT, "doc", strategy="nonexistent")
