"""
Unit tests for ResolutionRetriever.
Validates isolated retrieval corpus loading and exemplar relevance matching.
"""

import os
import pytest
from src.retrieval.retriever import ResolutionRetriever

def test_retriever_initialization():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(base_dir, "data", "train_retrieval_corpus.csv")
    kb_path = os.path.join(base_dir, "data", "knowledge_base.json")

    retriever = ResolutionRetriever(retrieval_corpus_path=corpus_path, kb_path=kb_path)
    assert len(retriever.historical_pairs) > 0
    assert retriever.tfidf_matrix is not None

def test_retriever_query_matching():
    retriever = ResolutionRetriever()
    result = retriever.retrieve("Wi-Fi dropping frequently on home router", intent="connectivity_features", top_k=3)

    assert result.kb_guide is not None
    assert len(result.historical_exemplars) > 0
    assert len(result.similarity_scores) == len(result.historical_exemplars)
    top_score = result.similarity_scores[0]
    assert 0.0 <= top_score <= 1.0
    assert "agent_text" in result.historical_exemplars[0]
