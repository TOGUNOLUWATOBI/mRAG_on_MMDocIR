# src/query_techniques/kg_multihop.py
# Query technique that routes retrieval through the KG+Dense+BM25 hybrid
# retriever instead of the standard BM25+dense HybridRetriever. Selectable by
# the agentic query-rewriter agent when it judges a question to be multi-hop/
# relational (see KG_ROADMAP.md Epic 5).
from typing import Any, Dict, List

from .base import QueryTechnique


class KGMultihopRetrieval(QueryTechnique):
    """Wraps a HybridKGRetriever (KG traversal + dense + BM25 via RRF) behind
    the standard QueryTechnique interface so the agentic query rewriter can
    select it like any of the other 8 techniques."""

    def __init__(self, embedder, retriever, generator, config=None):
        # `retriever` here is a HybridKGRetriever instance — a different object
        # from the plain BM25+dense HybridRetriever the other techniques share.
        super().__init__(embedder, retriever, generator, config)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(question, top_k=top_k)
