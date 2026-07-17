# src/indexing/hybrid_kg_retriever.py
# Fuses three retrieval signals via Reciprocal Rank Fusion (RRF):
#   1. KG graph traversal (KGRetriever)
#   2. Dense vector search (Qdrant)
#   3. BM25 keyword search
#
# Promoted from the src/notebooks/hybrid_retrieval_test.ipynb prototype into a
# reusable, config-driven module — no notebook-only globals.
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from indexing.kg_retriever import KGRetriever


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class HybridKGRetriever:
    """Retrieves candidates from KG traversal, dense search, and BM25, then
    fuses the three ranked lists with Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        kg_retriever: KGRetriever,
        vector_db,
        embedder,
        chunks: List[Dict[str, Any]],
        rrf_k: int = 60,
        candidate_multiplier: int = 10,
        kg_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.kg_retriever = kg_retriever
        self.vector_db = vector_db
        self.embedder = embedder
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        # Per-signal RRF weights. KG traversal is empirically noisier than dense/BM25
        # on this corpus (see KG_ROADMAP.md Epic 2/3 findings) — equal weighting lets
        # its noise outvote good dense/BM25 candidates. Default 1.0/1.0/1.0 reproduces
        # plain unweighted RRF; callers can down-weight the KG signal instead.
        self.kg_weight = kg_weight
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    # ------------------------------------------------------------------
    def _kg_candidates(self, query: str, n: int) -> List[Dict[str, Any]]:
        results = self.kg_retriever.retrieve(query, top_k=n)
        for r in results:
            r["uid"] = r["payload"]["uid"]
        return results

    def _dense_candidates(self, query: str, n: int) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed_query(query)
        results = self.vector_db.retrieve(query_emb, top_k=n, allowed_types=["text"])
        for r in results:
            r["uid"] = str(r["id"])
        return results

    def _bm25_candidates(self, query: str, n: int) -> List[Dict[str, Any]]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_ids = np.argsort(scores)[::-1][:n]
        results = []
        for idx in top_ids:
            c = self.chunks[int(idx)]
            uid = f"{c['pdf_name']}::{c['chunk_id']}"
            results.append({
                "uid": uid,
                "text": c["text"],
                "score": float(scores[idx]),
                "payload": {
                    "pdf_name": c["pdf_name"],
                    "chunk_id": c["chunk_id"],
                    "page_numbers": c.get("page_numbers"),
                    "uid": uid,
                },
            })
        return results

    def _rrf_fuse(self, weighted_lists: List[Tuple[List[Dict[str, Any]], float]], top_k: int) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        docs: Dict[str, Dict[str, Any]] = {}
        for ranked, weight in weighted_lists:
            for rank, doc in enumerate(ranked):
                uid = doc["uid"]
                scores[uid] = scores.get(uid, 0.0) + weight / (self.rrf_k + rank)
                docs[uid] = doc
        top_uids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [{**docs[u], "score": scores[u]} for u in top_uids]

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        n = top_k * self.candidate_multiplier
        kg_results = self._kg_candidates(query, n)
        dense_results = self._dense_candidates(query, n)
        bm25_results = self._bm25_candidates(query, n)
        return self._rrf_fuse(
            [
                (kg_results, self.kg_weight),
                (dense_results, self.dense_weight),
                (bm25_results, self.bm25_weight),
            ],
            top_k,
        )
