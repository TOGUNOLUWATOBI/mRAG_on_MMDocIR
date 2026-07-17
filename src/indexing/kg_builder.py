# src/indexing/kg_builder.py
# Builds a NetworkX knowledge graph from preprocessed chunks.
# Uses an Ollama LLM to extract (subject, relation, object) triples from each chunk,
# then constructs a directed graph where nodes are entities and edges are relations.
import json
import logging
import re
from typing import Any, Dict, List

import networkx as nx
from ollama import Client
from tqdm import tqdm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a knowledge extraction system. "
    "Extract factual entity-relation-entity triples from text. "
    "Entities should be specific nouns (companies, people, products, numbers, dates, concepts). "
    "Relations should be concise verb phrases. "
    "Only extract triples clearly supported by the text. "
    "Return ONLY valid JSON — no markdown, no explanation."
)

_USER_TEMPLATE = (
    "Extract all entities and relationships from the text below as triples.\n"
    'Return a JSON array of objects with keys "subject", "relation", "object".\n'
    'Example: [{{"subject": "Apple", "relation": "reported revenue of", "object": "$94.9B"}}]\n\n'
    "Text:\n{text}\n\nJSON:"
)


class KGBuilder:
    """
    Extracts (subject, relation, object) triples from text chunks via an Ollama LLM
    and assembles them into a NetworkX DiGraph.

    Node attributes:
        label (str)               — display name (original casing)
        source_chunk_uids (list)  — chunk UIDs where this entity was mentioned

    Edge attributes:
        relation (str)            — relation label from first extraction
        source_chunk_uids (list)  — chunk UIDs where this triple was found
    """

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = Client(host=base_url.rstrip("/"), headers=headers, timeout=120)
        self.model = model

    # ------------------------------------------------------------------
    def _extract_triples(self, text: str, chunk_uid: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text[:2000])},
        ]
        try:
            resp = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.0, "top_p": 0.1, "num_predict": 1024},
                stream=False,
                think=False,
            )
            raw = getattr(getattr(resp, "message", None), "content", "") or ""
            # Strip markdown fences if the model wrapped its output
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            triples = json.loads(raw)
            if not isinstance(triples, list):
                return []
            return [
                t for t in triples
                if isinstance(t, dict)
                and all(k in t for k in ("subject", "relation", "object"))
                and all(isinstance(t[k], str) for k in ("subject", "relation", "object"))
            ]
        except Exception as exc:
            logger.warning("Triple extraction failed for chunk %s: %s", chunk_uid, exc)
            return []

    # ------------------------------------------------------------------
    def build_graph(self, chunks: List[Dict[str, Any]]) -> nx.DiGraph:
        """Extract triples from every chunk and return the assembled DiGraph."""
        graph = nx.DiGraph()

        for chunk in tqdm(chunks, desc="Building knowledge graph"):
            uid = f"{chunk['pdf_name']}::{chunk['chunk_id']}"
            triples = self._extract_triples(chunk["text"], uid)

            for triple in triples:
                subj = triple["subject"].strip()
                obj = triple["object"].strip()
                rel = triple["relation"].strip()
                if not subj or not obj or not rel:
                    continue

                for entity in (subj, obj):
                    if entity not in graph:
                        graph.add_node(entity, label=entity, source_chunk_uids=[])
                    if uid not in graph.nodes[entity]["source_chunk_uids"]:
                        graph.nodes[entity]["source_chunk_uids"].append(uid)

                if graph.has_edge(subj, obj):
                    if uid not in graph.edges[subj, obj]["source_chunk_uids"]:
                        graph.edges[subj, obj]["source_chunk_uids"].append(uid)
                else:
                    graph.add_edge(subj, obj, relation=rel, source_chunk_uids=[uid])

        logger.info(
            "Graph built: %d nodes, %d edges from %d chunks",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(chunks),
        )
        return graph
