"""
Semantic retrieval from ChromaDB.
Uses a two-pass strategy: broad pass + work_narrative-filtered pass,
deduplicated and ranked by cosine distance.
"""
from ingest.embedder import get_collection


def retrieve_relevant_chunks(
    query: str,
    n_total: int = 8,
    n_narrative: int = 4,
) -> list[dict]:
    """
    Retrieve the most relevant career experience chunks for a query.

    Two-pass strategy:
    1. Broad pass (n_total results, no filter) — catches skills, positioning, peer feedback
    2. Narrative-filtered pass (n_narrative results, work_narrative only) — ensures
       project-level stories are always represented

    Returns deduplicated list sorted by distance (lowest = most similar).
    Each item: {text, metadata, distance}
    """
    collection = get_collection()

    # Pass 1: broad
    broad = collection.query(
        query_texts=[query],
        n_results=n_total,
        include=["documents", "metadatas", "distances"],
    )

    # Pass 2: narrative only
    try:
        narrative = collection.query(
            query_texts=[query],
            n_results=n_narrative,
            where={"source_type": "work_narrative"},
            include=["documents", "metadatas", "distances"],
        )
        narrative_docs = narrative["documents"][0]
        narrative_metas = narrative["metadatas"][0]
        narrative_dists = narrative["distances"][0]
    except Exception:
        # Collection may not have work_narrative chunks yet
        narrative_docs, narrative_metas, narrative_dists = [], [], []

    # Merge and deduplicate by text content
    seen: set[str] = set()
    results: list[dict] = []

    for doc, meta, dist in zip(
        broad["documents"][0],
        broad["metadatas"][0],
        broad["distances"][0],
    ):
        if doc not in seen:
            seen.add(doc)
            results.append({"text": doc, "metadata": meta, "distance": dist})

    for doc, meta, dist in zip(narrative_docs, narrative_metas, narrative_dists):
        if doc not in seen:
            seen.add(doc)
            results.append({"text": doc, "metadata": meta, "distance": dist})

    # Sort by distance (cosine: lower = more similar)
    results.sort(key=lambda x: x["distance"])

    return results
