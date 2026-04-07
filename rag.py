"""
RAG module using ChromaDB for vector storage and retrieval.
Supports OpenAI for answer generation, with a keyword-based fallback.
"""
import logging
from config import CHROMA_DIR, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL
from utils import flatten_detections_for_rag

log = logging.getLogger(__name__)

_chroma_client = None
_collection = None
_embedder = None

COLLECTION_NAME = "cricket_brand_detections"


def _get_chroma():
    global _chroma_client, _collection
    if _chroma_client is not None:
        return _chroma_client, _collection

    import chromadb
    try:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception:
        _chroma_client = chromadb.Client()

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"ChromaDB ready, collection '{COLLECTION_NAME}' has {_collection.count()} docs")
    return _chroma_client, _collection


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        log.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
    except Exception as err:
        log.warning(f"SentenceTransformer not available ({err}), using hash-based fallback.")
        _embedder = None
    return _embedder


def _embed(texts):
    model = _get_embedder()
    if model is not None:
        return model.encode(texts, show_progress_bar=False).tolist()

    # simple hash fallback if no model
    import hashlib
    vecs = []
    for t in texts:
        h = hashlib.sha384(t.encode()).digest()
        vecs.append([float(b) / 255.0 for b in h])
    return vecs


# ---- Store detections in vector DB ----

def store_detections_in_vectordb(detections, match_id):
    _, col = _get_chroma()

    texts = flatten_detections_for_rag(detections)
    if not texts:
        return

    embeddings = _embed(texts)
    ids = [f"{match_id}_det_{i}" for i in range(len(texts))]
    metas = []
    for d in detections:
        metas.append({
            "match_id": d.get("match_id", match_id),
            "brand_name": d.get("brand_name", "unknown"),
            "timestamp": float(d.get("timestamp", 0)),
            "placement": d.get("placement", "unknown"),
            "event": d.get("event", "none"),
            "confidence": float(d.get("confidence", 0)),
            "source": d.get("detection_source", "yolo"),
        })

    col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    log.info(f"Stored {len(texts)} embeddings for match {match_id}")


# ---- Retrieval ----

def retrieve_context(query, n_results=10):
    _, col = _get_chroma()
    if col.count() == 0:
        return []

    q_emb = _embed([query])[0]
    results = col.query(
        query_embeddings=[q_emb],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    out = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"document": doc, "metadata": meta, "distance": dist})
    return out


# ---- Answer a natural language query ----

def answer_query(query):
    context_docs = retrieve_context(query, n_results=15)

    if not context_docs:
        return ("No data found yet. Please process a video first "
                "so the system has detection records to search through.")

    context_text = "\n".join([
        f"- {d['document']} (relevance: {1 - d['distance']:.2f})"
        for d in context_docs
    ])

    if OPENAI_API_KEY:
        return _ask_openai(query, context_text)

    return _build_fallback_answer(query, context_docs)


def _ask_openai(query, context):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        messages = [
            {"role": "system", "content": (
                "You are a cricket broadcast analytics assistant for Jio Hotstar. "
                "Answer questions about brand ad visibility using only the provided data. "
                "Be precise with numbers and timestamps."
            )},
            {"role": "user", "content": (
                f"Detection data:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )},
        ]

        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=messages,
            temperature=0.3, max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as err:
        log.error(f"OpenAI error: {err}")
        return f"LLM error: {err}"


def _build_fallback_answer(query, docs):
    brand_counts = {}
    brand_placements = {}
    brand_events = {}

    for d in docs:
        m = d["metadata"]
        brand = m.get("brand_name", "unknown")
        brand_counts[brand] = brand_counts.get(brand, 0) + 1

        if brand not in brand_placements:
            brand_placements[brand] = {}
        pl = m.get("placement", "unknown")
        brand_placements[brand][pl] = brand_placements[brand].get(pl, 0) + 1

        ev = m.get("event", "none")
        if ev != "none":
            if brand not in brand_events:
                brand_events[brand] = {}
            brand_events[brand][ev] = brand_events[brand].get(ev, 0) + 1

    lines = [f"Query: {query}", f"Found {len(docs)} relevant records.", ""]
    for brand, cnt in sorted(brand_counts.items(), key=lambda x: -x[1]):
        lines.append(f"{brand}: appeared {cnt} times")
        if brand in brand_placements:
            pl_str = ", ".join(f"{k}: {v}" for k, v in brand_placements[brand].items())
            lines.append(f"  Placements: {pl_str}")
        if brand in brand_events:
            ev_str = ", ".join(f"{k}: {v}" for k, v in brand_events[brand].items())
            lines.append(f"  During events: {ev_str}")
        lines.append("")

    return "\n".join(lines)


def get_collection_stats():
    _, col = _get_chroma()
    return {"collection": COLLECTION_NAME, "documents": col.count()}


def clear_collection():
    client, _ = _get_chroma()
    global _collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    log.info("Vector DB collection cleared.")
