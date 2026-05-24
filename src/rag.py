import re
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Module-level singletons (initialised once by setup_rag) ──────────────────
_embedder = None
_index    = None
_chunks   = []

OVERVIEW_TRIGGERS = [
    "what is", "what's", "what does", "tell me about",
    "describe", "overview", "summary", "about",
    "explain", "main topic", "main idea",
]


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_rag(cleaned_text, model_name="all-MiniLM-L6-v2",
              chunk_size=4, overlap=2):
    """
    Call once after scraping.
    Chunks cleaned_text → embeds → builds FAISS index.
    Returns (chunks, embedder, index).
    """
    global _embedder, _index, _chunks

    _embedder = SentenceTransformer(model_name)
    _chunks, _ = _chunk_text(cleaned_text, chunk_size, overlap)

    print(f"⏳ Encoding {len(_chunks)} chunks...")
    embeddings = _embedder.encode(
        _chunks, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)

    print(f"✅ FAISS index ready — {_index.ntotal} vectors, "
          f"dim={embeddings.shape[1]}")
    return _chunks, _embedder, _index


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text, chunk_size=4, overlap=2, min_chunk_chars=120):
    text = re.sub(r"(?m)^\s*[-•*]\s*", "", text)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 30
    ]
    if not sentences:
        raise ValueError("No usable sentences found in cleaned_text.")

    chunks, step = [], max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + chunk_size]).strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)

    print(f"✅ {len(sentences)} sentences → {len(chunks)} chunks")
    return chunks, sentences


# ── Retrieval ─────────────────────────────────────────────────────────────────

# src/rag.py

def retrieve(query, top_k=5, min_score=0.10):  # lowered from 0.15 → 0.10
    q_vec = _embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    )
    scores, indices = _index.search(q_vec, top_k)
    return [
        {"chunk_id": int(idx), "score": round(float(sc), 4),
         "text": _chunks[idx]}
        for sc, idx in zip(scores[0], indices[0])
        if sc >= min_score
    ]


# ── Answer generation ─────────────────────────────────────────────────────────

def _is_overview_query(q):
    q = q.lower().strip()
    return any(q.startswith(t) or t in q for t in OVERVIEW_TRIGGERS)


def _build_overview_answer(context_chunks, max_sentences=3):
    all_sentences = []
    for item in context_chunks:
        for s in re.split(r"(?<=[.!?])\s+", item["text"]):
            s = s.strip()
            if 40 <= len(s) <= 300:
                all_sentences.append((s, item["score"]))

    seen, unique = set(), []
    for s, sc in all_sentences:
        key = s.lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append((s, sc))

    top_set = {s for s, _ in sorted(unique, key=lambda x: -x[1])[:max_sentences]}
    ordered = [s for s, _ in unique if s in top_set]
    return " ".join(ordered) if ordered else context_chunks[0]["text"]


def _extract_specific_answer(query, context_chunks):
    q_vec = _embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    )
    best_score, best_sentence = -1.0, ""

    for item in context_chunks:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", item["text"])
                 if len(s.strip()) > 25]
        if not sents:
            continue
        s_vecs = _embedder.encode(
            sents, convert_to_numpy=True, normalize_embeddings=True,
        )
        for i, sim in enumerate((s_vecs @ q_vec.T).flatten()):
            combined = 0.7 * sim + 0.3 * item["score"]
            if combined > best_score:
                best_score, best_sentence = combined, sents[i]

    return best_sentence or context_chunks[0]["text"]


# ── Public query interface ────────────────────────────────────────────────────

def rag_query(question, top_k=5, show_sources=False):
    print(f"\n{'='*60}\n❓  {question}\n{'='*60}")

    results = retrieve(question, top_k=top_k)

    # ── Fallback: if no results pass threshold, take top-3 anyway ──
    if not results:
        q_vec = _embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True,
        )
        scores, indices = _index.search(q_vec, min(3, len(_chunks)))
        results = [
            {"chunk_id": int(idx), "score": round(float(sc), 4),
             "text": _chunks[idx]}
            for sc, idx in zip(scores[0], indices[0])
        ]
        print("ℹ️  Low confidence — showing best available context.\n")

    if _is_overview_query(question):
        answer = _build_overview_answer(results, max_sentences=3)
        mode   = "overview"
    else:
        answer = _extract_specific_answer(question, results)
        mode   = "specific"

    print(f"\n💡 Answer ({mode}):\n   "
          f"{textwrap.fill(answer, 80, subsequent_indent='   ')}")

    if show_sources:
        print(f"\n📄 Sources ({len(results)} chunks):")
        for r in results:
            print(f"\n  [chunk {r['chunk_id']}]  score={r['score']}")
            print(textwrap.fill(r["text"], 80,
                                initial_indent="  ", subsequent_indent="  "))