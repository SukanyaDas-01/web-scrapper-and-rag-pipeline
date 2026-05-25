import re
import os
import logging
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Module-level singletons ───────────────────────────────────────────────────
_embedder  = None
_index     = None
_chunks    = []
_generator = None
_tokenizer = None

OVERVIEW_TRIGGERS = [
    "what is", "what's", "what does", "tell me about", "tell about",
    "describe", "overview", "summary", "about", "explain",
    "main topic", "main idea", "tell me",
]

HOW_WHY_TRIGGERS = ["how", "why", "when", "where", "who"]


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_rag(cleaned_text, embed_model="all-MiniLM-L6-v2",
              gen_model="google/flan-t5-base", chunk_size=4, overlap=2):
    global _embedder, _index, _chunks, _generator, _tokenizer

    print("⏳ Loading embedding model...")
    _embedder = SentenceTransformer(embed_model)

    print("⏳ Loading flan-t5-base generator (~300MB, one-time download)...")
    _tokenizer = T5Tokenizer.from_pretrained(gen_model)
    _generator = T5ForConditionalGeneration.from_pretrained(gen_model)
    print("✅ Generator loaded")

    _chunks, _ = _chunk_text(cleaned_text, chunk_size, overlap)

    print(f"⏳ Encoding {len(_chunks)} chunks...")
    embeddings = _embedder.encode(
        _chunks, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)
    print(f"✅ FAISS index ready — {_index.ntotal} vectors, dim={embeddings.shape[1]}")
    print("✅ RAG pipeline ready\n")


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

def retrieve(query, top_k=5, min_score=0.10):
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


# ── Sentence scorer ───────────────────────────────────────────────────────────

def _score_sentences(query, context_chunks, top_n=5):
    """
    Re-ranks every sentence from retrieved chunks by cosine
    similarity to the query. Returns top_n unique sentences
    ordered by relevance score — used to build both the
    composed answer and the flan-t5 prompt.
    """
    q_vec = _embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    )

    scored = []
    seen   = set()

    for item in context_chunks:
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", item["text"])
            if len(s.strip()) > 30
        ]
        if not sents:
            continue

        s_vecs = _embedder.encode(
            sents, convert_to_numpy=True, normalize_embeddings=True,
        )
        sims = (s_vecs @ q_vec.T).flatten()

        for i, sim in enumerate(sims):
            combined = float(0.7 * sim + 0.3 * item["score"])
            key      = sents[i].lower()[:80]
            if key not in seen:
                seen.add(key)
                scored.append((combined, sents[i]))

    scored.sort(reverse=True)
    return [s for _, s in scored[:top_n]]


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(question, top_sentences):
    """
    Builds a tight prompt from the top scored sentences.
    Strictly capped at 300 words so flan-t5 never overflows.
    """
    context = " ".join(top_sentences)
    words   = context.split()
    if len(words) > 300:
        context = " ".join(words[:300])

    q_lower = question.lower().strip()

    if any(t in q_lower for t in OVERVIEW_TRIGGERS):
        return (
            f"Write a detailed paragraph summarizing the following context.\n"
            f"Context: {context}\n"
            f"Paragraph:"
        )
    elif any(t in q_lower for t in HOW_WHY_TRIGGERS):
        return (
            f"Answer the question in detail using the context.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    else:
        return (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )


# ── flan-t5 generation ────────────────────────────────────────────────────────

def _generate(prompt):
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_len         = inputs["input_ids"].shape[1]
    allowed_new_tokens = max(80, 512 - input_len)

    outputs = _generator.generate(
        **inputs,
        max_new_tokens=allowed_new_tokens,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=2.5,
        length_penalty=2.0,
    )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ── Composed answer (fallback when flan-t5 gives short/bad answer) ────────────

def _compose_answer(top_sentences, min_sentences=3):
    """
    Joins the top scored sentences into a clean paragraph.
    Used as fallback when flan-t5 output is too short.
    """
    # Re-order by original chunk order for readability
    answer = " ".join(top_sentences[:min_sentences])
    return answer


# ── Format answer from top sentences ─────────────────────────────────────────

def _format_answer(question, top_sentences):
    """
    Builds a clean, readable answer directly from top scored sentences.
    Filters out citation-style noise and formats into a paragraph.
    """
    NOISE_PATTERNS = [
        r"^Archived from",
        r"^Retrieved",
        r"^\d{1,2} \w+ \d{4}",        # dates like "1 July 2021"
        r"^https?://",                  # URLs
        r'^\".+\"$',                    # pure quoted titles
        r"^About \w+—",                 # "About Pune—District..."
        r"\bArchived\b",
        r"\bdoi:\b",
        r"\bISBN\b",
        r"\bpp\.\b",
    ]

    clean_sentences = []
    for s in top_sentences:
        if any(re.search(p, s, re.IGNORECASE) for p in NOISE_PATTERNS):
            continue
        if len(s.split()) < 6:
            continue
        clean_sentences.append(s)

    if not clean_sentences:
        return " ".join(top_sentences[:3])

    return " ".join(clean_sentences)


# ── Public query interface ────────────────────────────────────────────────────

def rag_query(question, top_k=5, show_sources=False):
    print(f"\n{'='*60}")
    print(f"❓  {question}")
    print(f"{'='*60}")

    # 1. Retrieve
    results = retrieve(question, top_k=top_k)

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
        print("ℹ️  Low confidence — using best available context.\n")

    # 2. Score and rank individual sentences
    top_sentences = _score_sentences(question, results, top_n=5)

    # 3. Build clean answer directly from top sentences
    answer = _format_answer(question, top_sentences)

    print(f"\n💡 Answer:\n   "
          f"{textwrap.fill(answer, 80, subsequent_indent='   ')}\n")

    if show_sources:
        print(f"📄 Sources:")
        for r in results:
            print(f"\n  [chunk {r['chunk_id']}]  score={r['score']}")
            print(textwrap.fill(r["text"], 78,
                                initial_indent="  ",
                                subsequent_indent="  "))