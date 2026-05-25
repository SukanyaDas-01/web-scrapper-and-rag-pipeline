import re
import os
import logging
import textwrap
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)

# ── Suppress noisy transformer warnings ──────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Module-level singletons ───────────────────────────────────────────────────
_embedder    = None
_index       = None
_chunks      = []
_generator   = None
_tokenizer   = None
_qa_pipeline = None

# ── Question type triggers ────────────────────────────────────────────────────
FACTUAL_TRIGGERS  = ["who", "when", "where", "which", "how many", "how much"]
OVERVIEW_TRIGGERS = [
    "what is", "what's", "what does", "tell me about", "tell about",
    "describe", "overview", "summary", "about", "explain",
    "main topic", "main idea", "tell me",
]
HOW_WHY_TRIGGERS  = ["how", "why"]

# ── Confidence thresholds ─────────────────────────────────────────────────────
HIGH_CONFIDENCE = 0.60
LOW_CONFIDENCE  = 0.55   # below this → reject as off-topic


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_rag(
    cleaned_text,
    embed_model = "multi-qa-MiniLM-L6-cos-v1",
    gen_model   = "google/flan-t5-large",
    qa_model    = "deepset/roberta-base-squad2",
    chunk_size  = 4,
    overlap     = 2,
):
    """
    Initialises the full RAG pipeline:
      - multi-qa-MiniLM-L6-cos-v1  → embeddings (QA-tuned)
      - flan-t5-large               → overview / summary generation
      - roberta-base-squad2         → factual extractive QA
      - FAISS IndexFlatIP           → cosine similarity search
    """
    global _embedder, _index, _chunks, _generator, _tokenizer, _qa_pipeline

    # 1 ── Embedding model
    print("⏳ Loading embedding model (multi-qa-MiniLM-L6-cos-v1)...")
    _embedder = SentenceTransformer(embed_model)
    print("✅ Embedding model loaded")

    # 2 ── flan-t5-large for overview / summary answers
    print("⏳ Loading flan-t5-large (~780MB, one-time download)...")
    _tokenizer = T5Tokenizer.from_pretrained(gen_model)
    _generator = T5ForConditionalGeneration.from_pretrained(gen_model)
    print("✅ flan-t5-large loaded")

    # 3 ── RoBERTa for factual extractive QA
    print("⏳ Loading roberta-base-squad2 (~500MB, one-time download)...")
    _qa_pipeline = pipeline(
        "question-answering",
        model=qa_model,
        tokenizer=qa_model,
    )
    print("✅ RoBERTa QA model loaded")

    # 4 ── Chunk the cleaned text
    _chunks, _ = _chunk_text(cleaned_text, chunk_size, overlap)

    # 5 ── Embed all chunks
    print(f"⏳ Encoding {len(_chunks)} chunks...")
    embeddings = _embedder.encode(
        _chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 6 ── Build FAISS index
    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)

    print(f"✅ FAISS index ready — {_index.ntotal} vectors, dim={embeddings.shape[1]}")
    print("✅ RAG pipeline ready\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text, chunk_size=4, overlap=2, min_chunk_chars=120):
    """
    Splits cleaned text into overlapping sentence-window chunks.
    Bullet markers are stripped first to avoid fragment answers.
    """
    text = re.sub(r"(?m)^\s*[-•*]\s*", "", text)

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 30
    ]

    if not sentences:
        raise ValueError("No usable sentences found in cleaned_text.")

    chunks = []
    step   = max(1, chunk_size - overlap)

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + chunk_size]).strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)

    print(f"✅ {len(sentences)} sentences → {len(chunks)} chunks")
    return chunks, sentences


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(query, top_k=5, min_score=0.10):
    """Returns top_k chunks with cosine similarity >= min_score."""
    q_vec = _embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
    )
    scores, indices = _index.search(q_vec, top_k)

    return [
        {
            "chunk_id": int(idx),
            "score":    round(float(sc), 4),
            "text":     _chunks[idx],
        }
        for sc, idx in zip(scores[0], indices[0])
        if sc >= min_score
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SENTENCE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _score_sentences(query, context_chunks, top_n=5):
    """
    Re-ranks every individual sentence inside retrieved chunks
    using a blended score:
        0.7 × sentence_cosine_sim + 0.3 × parent_chunk_score
    Returns top_n unique sentences ordered by relevance.
    """
    q_vec  = _embedder.encode(
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


# ═══════════════════════════════════════════════════════════════════════════════
# NOISE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def _format_answer(question, top_sentences):
    """
    Removes citation noise, archive lines, and URL fragments
    from the top sentences before forming the final answer.
    """
    NOISE_PATTERNS = [
        r"^Archived from",
        r"^Retrieved",
        r"^\d{1,2} \w+ \d{4}",
        r"^https?://",
        r'^\".+\"$',
        r"^About \w+[\-—]",
        r"\bArchived\b",
        r"\bdoi:\b",
        r"\bISBN\b",
        r"\bpp\.\b",
        r"Archived from the original",
        r"acquires space in",
        r"first India office",
        r"invest Rs \d+",
        r"crore to set up",
        r"^\w+,\s+\w+\s+\(\d{4}\)\.",    # bibliography entries
        r"Archived on \d+",
    ]

    clean = []
    for s in top_sentences:
        if any(re.search(p, s, re.IGNORECASE) for p in NOISE_PATTERNS):
            continue
        if len(s.split()) < 6:
            continue
        clean.append(s)

    return " ".join(clean) if clean else " ".join(top_sentences[:3])


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def _answer_with_roberta(question, context_chunks):
    """
    Extractive QA using RoBERTa — best for factual questions
    (who, when, where, which, how many).
    Tries each chunk independently and returns the highest-confidence answer.
    """
    best_score, best_answer = -1.0, ""

    for item in context_chunks:
        try:
            result = _qa_pipeline(
                question=question,
                context=item["text"],
                max_answer_len=120,
            )
            if result["score"] > best_score:
                best_score  = result["score"]
                best_answer = result["answer"]
        except Exception:
            continue

    return best_answer.strip(), round(best_score, 4)


def _build_prompt(question, top_sentences):
    """
    Assembles a tight flan-t5 prompt from top-scored sentences.
    Hard-capped at 300 words to stay within the 512-token limit.
    """
    context = " ".join(top_sentences)
    words   = context.split()
    if len(words) > 300:
        context = " ".join(words[:300])

    q_lower = question.lower().strip()

    if any(t in q_lower for t in OVERVIEW_TRIGGERS):
        return (
            f"Write a detailed paragraph summarizing the context below.\n"
            f"Context: {context}\n"
            f"Paragraph:"
        )
    elif any(t in q_lower for t in HOW_WHY_TRIGGERS):
        return (
            f"Answer the question in detail using only the context below.\n"
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


def _generate_with_flant5(prompt):
    """
    Generates a natural language answer using flan-t5-large.
    Dynamically adjusts max_new_tokens based on remaining token budget.
    """
    inputs    = _tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
    )
    input_len          = inputs["input_ids"].shape[1]
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


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC QUERY INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def rag_query(question, top_k=5, show_sources=False):
    print(f"\n{'='*60}")
    print(f"❓  {question}")
    print(f"{'='*60}")

    # ── 1. Retrieve top-k chunks ───────────────────────────────────────────
    results = retrieve(question, top_k=top_k)

    if not results:
        print("\n🤷 I don't have enough information to answer that.\n")
        return

    best_score = results[0]["score"]

    # ── 2. Hard out-of-scope gate ──────────────────────────────────────────
    if best_score < LOW_CONFIDENCE:
        print(
            f"\n🤷 This question doesn't seem related to the scraped page.\n"
            f"   Best match score : {best_score}\n"
            f"   Threshold        : {LOW_CONFIDENCE}\n"
            f"   Try asking something directly about the page content.\n"
        )
        return

    q_lower = question.lower().strip()

    # ── 3. Route to correct answer strategy ───────────────────────────────
    if any(q_lower.startswith(t) or t in q_lower for t in FACTUAL_TRIGGERS):
        # ── Path A: Factual → RoBERTa extractive QA ───────────────────────
        answer, qa_score = _answer_with_roberta(question, results)
        mode = "factual · RoBERTa"

        # Fallback if RoBERTa returns too short or low-confidence answer
        if not answer or len(answer.split()) < 3 or qa_score < 0.10:
            top_sentences = _score_sentences(question, results, top_n=3)
            answer = _format_answer(question, top_sentences)
            mode   = "factual · composed fallback"

    elif any(t in q_lower for t in OVERVIEW_TRIGGERS + HOW_WHY_TRIGGERS):
        # ── Path B: Overview/How/Why → flan-t5-large generation ───────────
        top_sentences = _score_sentences(question, results, top_n=5)
        prompt        = _build_prompt(question, top_sentences)
        answer        = _generate_with_flant5(prompt)
        mode          = "generative · flan-t5-large"

        # Fallback if flan-t5 gives evasive or very short answer
        EVASIVE = ["i don't know", "i do not know", "unknown", "n/a", "none"]
        if len(answer.split()) < 8 or any(e in answer.lower() for e in EVASIVE):
            answer = _format_answer(question, top_sentences)
            mode   = "generative · composed fallback"

    else:
        # ── Path C: General → composed sentence answer ─────────────────────
        top_sentences = _score_sentences(question, results, top_n=4)
        answer        = _format_answer(question, top_sentences)
        mode          = "general · composed"

    # ── 4. Confidence label ────────────────────────────────────────────────
    confidence = "🟢 High" if best_score >= HIGH_CONFIDENCE else "🟡 Medium"

    print(f"\n💡 Answer ({confidence} | {mode}):\n   "
          f"{textwrap.fill(answer, 80, subsequent_indent='   ')}\n")

    # ── 5. Optional sources ────────────────────────────────────────────────
    if show_sources:
        print(f"📄 Sources ({len(results)} chunks retrieved):")
        for r in results:
            print(f"\n  [chunk {r['chunk_id']}]  score={r['score']}")
            print(textwrap.fill(
                r["text"], 78,
                initial_indent="  ",
                subsequent_indent="  ",
            ))