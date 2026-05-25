import re
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Module-level singletons ───────────────────────────────────────────────────
_embedder  = None
_index     = None
_chunks    = []
_generator = None
_tokenizer = None

OVERVIEW_TRIGGERS = [
    "what is", "what's", "what does", "tell me about",
    "describe", "overview", "summary", "about",
    "explain", "main topic", "main idea",
]


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_rag(cleaned_text, embed_model="all-MiniLM-L6-v2",
              gen_model="google/flan-t5-base", chunk_size=4, overlap=2):
    """
    Call once after scraping.
    Loads:
      - SentenceTransformer  → embeddings
      - flan-t5-base         → answer generation
      - FAISS index          → vector search
    """
    global _embedder, _index, _chunks, _generator, _tokenizer

    # 1. Load embedding model
    print("⏳ Loading embedding model...")
    _embedder = SentenceTransformer(embed_model)

    # 2. Load flan-t5 generator
    print("⏳ Loading flan-t5-base generator (~300MB, one-time download)...")
    _tokenizer = T5Tokenizer.from_pretrained(gen_model)
    _generator = T5ForConditionalGeneration.from_pretrained(gen_model)
    print("✅ Generator loaded")

    # 3. Chunk text
    _chunks, _ = _chunk_text(cleaned_text, chunk_size, overlap)

    # 4. Embed chunks
    print(f"⏳ Encoding {len(_chunks)} chunks...")
    embeddings = _embedder.encode(
        _chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 5. Build FAISS index
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
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, indices = _index.search(q_vec, top_k)

    results = [
        {
            "chunk_id": int(idx),
            "score":    round(float(sc), 4),
            "text":     _chunks[idx],
        }
        for sc, idx in zip(scores[0], indices[0])
        if sc >= min_score
    ]
    return results


# ── Prompt Builder ────────────────────────────────────────────────────────────

def _build_prompt(question, context_chunks):
    """
    Use only top 2 chunks — flan-t5-base has a hard 512 token limit.
    Exceeding it causes the model to ignore context and hallucinate.
    """
    # Sort by score, take only top 2
    top_chunks = sorted(context_chunks, key=lambda x: -x["score"])[:2]

    context = " ".join([chunk["text"] for chunk in top_chunks])

    # Hard trim to 350 words to stay safely within 512 tokens
    context_words = context.split()
    if len(context_words) > 350:
        context = " ".join(context_words[:350])

    q_lower = question.lower().strip()

    # Overview / summary questions
    if any(t in q_lower for t in ["about", "what is", "overview",
                                   "describe", "tell me about",
                                   "explain", "main topic"]):
        prompt = (
            f"Read the context carefully and write a clear summary.\n"
            f"Context: {context}\n"
            f"Summary:"
        )

    # How / Why questions
    elif any(t in q_lower for t in ["how", "why"]):
        prompt = (
            f"Read the context and answer the question.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    # Tell me about / describe specific topic
    elif "tell" in q_lower or "describe" in q_lower:
        prompt = (
            f"Based only on the context below, describe the topic "
            f"asked in the question.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    # Default direct QA
    else:
        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    return prompt


# ── Generative Answer ─────────────────────────────────────────────────────────

def _generate_answer(prompt, max_new_tokens=200):
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Check how many tokens the prompt actually consumed
    input_len = inputs["input_ids"].shape[1]
    # Leave enough room for the answer
    allowed_new_tokens = min(max_new_tokens, max(50, 512 - input_len))

    outputs = _generator.generate(
        **inputs,
        max_new_tokens=allowed_new_tokens,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=2.5,
        length_penalty=1.5,
    )

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


# ── Public Query Interface ────────────────────────────────────────────────────

def rag_query(question, top_k=5, show_sources=False):
    print(f"\n{'='*60}")
    print(f"❓  {question}")
    print(f"{'='*60}")

    # Retrieve top-k for display
    results = retrieve(question, top_k=top_k)

    # Fallback if nothing passes threshold
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

    # ── Pass only top 2 to prompt to stay within token limit ──
    prompt_chunks = sorted(results, key=lambda x: -x["score"])[:2]
    prompt  = _build_prompt(question, prompt_chunks)
    answer  = _generate_answer(prompt)

    print(f"\n💡 Answer:\n   {textwrap.fill(answer, 80, subsequent_indent='   ')}")

    if show_sources:
        print(f"\n📄 Sources ({len(results)} chunks):")
        for r in results:
            print(f"\n  [chunk {r['chunk_id']}]  score={r['score']}")
            print(textwrap.fill(r["text"], 80,
                                initial_indent="  ", subsequent_indent="  "))