```markdown
# 🌐 Web Scraper & RAG Pipeline

A fully local, no-API-key pipeline that scrapes any webpage, extracts clean content,
generates an extractive summary, and lets you query it using a Retrieval-Augmented
Generation (RAG) system powered by sentence embeddings and FAISS.

---

## 📌 What It Does
```

URL ──► Scrape ──► Clean & Deduplicate ──► Summarize ──► Save to .txt
│
▼
Chunk ──► Embed ──► FAISS Index
│
▼
Query ──► Retrieve ──► Answer

```

---

## 🗂️ Repository Structure

```

web-scrapper-and-rag-pipeline/
│
├── notebooks/
│ └── Web*Scrapping*&\_RAG_pipeline.ipynb ← Full pipeline (run this in Colab)
│
├── src/
│ ├── **init**.py
│ ├── scraper.py ← URL scraping & text cleaning
│ ├── summarizer.py ← Extractive summarization & keywords
│ └── rag.py ← Chunking, FAISS indexing & querying
│
├── outputs/
│ └── summary.txt ← Auto-generated (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md

```

---

## ⚙️ Tech Stack

| Layer | Library |
|---|---|
| Web Scraping | `requests`, `trafilatura`, `BeautifulSoup4` |
| NLP & Keywords | `spaCy` (`en_core_web_sm`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Search | `FAISS` (`faiss-cpu`) |
| Runtime | Google Colab (CPU) |

---

## 🚀 Quickstart

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SukanyaDas-01/web-scrapper-and-rag-pipeline/blob/main/notebooks/Web_Scrapping_%26_RAG_pipeline.ipynb)

### 2. Run cells in order

| Cell | What it does |
|---|---|
| **Cell 1** | Install all dependencies |
| **Cell 2** | Import all libraries |
| **Cell 3** | Load scraper helper functions |
| **Cell 4** | Load summarizer helper functions |
| **Cell 5** | Enter URL → scrape → summarize → download `summary.txt` |
| **Cell 6** | Load sentence-transformer embedder |
| **Cell 7** | Chunk → embed → build FAISS index |
| **Cell 8** | Load RAG query functions |
| **Cell 9** | Interactive Q&A loop |

### 3. Ask questions

```

--- 💬 Ask anything about the scraped page ---

# Your question: What is this page about?

# ❓ What is this page about?

💡 Answer (overview):
Transformer is a neural network architecture used for various
machine learning tasks, especially in natural language processing
and computer vision.

Your question: exit
Exiting RAG session.

````

---

## 🧠 How the RAG Pipeline Works

### Chunking
The cleaned text is split into overlapping sentence windows:
- **Window size**: 4 sentences
- **Overlap**: 2 sentences
- Bullet markers are stripped before chunking

### Embedding
Each chunk is encoded using `all-MiniLM-L6-v2`, a lightweight
80MB model that runs efficiently on CPU with no API key needed.

### Retrieval
FAISS `IndexFlatIP` performs exact cosine similarity search
(inner product on L2-normalised vectors) to find the top-5
most relevant chunks for any query.

### Answer Generation
Query type is detected automatically:

| Query type | Example | Strategy |
|---|---|---|
| **Overview** | "What is this page about?" | Multi-sentence extractive answer from top chunks |
| **Specific** | "How does attention work?" | Single best sentence (blended similarity score) |

---

## 📦 Installation (local)

```bash
git clone https://github.com/SukanyaDas-01/web-scrapper-and-rag-pipeline.git
cd web-scrapper-and-rag-pipeline

pip install -r requirements.txt
python -m spacy download en_core_web_sm
````

---

## 📄 Output Format

`summary.txt` is structured as:

```
Website Text Summary
====================

Source URL: https://example.com

Summary:
--------
<5-sentence extractive summary>

Top Keywords:
-------------
word1, word2, word3, ...

Cleaned Text Preview:
---------------------
<first 2500 characters of cleaned page text>
```

---

## 🔧 Configuration

Key parameters you can tune in the notebook:

| Parameter       | Default | Effect                                                             |
| --------------- | ------- | ------------------------------------------------------------------ |
| `chunk_size`    | `4`     | Sentences per chunk — increase for more context                    |
| `overlap`       | `2`     | Shared sentences between chunks — reduce for less redundancy       |
| `top_k`         | `5`     | Chunks retrieved per query — increase for broader answers          |
| `min_score`     | `0.15`  | Minimum cosine similarity threshold — raise to filter weak matches |
| `max_sentences` | `5`     | Sentences in the extractive summary                                |

---

## ⚠️ Limitations

- **Extractive only** — answers are pulled directly from the page text,
  not generated. Adding a local LLM (e.g. `flan-t5-base`) would enable
  generative answers.
- **Single page** — pipeline scrapes one URL at a time.
- **2500 char preview cap** — `save_summary_to_file` truncates cleaned
  text; RAG uses the full `cleaned_text` variable in memory.
- **JavaScript-heavy sites** — `trafilatura` + `BeautifulSoup` cannot
  execute JS; sites that render content client-side may return little text.

---

## 🛣️ Roadmap

- [ ] Add `flan-t5-base` for generative (not just extractive) answers
- [ ] Multi-URL batch scraping
- [ ] Persistent FAISS index (save/load between sessions)
- [ ] Streamlit or Gradio UI wrapper

---

## 👤 Author

**Sukanya Das**
[GitHub](https://github.com/SukanyaDas-01)

---
