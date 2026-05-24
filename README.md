# 🌐 Web Scraper & RAG Pipeline

A fully local pipeline that scrapes any webpage, extracts clean content, generates an extractive summary, and lets you query it using a Retrieval-Augmented Generation (RAG) system — powered by sentence embeddings, FAISS vector search, and **flan-t5-base** for natural language answer generation. No API key needed.

---

## 📌 What It Does

```
URL ──► Scrape ──► Clean & Deduplicate ──► Summarize ──► Save to .txt
                                                              │
                                                              ▼
                                               Chunk ──► Embed ──► FAISS Index
                                                                         │
                                                                         ▼
                                                         Query ──► Retrieve ──► Prompt
                                                                                   │
                                                                                   ▼
                                                                           flan-t5-base
                                                                                   │
                                                                                   ▼
                                                                         Generated Answer
```

---

## 🗂️ Repository Structure

```
web-scrapper-and-rag-pipeline/
│
├── notebooks/
│   └── Web_Scrapping_&_RAG_pipeline.ipynb   ← Full pipeline (run this in Colab)
│
├── src/
│   ├── __init__.py
│   ├── scraper.py                            ← URL scraping & text cleaning
│   ├── summarizer.py                         ← Extractive summarization & keywords
│   └── rag.py                                ← Chunking, FAISS indexing, prompting & generation
│
├── outputs/
│   └── summary.txt                           ← Auto-generated after each run (gitignored)
│
├── main.py                                   ← Entry point — run this
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Layer             | Library                                      | Purpose                                         |
| ----------------- | -------------------------------------------- | ----------------------------------------------- |
| Web Scraping      | `requests`, `trafilatura`, `BeautifulSoup4`  | Fetch & extract clean page text                 |
| NLP & Keywords    | `spaCy` (`en_core_web_sm`)                   | Tokenization, lemmatization, keyword extraction |
| Embeddings        | `sentence-transformers` (`all-MiniLM-L6-v2`) | Encode chunks into vectors                      |
| Vector Search     | `FAISS` (`faiss-cpu`)                        | Fast cosine similarity retrieval                |
| Answer Generation | `transformers` (`google/flan-t5-base`)       | Generate natural language answers from context  |
| Runtime           | Python 3.10+ / Google Colab (CPU)            | —                                               |

---

## 🚀 Quickstart

### Option 1 — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/SukanyaDas-01/web-scrapper-and-rag-pipeline.git
cd web-scrapper-and-rag-pipeline

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Run the pipeline
python main.py
```

### Option 2 — Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SukanyaDas-01/web-scrapper-and-rag-pipeline/blob/main/notebooks/Web_Scrapping_%26_RAG_pipeline.ipynb)

```python
# Cell 1 — Clone repo
!git clone https://github.com/SukanyaDas-01/web-scrapper-and-rag-pipeline.git

# Cell 2 — Move into folder
import os
os.chdir("/content/web-scrapper-and-rag-pipeline")

# Cell 3 — Install dependencies
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm --quiet

# Cell 4 — Run
!python main.py
```

---

## 🧠 How the RAG Pipeline Works

### 1. Scraping

`trafilatura` extracts the main article text from any URL. A `BeautifulSoup4` fallback handles homepages and JavaScript-light sites. Boilerplate (navbars, footers, ads) is removed before text is returned.

### 2. Chunking

The cleaned text is split into overlapping sentence windows:

- **Window size:** 4 sentences per chunk
- **Overlap:** 2 sentences shared between consecutive chunks
- Bullet markers are stripped before chunking to avoid fragment answers

### 3. Embedding

Each chunk is encoded using `all-MiniLM-L6-v2`, a lightweight 80MB model that runs efficiently on CPU with no API key needed. Embeddings are L2-normalised to enable cosine similarity via dot product.

### 4. Retrieval

FAISS `IndexFlatIP` performs exact inner-product search to find the **top-5 most relevant chunks** for any query. A fallback returns the best available chunks if no result passes the similarity threshold.

### 5. Prompt Building

Retrieved chunks are assembled into a structured prompt for flan-t5. The prompt format adapts based on query type:

| Query Type    | Example                      | Prompt Style                 |
| ------------- | ---------------------------- | ---------------------------- |
| **Overview**  | `"What is this page about?"` | Summary-style instruction    |
| **How / Why** | `"How do the trees bloom?"`  | Detailed explanation request |
| **Specific**  | `"Who is Krumbiegel?"`       | Direct QA format             |

### 6. Answer Generation

`google/flan-t5-base` reads the prompt and **generates a new natural language answer** — it does not copy sentences from the page. Generation uses beam search (`num_beams=4`) with repetition penalty for clean, fluent output.

---

## 💬 Example Output

```
Enter a URL: https://en.wikipedia.org/wiki/Bangalore

✅ URL        : https://en.wikipedia.org/wiki/Bangalore
📝 Text length : 48320 chars
🔑 Keywords    : bangalore, city, india, population, district, urban, karnataka, hub, tech, garden

📄 Summary:
Bangalore, officially known as Bengaluru, is the capital of Karnataka.
It is known as the Silicon Valley of India and the Garden City...

⏳ Building RAG pipeline...
✅ 310 sentences → 154 chunks
✅ FAISS index ready — 154 vectors, dim=384
✅ RAG pipeline ready

--- 💬 Ask anything about the scraped page ---

Your question: What is this page about?
============================================================
💡 Answer:
   Bangalore, officially known as Bengaluru, is the capital city
   of Karnataka, India. It is known as the Silicon Valley of India
   and the Garden City for its parks and greenery.

Your question: How do trees bloom in Bangalore?
============================================================
💡 Answer:
   Trees in Bangalore bloom throughout the year due to the concept
   of serial blooming, introduced by German botanist Gustav Hermann
   Krumbiegel over 100 years ago at the Lalbagh Botanical Garden.

Your question: exit
Exiting. Goodbye!
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**`requirements.txt`**

```
trafilatura
beautifulsoup4
requests
spacy
sentence-transformers
faiss-cpu
numpy
transformers
torch
```

---

## 📄 Output File Format

`outputs/summary.txt` is structured as:

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

Key parameters you can tune in `src/rag.py` and `main.py`:

| Parameter        | Default | Effect                                                       |
| ---------------- | ------- | ------------------------------------------------------------ |
| `chunk_size`     | `4`     | Sentences per chunk — increase for more context              |
| `overlap`        | `2`     | Shared sentences between chunks — reduce for less redundancy |
| `top_k`          | `5`     | Chunks retrieved per query — increase for broader answers    |
| `min_score`      | `0.10`  | Minimum cosine similarity threshold                          |
| `max_new_tokens` | `200`   | Maximum length of generated answer                           |
| `num_beams`      | `4`     | Beam search width — higher = better quality, slower          |
| `max_sentences`  | `5`     | Sentences in the extractive summary                          |

---

## ⚠️ Limitations

- **Paywalled sites** — Medium, NYT, Bloomberg return only a short preview. Use open URLs like Wikipedia, official docs, or public blogs for best results.
- **JavaScript-heavy sites** — `trafilatura` + `BeautifulSoup` cannot execute JS. Sites that render content client-side may return little usable text.
- **flan-t5-base size** — At 250M parameters, flan-t5-base is a small model. Answers improve significantly with more scraped content (aim for 5000+ chars).
- **Single page** — The pipeline scrapes one URL per session.
- **CPU only** — Generation is slower on CPU (~3–5 seconds per answer). GPU would significantly speed this up.

---

## ✅ Best URLs to Test With

| URL                                                                      | Why it works well                 |
| ------------------------------------------------------------------------ | --------------------------------- |
| `https://en.wikipedia.org/wiki/Bangalore`                                | Long, well-structured, fully open |
| `https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)` | Rich technical content            |
| `https://docs.python.org/3/library/functions.html`                       | Clean docs, no paywall            |
| `https://realpython.com/python-f-strings/`                               | Open blog, detailed article       |

---

## 🛣️ Roadmap

- [ ] Upgrade to `flan-t5-large` or `flan-t5-xl` for better answer quality
- [ ] Add GPU support via `torch.cuda`
- [ ] Multi-URL batch scraping
- [ ] Persistent FAISS index (save/load between sessions)
- [ ] Streamlit UI for browser-based interaction
- [ ] Support for PDF and local file input

---

## 👤 Author

**Sukanya Das**
[GitHub](https://github.com/SukanyaDas-01)
