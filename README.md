# Web RAG Summarizer

Scrape any webpage → extract clean text → summarize →
query it with a local RAG pipeline (no API key needed).

## Stack

- **Scraping** : `trafilatura` + `BeautifulSoup`
- **NLP** : `spaCy en_core_web_sm`
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB** : `FAISS (faiss-cpu)`

## Notebook cell execution order

| Cell | File it maps to   | What it does                 |
| ---- | ----------------- | ---------------------------- |
| 1    | —                 | Install dependencies         |
| 2    | —                 | Imports                      |
| 3    | src/scraper.py    | Load scraper functions       |
| 4    | src/summarizer.py | Load summarizer functions    |
| 5    | —                 | Run scraper (input URL here) |
| 6    | src/rag.py        | setup_rag(cleaned_text)      |
| 7    | src/rag.py        | Interactive Q&A loop         |

## Quickstart (Colab)

```python
from src.scraper    import scrape_website, extract_clean_text
from src.summarizer import summarize_text, get_keywords, save_summary_to_file
from src.rag        import setup_rag, rag_query

html, final_url = scrape_website("https://example.com")
cleaned_text    = extract_clean_text(html, final_url)
setup_rag(cleaned_text)
rag_query("What is this page about?")
```
