import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import spacy
from src.scraper import scrape_website, extract_clean_text
from src.summarizer import summarize_text, get_keywords, save_summary_to_file
from src.rag import setup_rag, rag_query

def main():
    # ── Step 1: Scrape ────────────────────────────────────────
    url_input = input("Enter a URL: ").strip()

    print("\n⏳ Scraping...")
    nlp             = spacy.load("en_core_web_sm")
    html, final_url = scrape_website(url_input)
    cleaned_text    = extract_clean_text(html, final_url)
    summary         = summarize_text(cleaned_text, nlp, max_sentences=5)
    keywords        = get_keywords(cleaned_text, nlp)

    os.makedirs("outputs", exist_ok=True)
    save_summary_to_file(
        final_url, cleaned_text, summary, keywords,
        filename="outputs/summary.txt"
    )

    print(f"\n✅ URL        : {final_url}")
    print(f"📝 Text length : {len(cleaned_text)} chars")
    print(f"🔑 Keywords    : {', '.join(keywords)}")
    print(f"\n📄 Summary:\n{summary}")
    print("\n✅ Summary saved to outputs/summary.txt")

    # ── Step 2: Build RAG index ───────────────────────────────
    print("\n⏳ Building RAG pipeline...")
    setup_rag(cleaned_text)
    print("✅ RAG pipeline ready\n")

    # ── Step 3: Interactive Q&A ───────────────────────────────
    print("--- 💬 Ask anything about the scraped page ---")
    print("    (type 'exit' to quit)\n")

    while True:
        q = input("Your question: ").strip()
        if not q or q.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break
        rag_query(q, top_k=5, show_sources=True)


if __name__ == "__main__":
    main()