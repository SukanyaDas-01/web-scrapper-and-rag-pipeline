import re
from collections import Counter
from src.scraper import clean_text, is_useful_block, deduplicate_blocks


def summarize_text(text, nlp, max_sentences=5):
    if not text or len(text.strip()) < 150:
        return "Not enough meaningful text was found on the page."

    blocks = []
    for block in text.splitlines():
        block = clean_text(block)
        if not block:
            continue
        block = re.sub(r"^[-•*]\s*", "", block).strip()
        if is_useful_block(block):
            blocks.append(block)

    blocks = deduplicate_blocks(blocks)
    if not blocks:
        return "No meaningful text found for summarization."

    intro = blocks[0]
    candidate_sentences = []

    for block in blocks:
        for sent in nlp(block).sents:
            sentence = re.sub(r"^[-•*]\s*", "", clean_text(sent.text)).strip()
            if is_useful_block(sentence):
                candidate_sentences.append(sentence)

    candidate_sentences = deduplicate_blocks(candidate_sentences)
    if not candidate_sentences:
        return intro

    keyword_tokens = [
        token.lemma_.lower()
        for sentence in candidate_sentences
        for token in nlp(sentence)
        if not token.is_stop and not token.is_punct
        and token.is_alpha and len(token.text) > 2
    ]
    word_freq = Counter(keyword_tokens)
    if not word_freq:
        return " ".join(candidate_sentences[:max_sentences])

    important_topic_words = [
        "architecture", "attention", "transformer", "sequence", "context",
        "dependency", "encoder", "decoder", "embedding", "neural", "language", "vision",
    ]
    weak_start_words = ["so,", "after", "both", "these", "this", "each"]

    scored = []
    for idx, sentence in enumerate(candidate_sentences):
        lower = sentence.lower()
        lemmas = [
            token.lemma_.lower() for token in nlp(sentence)
            if not token.is_stop and not token.is_punct
            and token.is_alpha and len(token.text) > 2
        ]
        if len(lemmas) < 6:
            continue
        score = sum(word_freq.get(l, 0) for l in lemmas) / len(lemmas)
        if any(w in lower for w in important_topic_words): score *= 1.35
        if any(lower.startswith(w) for w in weak_start_words):  score *= 0.55
        if lower.count("token") >= 2:                           score *= 0.75
        score *= 1 - (idx / max(len(candidate_sentences), 1)) * 0.20
        scored.append((score, idx, sentence))

    selected = [intro]
    for _, _, sentence in sorted(scored, reverse=True):
        if sentence != intro and sentence not in selected:
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected)


def get_keywords(text, nlp, limit=10):
    words = [
        token.lemma_.lower() for token in nlp(text)
        if not token.is_stop and not token.is_punct
        and token.is_alpha and len(token.text) > 2
    ]
    return [w for w, _ in Counter(words).most_common(limit)]


def save_summary_to_file(url, cleaned_text, summary, keywords,
                         filename="outputs/summary.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Website Text Summary\n====================\n\n")
        f.write(f"Source URL: {url}\n\n")
        f.write("Summary:\n--------\n")
        f.write(summary + "\n\n")
        f.write("Top Keywords:\n-------------\n")
        f.write(", ".join(keywords) + "\n\n")
        f.write("Cleaned Text Preview:\n---------------------\n")
        f.write(cleaned_text[:2500])