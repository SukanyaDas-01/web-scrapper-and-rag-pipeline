import re
import logging
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logging.getLogger("trafilatura").setLevel(logging.ERROR)

USER_AGENT = "Mozilla/5.0 (compatible; EducationalSummaryBot/1.0)"

BOILERPLATE_KEYWORDS = [
    "click here", "read more", "scroll through", "come visit",
    "all rights reserved", "copyright", "privacy policy",
    "terms and conditions", "follow us", "facebook", "twitter",
    "instagram", "youtube", "login", "sign in", "subscribe",
    "newsletter", "image", "menu", "home about",
]

NOISY_SECTION_WORDS = [
    "nav", "navbar", "menu", "footer", "header", "sidebar",
    "carousel", "slider", "gallery", "breadcrumb", "social",
    "popup", "modal", "cookie", "advertisement", "ads",
]


def normalize_url(url):
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError("Invalid URL")
    return url


def scrape_website(url):
    url = normalize_url(url)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text, response.url


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_useful_block(text):
    text = clean_text(text)
    lower = text.lower()
    if len(text) < 45:                                          return False
    if any(w in lower for w in BOILERPLATE_KEYWORDS):          return False
    words = text.split()
    if len(words) < 7:                                          return False
    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars / max(len(text), 1) < 0.45:                 return False
    if len(set(words)) < max(4, len(words) * 0.35):            return False
    return True


def fingerprint(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()[:220]


def deduplicate_blocks(blocks):
    seen, unique = set(), []
    for block in blocks:
        block = clean_text(block)
        fp = fingerprint(block)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        unique.append(block)
    return unique


def remove_noisy_sections(soup):
    for tag in soup(["script", "style", "noscript", "svg", "canvas",
                     "iframe", "form", "button", "input"]):
        tag.decompose()
    to_remove = []
    for el in soup.find_all(True):
        if not hasattr(el, "attrs") or el.attrs is None:
            continue
        info = " ".join([
            el.get("id") or "",
            " ".join(el.get("class") or []),
            el.get("role") or "",
        ]).lower()
        if any(w in info for w in NOISY_SECTION_WORDS):
            to_remove.append(el)
    for el in to_remove:
        try:    el.decompose()
        except: pass


def extract_clean_text(html, url):
    extracted = trafilatura.extract(
        html, url=url, include_comments=False,
        include_tables=False, favor_precision=True, output_format="txt"
    )
    blocks = []
    if extracted:
        for line in extracted.splitlines():
            line = clean_text(line)
            if is_useful_block(line):
                blocks.append(line)

    if len(" ".join(blocks)) < 500:
        soup = BeautifulSoup(html, "html.parser")
        remove_noisy_sections(soup)
        main = (
            soup.find("main") or soup.find("article")
            or soup.select_one("[role='main']") or soup.body or soup
        )
        for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
            line = clean_text(tag.get_text(" ", strip=True))
            if is_useful_block(line):
                blocks.append(line)

    return "\n".join(deduplicate_blocks(blocks))