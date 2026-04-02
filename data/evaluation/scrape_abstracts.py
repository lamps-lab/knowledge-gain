#!/usr/bin/env python3

import json
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================
# Configuration
# =========================
INPUT_CANDIDATES = "candidate_abstract_links_flat.csv"
OUTPUT_JSON = "collected_abstracts.json"
OUTPUT_CSV = "collected_abstracts.csv"

REQUEST_TIMEOUT = 20
SLEEP_SECONDS = 0.3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(abstract|summary)\s*[:.\-]?\s*", "", text, flags=re.I)
    return text.strip()


def fetch_soup(url: str):
    time.sleep(SLEEP_SECONDS)
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def get_host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def text_from_node(node) -> str:
    if not node:
        return ""
    return clean_text(node.get_text(" ", strip=True))


# =========================
# MDPI
# =========================
def extract_mdpi_abstract(soup: BeautifulSoup) -> str:
    selectors = [
        "div.art-abstract.art-abstract-new.in-tab.hypothesis_container",
        "div.art-abstract.art-abstract-new",
        "div.art-abstract",
    ]

    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            text = text_from_node(node)
            if len(text) >= 120:
                return text

    return ""


# =========================
# Nature / Springer Nature
# =========================
def extract_nature_abstract(soup: BeautifulSoup) -> str:
    # First: sections explicitly marked as Abstract
    for section in soup.find_all("section"):
        data_title = (section.get("data-title") or "").strip().lower()
        if data_title == "abstract":
            content = section.find("div", id=lambda x: x and x.endswith("-content"))
            if content:
                text = text_from_node(content)
                if len(text) >= 120:
                    return text

            section_div = section.find("div", class_=lambda c: c and "c-article-section" in str(c))
            if section_div:
                text = text_from_node(section_div)
                if len(text) >= 120:
                    return text

            text = text_from_node(section)
            if len(text) >= 120:
                return text

    # Second: section whose heading is literally Abstract
    for section in soup.find_all("section"):
        heading = section.find(["h1", "h2", "h3", "h4"])
        if heading and heading.get_text(" ", strip=True).strip().lower() == "abstract":
            content = section.find("div", id=lambda x: x and x.endswith("-content"))
            if content:
                text = text_from_node(content)
                if len(text) >= 120:
                    return text

            text = text_from_node(section)
            if len(text) >= 120:
                return text

    # Older fallback patterns
    fallback_selectors = [
        "section#abstract",
        "div#abstract",
        "section.abstract",
        "div.abstract",
    ]

    for sel in fallback_selectors:
        for node in soup.select(sel):
            text = text_from_node(node)
            if len(text) >= 120:
                return text

    return ""


# =========================
# Frontiers
# =========================
def extract_frontiers_abstract(soup: BeautifulSoup) -> str:
    # Main Frontiers V4 layout:
    # <div class="ArticleDetailsV4__main__content">
    #   <div class="ArticleContent">
    #     <div id="h1">
    #       <h2>Abstract</h2>
    #       <p>...</p>
    #     </div>

    main = soup.select_one("div.ArticleDetailsV4__main__content div.ArticleContent")
    if main:
        for block in main.find_all("div", recursive=True):
            h2 = block.find("h2")
            if h2 and h2.get_text(" ", strip=True).strip().lower() == "abstract":
                parts = []

                for node in block.find_all(["p", "li"]):
                    txt = text_from_node(node)
                    if txt:
                        parts.append(txt)

                text = clean_text(" ".join(parts))
                if len(text) >= 120:
                    return text

    # Fallback: find any h2 with Abstract and read from its parent block
    h2 = soup.find(
        lambda tag: tag.name == "h2"
        and tag.get_text(" ", strip=True).strip().lower() == "abstract"
    )
    if h2:
        parent = h2.parent
        if parent:
            parts = []
            for node in parent.find_all(["p", "li"]):
                txt = text_from_node(node)
                if txt:
                    parts.append(txt)

            text = clean_text(" ".join(parts))
            if len(text) >= 120:
                return text

    return ""


# =========================
# Generic router
# =========================
def extract_abstract(url: str, publisher_name: str, soup: BeautifulSoup) -> str:
    publisher_name = str(publisher_name).strip().lower()
    host = get_host(url)

    if publisher_name == "mdpi" or "mdpi.com" in host:
        return extract_mdpi_abstract(soup)

    if publisher_name == "springer nature" or "nature.com" in host or "springer.com" in host:
        return extract_nature_abstract(soup)

    if publisher_name == "frontiers" or "frontiersin.org" in host:
        return extract_frontiers_abstract(soup)

    return ""


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CANDIDATES)

    results = []
    seen_urls = set()

    for _, row in df.iterrows():
        url = str(row.get("abstract_url", "")).strip()
        publisher_name = str(row.get("publisher_name", "")).strip()

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        try:
            soup = fetch_soup(url)
            abstract = extract_abstract(url, publisher_name, soup)
        except Exception:
            continue

        if not abstract:
            continue

        result = {
            "id": row.get("id"),
            "date": row.get("date"),
            "news": row.get("news", ""),
            "category": row.get("category", ""),
            "news_text": row.get("news_text", ""),
            "news_url": row.get("news_url", ""),
            "publisher_name": publisher_name,
            "abstract_url": url,
            "abstract": abstract,
        }
        results.append(result)

        print(f"[OK] {publisher_name} | {url}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Collected abstracts: {len(results)}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()