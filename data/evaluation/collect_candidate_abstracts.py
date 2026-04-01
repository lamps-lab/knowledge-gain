#!/usr/bin/env python3

import ast
import json
from urllib.parse import urlparse, urldefrag

import pandas as pd


# =========================
# Configuration
# =========================
DATA_PATH = "22-0126.csv"
PUBLISHERS_PATH = "04_publishers.csv"
CUTOFF_DATE = pd.to_datetime("2024-12-31")

TARGET_PUBLISHERS = {
    "Springer Nature",
    "MDPI",
    "Frontiers",
}

OUTPUT_GROUPED_JSON = "candidate_abstract_links_grouped.json"
OUTPUT_GROUPED_CSV = "candidate_abstract_links_grouped.csv"
OUTPUT_FLAT_CSV = "candidate_abstract_links_flat.csv"

skip = [
    29739, 29866, 29929, 29775, 29873, 29733, 29734, 29958,
    30052, 29943, 29902, 29846, 30086, 30184, 30061, 29756
]


# =========================
# URL helpers
# =========================
def canonicalize_url(url: str) -> str:
    """Keep query params, drop fragment."""
    try:
        return urldefrag(str(url).strip())[0]
    except Exception:
        return str(url).strip()


def normalize_url_for_match(url: str) -> str:
    try:
        url = urldefrag(str(url).strip())[0]
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        return f"{scheme}://{netloc}{path}".lower()
    except Exception:
        return ""


def normalize_host(url: str) -> str:
    try:
        netloc = urlparse(str(url).strip()).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def is_probably_useless_link(url: str) -> bool:
    """
    Very light cleanup only.
    This step is for candidate collection, so we keep it permissive.
    """
    low = str(url).lower().strip()

    if not low:
        return True

    bad_suffixes = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".doc", ".docx", ".xls", ".xlsx", ".zip"
    )
    if low.endswith(bad_suffixes):
        return True

    bad_substrings = [
        "/news-room/",
    ]
    if any(x in low for x in bad_substrings):
        return True

    try:
        parsed = urlparse(low)
        if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
            return True
    except Exception:
        pass

    return False


# =========================
# Input helpers
# =========================
def parse_links(links_str):
    if pd.isna(links_str):
        return []

    try:
        links = ast.literal_eval(links_str)
        if isinstance(links, str):
            return [links]
        if isinstance(links, list):
            return links
        return [str(links)]
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(links_str).split(",") if x.strip()]


def clean_link(link: str) -> str:
    return str(link).strip("[]'\" ")


# =========================
# Publisher matching
# =========================
def load_target_publishers(publishers_path: str, target_publishers: set):
    publishers_df = pd.read_csv(publishers_path)

    required = {"PUBLISHER_NAME", "PUBLISHER_URL"}
    missing = required - set(publishers_df.columns)
    if missing:
        raise KeyError(f"04_publishers.csv is missing columns: {sorted(missing)}")

    publishers_df["PUBLISHER_NAME"] = publishers_df["PUBLISHER_NAME"].astype(str).str.strip()
    publishers_df["PUBLISHER_URL"] = publishers_df["PUBLISHER_URL"].astype(str).str.strip()

    publishers_df = publishers_df[publishers_df["PUBLISHER_NAME"].isin(target_publishers)]
    publishers_df = publishers_df[publishers_df["PUBLISHER_URL"].notna()]
    publishers_df = publishers_df[publishers_df["PUBLISHER_URL"] != ""]
    publishers_df = publishers_df[publishers_df["PUBLISHER_URL"] != "nan"]

    records = []

    for _, row in publishers_df.iterrows():
        name = row["PUBLISHER_NAME"]
        url = row["PUBLISHER_URL"]

        norm_url = normalize_url_for_match(url)
        host = normalize_host(url)

        if not norm_url or not host:
            continue

        records.append({
            "publisher_name": name,
            "publisher_url": url,
            "norm_url": norm_url,
            "host": host,
        })

    records = sorted(records, key=lambda r: len(r["norm_url"]), reverse=True)
    return records


def match_link_to_publisher(link: str, publisher_records):
    """
    Match by most specific PUBLISHER_URL prefix first.
    Fallback: exact host match only.
    """
    norm_link = normalize_url_for_match(link)
    link_host = normalize_host(link)

    if not norm_link or not link_host:
        return None

    for rec in publisher_records:
        prefix = rec["norm_url"]
        if norm_link == prefix or norm_link.startswith(prefix + "/"):
            return rec["publisher_name"]

    for rec in publisher_records:
        if link_host == rec["host"]:
            return rec["publisher_name"]

    return None


# =========================
# Metadata helpers
# =========================
def get_news_field(row):
    for col in ["news", "title", "headline", "name"]:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value and value.lower() != "nan":
                return value
    return ""


def get_news_text_field(row):
    for col in ["plaintext", "news_text", "text", "body"]:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value and value.lower() != "nan":
                return value
    return ""


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(DATA_PATH)
    publisher_records = load_target_publishers(PUBLISHERS_PATH, TARGET_PUBLISHERS)

    print(f"{len(df)} news articles")

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_parsed"].notna() & (df["date_parsed"] <= CUTOFF_DATE)].copy()

    grouped_rows = []
    flat_rows = []
    seen_group_ids = set()

    for _, row in df.iterrows():
        try:
            sample_id = int(row["id"])
        except Exception:
            continue

        if sample_id in skip or sample_id in seen_group_ids:
            continue

        links = [clean_link(x) for x in parse_links(row.get("all_links", ""))]
        links = [x for x in links if x]

        candidate_links = []
        seen_links_in_row = set()

        for raw_link in links:
            link = canonicalize_url(raw_link)

            if is_probably_useless_link(link):
                continue

            publisher_name = match_link_to_publisher(link, publisher_records)
            if publisher_name is None:
                continue

            if link in seen_links_in_row:
                continue

            seen_links_in_row.add(link)
            candidate_links.append({
                "publisher_name": publisher_name,
                "abstract_url": link
            })

        if not candidate_links:
            continue

        date_str = row["date_parsed"].strftime("%Y-%m-%d")
        category = str(row.get("category", "")).strip()
        news = get_news_field(row)
        news_text = get_news_text_field(row)
        news_url = str(row.get("url", "")).strip()

        grouped_rows.append({
            "id": sample_id,
            "date": date_str,
            "news": news,
            "category": category,
            "news_text": news_text,
            "news_url": news_url,
            "abstract_links": [x["abstract_url"] for x in candidate_links],
            "publishers": [x["publisher_name"] for x in candidate_links],
            "n_abstract_links": len(candidate_links),
        })

        for item in candidate_links:
            flat_rows.append({
                "id": sample_id,
                "date": date_str,
                "news": news,
                "category": category,
                "news_text": news_text,
                "news_url": news_url,
                "publisher_name": item["publisher_name"],
                "abstract_url": item["abstract_url"],
            })

            print(f"[{date_str}] {item['publisher_name']} | {item['abstract_url']}")

        seen_group_ids.add(sample_id)

    with open(OUTPUT_GROUPED_JSON, "w", encoding="utf-8") as f:
        json.dump(grouped_rows, f, indent=4, ensure_ascii=False)

    grouped_df = pd.DataFrame(grouped_rows).copy()
    if not grouped_df.empty:
        grouped_df["abstract_links"] = grouped_df["abstract_links"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        grouped_df["publishers"] = grouped_df["publishers"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
    grouped_df.to_csv(OUTPUT_GROUPED_CSV, index=False, encoding="utf-8")

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(OUTPUT_FLAT_CSV, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Grouped rows: {len(grouped_rows)}")
    print(f"Candidate URLs: {len(flat_rows)}")
    print(f"Saved: {OUTPUT_GROUPED_JSON}")
    print(f"Saved: {OUTPUT_GROUPED_CSV}")
    print(f"Saved: {OUTPUT_FLAT_CSV}")


if __name__ == "__main__":
    main()