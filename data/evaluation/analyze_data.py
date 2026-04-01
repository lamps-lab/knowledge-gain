#!/usr/bin/env python3

import ast
from collections import Counter
from urllib.parse import urlparse, urldefrag

import pandas as pd


# =========================
# Configuration
# =========================
DATA_PATH = "22-0126.csv"
PUBLISHERS_PATH = "04_publishers.csv"
CUTOFF_DATE = pd.to_datetime("2024-12-31")

OUTPUT_ALL = "publisher_distribution_all.csv"
OUTPUT_CUTOFF = "publisher_distribution_cutoff.csv"
OUTPUT_UNMATCHED = "unmatched_hosts.csv"


# =========================
# URL helpers
# =========================
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
def load_publishers(publishers_path: str):
    publishers_df = pd.read_csv(publishers_path)

    required = {"PUBLISHER_NAME", "PUBLISHER_URL"}
    missing = required - set(publishers_df.columns)
    if missing:
        raise KeyError(f"04_publishers.csv is missing columns: {sorted(missing)}")

    records = []

    for _, row in publishers_df.iterrows():
        name = str(row["PUBLISHER_NAME"]).strip()
        url = str(row["PUBLISHER_URL"]).strip()

        if not url or url == "nan":
            continue

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

    # More specific publisher URLs first
    records = sorted(records, key=lambda r: len(r["norm_url"]), reverse=True)
    return records


def match_link_to_publisher(link: str, publisher_records):
    """
    Match a link to the most specific publisher URL first.
    Fallback: exact host match only.
    """
    norm_link = normalize_url_for_match(link)
    link_host = normalize_host(link)

    if not norm_link or not link_host:
        return None

    # First pass: exact prefix match against PUBLISHER_URL
    for rec in publisher_records:
        prefix = rec["norm_url"]
        if norm_link == prefix or norm_link.startswith(prefix + "/"):
            return rec["publisher_name"]

    # Second pass: exact host match only
    for rec in publisher_records:
        if link_host == rec["host"]:
            return rec["publisher_name"]

    return None


# =========================
# Analysis
# =========================
def analyze_subset(df_subset, publisher_records):
    rows_any_match = Counter()
    rows_first_match = Counter()
    matched_links = Counter()
    unmatched_hosts = Counter()

    total_rows = 0
    rows_with_any_match = 0

    for _, row in df_subset.iterrows():
        total_rows += 1
        links = [clean_link(x) for x in parse_links(row.get("all_links", ""))]
        links = [x for x in links if x]

        seen_publishers_in_row = set()
        first_match = None

        for link in links:
            publisher_name = match_link_to_publisher(link, publisher_records)

            if publisher_name:
                matched_links[publisher_name] += 1
                seen_publishers_in_row.add(publisher_name)
                if first_match is None:
                    first_match = publisher_name
            else:
                host = normalize_host(link)
                if host:
                    unmatched_hosts[host] += 1

        if seen_publishers_in_row:
            rows_with_any_match += 1

        for publisher_name in seen_publishers_in_row:
            rows_any_match[publisher_name] += 1

        if first_match is not None:
            rows_first_match[first_match] += 1

    dist_rows = []
    all_publishers = set(rows_any_match) | set(rows_first_match) | set(matched_links)

    for publisher_name in all_publishers:
        dist_rows.append({
            "publisher_name": publisher_name,
            "rows_any_match": rows_any_match[publisher_name],
            "rows_first_match": rows_first_match[publisher_name],
            "matched_links": matched_links[publisher_name],
        })

    dist_df = pd.DataFrame(dist_rows)
    if not dist_df.empty:
        dist_df = dist_df.sort_values(
            ["rows_any_match", "rows_first_match", "matched_links", "publisher_name"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)

    unmatched_df = pd.DataFrame(
        [{"host": host, "count": count} for host, count in unmatched_hosts.items()]
    )
    if not unmatched_df.empty:
        unmatched_df = unmatched_df.sort_values(
            ["count", "host"], ascending=[False, True]
        ).reset_index(drop=True)

    summary = {
        "total_rows": total_rows,
        "rows_with_any_match": rows_with_any_match,
        "unique_publishers_matched": len(all_publishers),
    }

    return dist_df, unmatched_df, summary


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(DATA_PATH)
    publisher_records = load_publishers(PUBLISHERS_PATH)

    print(f"{len(df)} news articles")

    # Full dataset
    dist_all, unmatched_all, summary_all = analyze_subset(df, publisher_records)

    # Cutoff subset
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df_cutoff = df[df["date_parsed"].notna() & (df["date_parsed"] <= CUTOFF_DATE)].copy()

    dist_cutoff, unmatched_cutoff, summary_cutoff = analyze_subset(df_cutoff, publisher_records)

    # Save outputs
    dist_all.to_csv(OUTPUT_ALL, index=False)
    dist_cutoff.to_csv(OUTPUT_CUTOFF, index=False)

    unmatched_all.head(200).to_csv("unmatched_hosts_all_top200.csv", index=False)
    unmatched_cutoff.head(200).to_csv("unmatched_hosts_cutoff_top200.csv", index=False)

    print("\n=== FULL DATASET ===")
    print(summary_all)
    if not dist_all.empty:
        print("\nTop publishers (all rows):")
        print(dist_all.head(30).to_string(index=False))

    print("\n=== CUTOFF SUBSET (date <= 2024-12-31) ===")
    print(summary_cutoff)
    if not dist_cutoff.empty:
        print("\nTop publishers (cutoff rows):")
        print(dist_cutoff.head(30).to_string(index=False))

    print("\nSaved:")
    print(f"  {OUTPUT_ALL}")
    print(f"  {OUTPUT_CUTOFF}")
    print("  unmatched_hosts_all_top200.csv")
    print("  unmatched_hosts_cutoff_top200.csv")


if __name__ == "__main__":
    main()