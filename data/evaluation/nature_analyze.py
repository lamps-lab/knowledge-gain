import ast
from urllib.parse import urlparse, urldefrag
import pandas as pd

DATA_PATH = "22-0126.csv"          # or 22_0126.csv
PUBLISHERS_PATH = "04_publishers.csv"
CUTOFF_DATE = pd.to_datetime("2024-12-31")


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


def load_publishers(publishers_path: str):
    publishers_df = pd.read_csv(publishers_path)

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
            "norm_url": norm_url,
            "host": host,
        })

    records = sorted(records, key=lambda r: len(r["norm_url"]), reverse=True)
    return records


def match_link_to_publisher(link: str, publisher_records):
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


df = pd.read_csv(DATA_PATH)
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date_parsed"].notna() & (df["date_parsed"] <= CUTOFF_DATE)].copy()

publisher_records = load_publishers(PUBLISHERS_PATH)

first_match_publishers = []

for _, row in df.iterrows():
    links = [clean_link(x) for x in parse_links(row.get("all_links", ""))]
    links = [x for x in links if x]

    first_match = None
    for link in links:
        pub = match_link_to_publisher(link, publisher_records)
        if pub is not None:
            first_match = pub
            break

    first_match_publishers.append(first_match)

df["first_match_publisher"] = first_match_publishers

springer = df[df["first_match_publisher"] == "Springer Nature"].copy()

print("\nSpringer Nature rows:", len(springer))
print("\nSpringer Nature category distribution:")
print(springer["category"].value_counts(dropna=False).to_string())

print("\nShare by category:")
print((springer["category"].value_counts(normalize=True, dropna=False) * 100).round(2).to_string())