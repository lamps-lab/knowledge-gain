from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LETTER_RE = re.compile(r"^[A-J]$", re.IGNORECASE)
LEADING_ENUM_RE = re.compile(
    r"""^\s*(?:                                   # START
        (?:q(?:uestion)?\s*\d+\s*[:.)-]) |       # e.g., Q1:  Question 2)  q10.
        (?:\(?\d+\)?\s*[:.)-]) |                 # e.g., 1.  (2)  3)  4: 
        (?:[ivxlcdm]+\s*[:.)-]) |                # roman numerals i. ii) v: 
        (?:[a-j]\s*[:.)-])                       # letter enumerations a) b. c:
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)

DK_CANONICAL_TEXTS = {
    "i do not know the answer",
    "i do not know the answer.",
    "i do not know",
    "i don't know",
    "i dont know",
    "dont know",
    "do not know",
    "dk",
    "idk",
    "not sure",
    "unsure",
}

def nrm_basic(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def strip_leading_enum(s: str) -> str:
    t = nrm_basic(s)
    prev = None
    while prev != t:
        prev = t
        t = LEADING_ENUM_RE.sub("", t)
    return t.strip()

def qnorm(s: str) -> str:
    t = strip_leading_enum(s)
    t = t.rstrip(" .?!:;-")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_dk_text(s: str) -> bool:
    t = nrm_basic(s).rstrip(".")
    return t in DK_CANONICAL_TEXTS

def options_dict_to_list_with_dk_last(opts: Dict[str, str]) -> Tuple[List[str], Dict[str, int]]:
    if not isinstance(opts, dict):
        raise ValueError("options must be a dict of {letter|DK: option_text}")

    non_dk_items = []
    dk_items = []

    for k, v in opts.items():
        if (isinstance(k, str) and k.upper() == "DK") or is_dk_text(v):
            dk_items.append((k, v))
        else:
            non_dk_items.append((k, v))

    def sortkey(item):
        k, _ = item
        if isinstance(k, str) and len(k) == 1 and k.isalpha():
            return (0, k.upper())
        return (1, str(k))

    non_dk_items_sorted = sorted(non_dk_items, key=sortkey)

    ordered = [v for _, v in non_dk_items_sorted]
    for _, v in dk_items:
        if v not in ordered:
            ordered.append(v)

    key2idx: Dict[str, int] = {}
    text2idx: Dict[str, int] = {nrm_basic(txt): i + 1 for i, txt in enumerate(ordered)}

    for k, v in non_dk_items_sorted + dk_items:
        key2idx[k] = text2idx[nrm_basic(v)]
    return ordered, key2idx

# robust answer parsing
# patterns like "A - True", "A) True", "A. True"
RE_PREFIX_LETTER = re.compile(r"^\s*([A-J])\s*[\)\].:\-–—]\s*(.*)$", re.IGNORECASE)
# patterns like "1 - True", "1) False", "2. Foo"
RE_PREFIX_NUM = re.compile(r"^\s*(\d+)\s*[\)\].:\-–—]\s*(.*)$", re.IGNORECASE)
# patterns like "True (A)" or "False [B]"
RE_TRAIL_LETTER = re.compile(r"^(.*?)[\s]*[\(\[]\s*([A-J])\s*[\)\]]\s*$", re.IGNORECASE)
# patterns like "True (1)"
RE_TRAIL_NUM = re.compile(r"^(.*?)[\s]*[\(\[]\s*(\d+)\s*[\)\]]\s*$", re.IGNORECASE)

def extract_letter_or_number_hint(s: str) -> Tuple[Optional[str], Optional[int], str]:
    """
    Try to pull a leading or trailing choice hint from a typical Qualtrics export.
    Returns (letter, number, residual_text).
    Only one of letter/number will be non-None if a hint is found.
    """
    if not isinstance(s, str):
        s = str(s)

    m = RE_PREFIX_LETTER.match(s)
    if m:
        return m.group(1).upper(), None, m.group(2)

    m = RE_PREFIX_NUM.match(s)
    if m:
        return None, int(m.group(1)), m.group(2)

    m = RE_TRAIL_LETTER.match(s)
    if m:
        return m.group(2).upper(), None, m.group(1)

    m = RE_TRAIL_NUM.match(s)
    if m:
        return None, int(m.group(2)), m.group(1)

    return None, None, s

def parse_option_to_index(raw_val, options: List[str]) -> Optional[int]:
    if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
        return None

    s = str(raw_val).strip()

    # 0) DK shortcuts
    if is_dk_text(s) or nrm_basic(s) in {"dk", "idk"}:
        return len(options)

    # 1) Try explicit letter / number hints with residual text
    letter, number, residual = extract_letter_or_number_hint(s)
    if letter and LETTER_RE.match(letter):
        idx = ord(letter.upper()) - ord("A") + 1
        if 1 <= idx <= len(options):
            return idx
    if number is not None:
        if 1 <= number <= len(options):
            return number
        if 0 <= number < len(options):
            return number + 1

    # 2) Plain letter (no delimiter)
    if LETTER_RE.match(s):
        idx = ord(s.upper()) - ord("A") + 1
        if 1 <= idx <= len(options):
            return idx

    # 3) Plain numeric code
    if re.fullmatch(r"-?\d+", s):
        num = int(s)
        if 1 <= num <= len(options):
            return num
        if 0 <= num < len(options):
            return num + 1

    # 4) Exact text match (normalize)
    ns = nrm_basic(s)
    for i, opt in enumerate(options, start=1):
        if nrm_basic(opt) == ns:
            return i

    # 5) Try residual text from hints
    rs = nrm_basic(residual)
    if rs and rs != ns:
        for i, opt in enumerate(options, start=1):
            if nrm_basic(opt) == rs:
                return i

    return None

# builder
def build_final_dataset(data_root: Path, out_path: Path) -> None:
    # Load dataset
    ds_path = data_root / "kgain_dataset.json"
    with open(ds_path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    articles = kg.get("articles", []) if isinstance(kg, dict) else kg

    # Build article signatures
    article_by_media_signature: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}
    per_article_qmeta: Dict[Tuple[str, str], List[Dict]] = {}

    for art in articles:
        article_id = art.get("article_id") or art.get("article-id")
        contents = art.get("contents", {})
        qas = art.get("qas", [])

        q_infos = []
        for q_pos, qa in enumerate(qas, start=1):
            q_text = qa.get("question")
            if not q_text:
                continue
            opts_dict = qa.get("options", {})
            options_list, origkey2newidx = options_dict_to_list_with_dk_last(opts_dict)

            correct_key = qa.get("answer")
            correct_idx = None
            if isinstance(correct_key, str) and correct_key in origkey2newidx:
                correct_idx = int(origkey2newidx[correct_key])
            else:
                if isinstance(correct_key, str):
                    ci = parse_option_to_index(correct_key, options_list)
                    correct_idx = int(ci) if ci is not None else None

            q_infos.append({
                "question_in_set": q_pos,
                "question_text": q_text,
                "question_key": qnorm(q_text),
                "options": options_list,
                "correct_option": correct_idx,
                "correct_answer": (options_list[correct_idx - 1] if isinstance(correct_idx, int) and 1 <= correct_idx <= len(options_list) else None),
            })

        for media in ("news", "abstract", "tweet"):
            content_text = contents.get(media)
            ordered_keys = tuple(q["question_key"] for q in q_infos)
            if len(ordered_keys) != 6:
                continue
            sig = (media, ordered_keys)
            meta = {
                "article_id": article_id,
                "content_type": media,
                "content": content_text,
            }
            article_by_media_signature[sig] = meta
            per_article_qmeta[(article_id, media)] = q_infos

    # Load human long
    hp = pd.read_csv(data_root / "kg_points_long.csv")
    hp["media"] = hp["media"].astype(str).str.strip().str.casefold()
    hp["phase"] = hp["phase"].astype(str).str.strip().str.casefold()
    hp["annotator_id_int"] = hp["annotator_id"].apply(lambda x: int(str(x).strip()) if str(x).strip().isdigit() else None)
    hp["q_in_set"] = hp["q_in_set"].astype(int)
    hp["question_key"] = hp["question_text"].apply(qnorm)
    hp = hp[hp["annotator_id_int"].notna() & hp["phase"].isin({"pre", "post"})].copy()
    hp["annotator_id_int"] = hp["annotator_id_int"].astype(int)

    # Signature per (media, qs_id)
    qs_groups = hp.groupby(["media", "qs_id"], dropna=False)
    qs_signature_to_article: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}
    for (media, qs_id), df in qs_groups:
        qmap = df.drop_duplicates(subset=["q_in_set"]).sort_values("q_in_set")
        ordered_keys = tuple(qmap["question_key"].tolist())
        sig = (media, ordered_keys)
        if sig in article_by_media_signature:
            qs_signature_to_article[sig] = article_by_media_signature[sig]

    # Build records
    records: List[Dict] = []
    grp = hp.groupby(["media", "qs_id", "annotator_id_int"], dropna=False)

    for (media, qs_id, annot), df in grp:
        qmap = df.drop_duplicates(subset=["q_in_set"]).sort_values("q_in_set")
        ordered_keys = tuple(qmap["question_key"].tolist())
        sig = (media, ordered_keys)
        meta = qs_signature_to_article.get(sig)
        if meta is None:
            continue

        article_id = meta["article_id"]
        content = meta["content"]
        q_infos = per_article_qmeta[(article_id, media)]

        # We'll look at both pre and post rows per q_in_set, prefer the first non-null value
        for q_info in q_infos:
            q_idx = int(q_info["question_in_set"])
            subset = df[df["q_in_set"] == q_idx]
            pre_raw = subset.loc[subset["phase"] == "pre", "answer_option"].dropna().astype(str)
            post_raw = subset.loc[subset["phase"] == "post", "answer_option"].dropna().astype(str)

            pre_val = pre_raw.iloc[0] if len(pre_raw) else None
            post_val = post_raw.iloc[0] if len(post_raw) else None

            options_list = q_info["options"]
            pre_idx = parse_option_to_index(pre_val, options_list) if pre_val is not None else None
            post_idx = parse_option_to_index(post_val, options_list) if post_val is not None else None

            rec = {
                "article-id": article_id,
                "content-type": media,
                "content": content,
                "question_in_set": q_idx,
                "question-text": q_info["question_text"],
                "options": options_list,
                "correct_option": int(q_info["correct_option"]) if q_info["correct_option"] is not None else None,
                "correct_answer": q_info["correct_answer"],
                "annotator_id": int(annot),
                "human-answer-pre": int(pre_idx) if pre_idx is not None else None,
                "human-answer-post": int(post_idx) if post_idx is not None else None,
            }
            records.append(rec)

    records.sort(key=lambda r: (str(r["article-id"]), r["content-type"], int(r["question_in_set"]), int(r["annotator_id"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(records)} datapoints to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".", help="Directory containing kgain_dataset.json and kg_points_long.csv")
    ap.add_argument("--out", type=str, default="kgain_final_dataset.json", help="Output JSON path (default: ./kgain_final_dataset.json)")
    args = ap.parse_args()
    build_final_dataset(Path(args.data_root).resolve(), Path(args.out).resolve())

if __name__ == "__main__":
    main()
