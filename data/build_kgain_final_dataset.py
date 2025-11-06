#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build document-centric grouped annotations

Output: one item per (article, content-type):
{
  "article-id": "<string>",                  # EXACT from kgain_dataset.json
  "content-type": "news|abstract|tweet",
  "content": "<verbatim string>",
  "human_annotations": [
    {
      "annotator_id": <int >= 1>,
      "qa_annotations": [
        {
          "question_in_set": 1,
          "question-text": "<string>",
          "options": ["<A>", "<B>", "...", "I do not know the answer."],
          "correct_option": 2,            # 1-based index
          "correct_answer": "<string>",
          "human-answer-pre": 3,          # 1..len(options)
          "human-answer-post": 3          # 1..len(options)
        },
        ... (6 total)
      ]
    },
    ... (30 annotators)
  ]
}

Run:
  python kgain_final_dataset.py --data-root . --out kgain_docs_grouped.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Diagnostics container
class Diag:
    def __init__(self):
        self.remap_gt_len_to_dk = 0
        self.imputed_null_to_dk = 0
        self.added_dk_to_questions = 0

# Normalization helpers
LETTER_RE = re.compile(r"^[A-J]$", re.IGNORECASE)
LEADING_ENUM_RE = re.compile(
    r"""^\s*(?:                                   # START
        (?:q(?:uestion)?\s*\d+\s*[:.)-]) |       # Q1:  Question 2)  q10.
        (?:\(?\d+\)?\s*[:.)-]) |                 # 1.  (2)  3)  4:
        (?:[ivxlcdm]+\s*[:.)-]) |                # i. ii) v:
        (?:[a-j]\s*[:.)-])                       # a) b. c:
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)

# canonical DK variants
DK_CANONICAL_TEXTS = {
    "i do not know the answer",
    "i do not know the answer.",
    "i do not know",
    "i don't know",
    "i dont know",
    "dont know",
    "do not know",
}
DK_LABEL = "I do not know the answer."

MEDIA_NORMALIZE = {
    "news": "news",
    "abstract": "abstract",
    "asbtract": "abstract",  # typo tolerance
    "tweet": "tweet",
    "tweets": "tweet",
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

# Options & answers
def options_dict_to_list_with_dk_last(opts: Dict[str, str], diag: Diag) -> Tuple[List[str], Dict[str, int]]:
    """
    Return (ordered_options, originalKey->newIndex).
    - Sort A,B,C,... in key order
    - Put DK last if present
    - If DK not present, APPEND canonical DK as the last option
    """
    if not isinstance(opts, dict):
        raise ValueError("options must be a dict of {letter|DK: option_text}")

    non_dk_items, dk_items = [], []
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
    if dk_items:
        # keep only first DK text; ensure last
        dk_text = dk_items[0][1]
        if dk_text not in ordered:
            ordered.append(dk_text)
        else:
            # if DK text already somewhere in ordered, move it to the end
            ordered = [o for o in ordered if nrm_basic(o) != nrm_basic(dk_text)]
            ordered.append(dk_text)
    else:
        # DK missing — append canonical DK label LAST
        ordered.append(DK_LABEL)
        diag.added_dk_to_questions += 1

    # Build key->idx using the final ordered list
    key2idx: Dict[str, int] = {}
    text2idx: Dict[str, int] = {nrm_basic(txt): i + 1 for i, txt in enumerate(ordered)}
    for k, v in non_dk_items_sorted + dk_items:
        key2idx[k] = text2idx[nrm_basic(v)]
    # Also map "DK" by text after enforcing presence
    key2idx["DK"] = text2idx[nrm_basic(ordered[-1])]  # DK is last

    return ordered, key2idx

RE_PREFIX_LETTER = re.compile(r"^\s*([A-J])\s*[\)\].:\-–—]\s*(.*)$", re.IGNORECASE)
RE_PREFIX_NUM    = re.compile(r"^\s*(\d+)\s*[\)\].:\-–—]\s*(.*)$", re.IGNORECASE)
RE_TRAIL_LETTER  = re.compile(r"^(.*?)[\s]*[\(\[]\s*([A-J])\s*[\)\]]\s*$", re.IGNORECASE)
RE_TRAIL_NUM     = re.compile(r"^(.*?)[\s]*[\(\[]\s*(\d+)\s*[\)\]]\s*$", re.IGNORECASE)

def extract_letter_or_number_hint(s: str) -> Tuple[Optional[str], Optional[int], str]:
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

def parse_option_to_index(raw_val, options: List[str], diag: Diag) -> Optional[int]:
    """
    Return 1-based index into options.
    - DK shortcuts map to last
    - Out-of-range numeric (> len) maps to DK (last)
    """
    if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
        return None
    s = str(raw_val).strip()
    if s == "" or s == "--":
        return None

    # DK shortcuts
    if is_dk_text(s) or nrm_basic(s) in {"dk", "idk"}:
        return len(options)

    # letter/number hints
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
        # Out-of-range number → DK (last)
        diag.remap_gt_len_to_dk += 1
        return len(options)

    # bare letter
    if LETTER_RE.match(s):
        idx = ord(s.upper()) - ord("A") + 1
        if 1 <= idx <= len(options):
            return idx

    # bare number
    if re.fullmatch(r"-?\d+", s):
        num = int(s)
        if 1 <= num <= len(options):
            return num
        if 0 <= num < len(options):
            return num + 1
        diag.remap_gt_len_to_dk += 1
        return len(options)

    # exact text
    ns = nrm_basic(s)
    for i, opt in enumerate(options, start=1):
        if nrm_basic(opt) == ns:
            return i

    # residual from hint
    rs = nrm_basic(residual)
    if rs and rs != ns:
        for i, opt in enumerate(options, start=1):
            if nrm_basic(opt) == rs:
                return i

    return None

def coerce_annotator_id(x) -> Optional[int]:
    try:
        i = int(str(x).strip())
        return i if i >= 1 else None
    except Exception:
        return None

# Mapping helpers
def is_subsequence(subseq: Tuple[str, ...], full: Tuple[str, ...]) -> bool:
    """True if `subseq` appears in order (not necessarily contiguous) inside `full`."""
    it = iter(full)
    return all(any(tok == f for f in it) for tok in subseq)

# Build document-centric grouped data
def build_docs(data_root: Path, out_path: Path) -> None:
    diag = Diag()

    # 1) Load kgain_dataset.json
    ds_path = data_root / "kgain_dataset.json"
    with open(ds_path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    articles = kg.get("articles", []) if isinstance(kg, dict) else kg

    # 2) Build article signatures and per-question meta (per media)
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
            options_list, origkey2newidx = options_dict_to_list_with_dk_last(opts_dict, diag)

            # Correct answer key (by option key if present, otherwise parse against list)
            correct_key = qa.get("answer")
            correct_idx = None
            if isinstance(correct_key, str) and correct_key in origkey2newidx:
                correct_idx = int(origkey2newidx[correct_key])
            else:
                if isinstance(correct_key, str):
                    ci = parse_option_to_index(correct_key, options_list, diag)
                    correct_idx = int(ci) if ci is not None else None

            q_infos.append({
                "question_in_set": q_pos,
                "question_text": q_text,
                "question_key": qnorm(q_text),
                "options": options_list,   # DK guaranteed present & last
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

    # 3) Load human answers
    hp = pd.read_csv(data_root / "kg_points_long.csv")

    # normalize media (tolerate typos), phase, annotators
    hp["media"] = hp["media"].astype(str).str.strip().str.casefold().map(MEDIA_NORMALIZE).fillna("unknown")
    hp = hp[hp["media"].isin({"news", "abstract", "tweet"})].copy()
    hp["phase"] = hp["phase"].astype(str).str.strip().str.casefold()
    hp["annotator_id_int"] = hp["annotator_id"].map(coerce_annotator_id)
    hp["q_in_set"] = hp["q_in_set"].astype(int)
    hp["qs_id"] = hp["qs_id"].astype(int)
    hp["question_key"] = hp["question_text"].apply(qnorm)
    hp = hp[hp["annotator_id_int"].notna() & hp["phase"].isin({"pre", "post"})].copy()
    hp["annotator_id_int"] = hp["annotator_id_int"].astype(int)

    # 4) Build signature mapping per (media, qs_id) allowing subsequence
    sig_map: Dict[Tuple[str, int], Dict] = {}
    articles_by_media: Dict[str, List[Tuple[Tuple[str, ...], Dict]]] = {}
    for (m, keys), meta in article_by_media_signature.items():
        articles_by_media.setdefault(m, []).append((keys, meta))

    for (media, qs_id), df in hp.groupby(["media", "qs_id"], dropna=False):
        # observed ordered question keys (by q_in_set order, dedup by text)
        qmap = df.drop_duplicates(subset=["question_key"]).sort_values("q_in_set")
        observed = tuple(qmap["question_key"].tolist())
        if not observed:
            continue
        # exact match first
        meta = article_by_media_signature.get((media, observed))
        if meta is None:
            # subsequence match (strict order, gaps allowed)
            candidates = []
            for full_keys, m in articles_by_media.get(media, []):
                if is_subsequence(observed, full_keys):
                    candidates.append((len(observed), m))
            meta = candidates[0][1] if candidates else None
        if meta is not None:
            sig_map[(media, qs_id)] = meta

    # 5) Build doc-centric structure: (article_id, media) -> list of annotators and their qa_annotations
    docs: Dict[Tuple[str, str], Dict] = {}
    total_pre = total_post = 0

    # group by (media, qs_id, annotator) to build their 6-Q annotation
    for (media, qs_id, annot), df in hp.groupby(["media", "qs_id", "annotator_id_int"], dropna=False):
        meta = sig_map.get((media, qs_id))
        if meta is None:
            continue  # unmatched set

        article_id = meta["article_id"]
        content = meta["content"]
        q_infos = per_article_qmeta[(article_id, media)]

        # ensure doc node exists
        doc_key = (article_id, media)
        if doc_key not in docs:
            docs[doc_key] = {
                "article-id": article_id,
                "content-type": media,
                "content": content,
                "human_annotations": []
            }

        # map answers by normalized question text (NOT by q_in_set)
        per_qkey: Dict[str, Dict[str, Optional[str]]] = {}
        for qk, sdf in df.groupby("question_key"):
            pre_raw = sdf.loc[sdf["phase"] == "pre", "answer_option"].dropna().astype(str)
            post_raw = sdf.loc[sdf["phase"] == "post", "answer_option"].dropna().astype(str)
            per_qkey[qk] = {
                "pre": pre_raw.iloc[0] if len(pre_raw) else None,
                "post": post_raw.iloc[0] if len(post_raw) else None,
            }

        qa_annots = []
        for q_info in q_infos:
            qidx = int(q_info["question_in_set"])
            qk = q_info["question_key"]
            options_list = q_info["options"]
            dk_idx = len(options_list)  # DK is last (forced)

            pre_val = per_qkey.get(qk, {}).get("pre")
            post_val = per_qkey.get(qk, {}).get("post")

            pre_idx = parse_option_to_index(pre_val, options_list, diag)
            post_idx = parse_option_to_index(post_val, options_list, diag)

            # Impute any missing/blank to DK
            if pre_idx is None:
                pre_idx = dk_idx
                diag.imputed_null_to_dk += 1
            if post_idx is None:
                post_idx = dk_idx
                diag.imputed_null_to_dk += 1

            total_pre += 1
            total_post += 1

            qa_annots.append({
                "question_in_set": qidx,
                "question-text": q_info["question_text"],
                "options": options_list,
                "correct_option": int(q_info["correct_option"]) if q_info["correct_option"] is not None else None,
                "correct_answer": q_info["correct_answer"],
                "human-answer-pre": int(pre_idx),
                "human-answer-post": int(post_idx),
            })

        # append annotator entry to this document
        docs[doc_key]["human_annotations"].append({
            "annotator_id": int(annot),
            "qa_annotations": qa_annots
        })

    # 6) Emit as a list sorted stably (by article-id, content-type)
    out_rows = sorted(docs.values(), key=lambda r: (str(r["article-id"]), r["content-type"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    # 7) Diagnostics
    print(f"[DONE] Wrote {len(out_rows)} document records to {out_path}")
    print(f"       Added DK to questions missing it: {diag.added_dk_to_questions}")
    print(f"       Remapped out-of-range numerics to DK: {diag.remap_gt_len_to_dk}")
    print(f"       Imputed NULL/blank to DK: {diag.imputed_null_to_dk}")
    media_counts = Counter([r["content-type"] for r in out_rows])
    print("       Documents by media:", dict(media_counts))
    annot_dist = Counter(len(r["human_annotations"]) for r in out_rows)
    print("       Annotator counts per document (histogram):", dict(sorted(annot_dist.items())))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".", help="Directory containing kgain_dataset.json and kg_points_long.csv")
    ap.add_argument("--out", type=str, default="kgain_annotated_dataset.json", help="Output JSON path (default: ./kgain_docs_grouped.json)")
    args = ap.parse_args()
    build_docs(Path(args.data_root).resolve(), Path(args.out).resolve())

if __name__ == "__main__":
    main()
