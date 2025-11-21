#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build document-centric grouped annotations - FIXED OFFSET VERSION

Fixes:
- Corrects the ID matching offset: qs_id (2..31) -> Article Index (0..29).
- Ensures all 30 articles (90 documents) are correctly matched.
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
    "asbtract": "abstract",
    "tweet": "tweet",
    "tweets": "tweet",
}

def nrm_basic(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()

def is_dk_text(s: str) -> bool:
    t = nrm_basic(s).rstrip(".")
    return t in DK_CANONICAL_TEXTS

# Options & Answer Parsing
def options_dict_to_list_with_dk_last(opts: Dict[str, str], diag: Diag) -> Tuple[List[str], Dict[str, int]]:
    if not isinstance(opts, dict):
        return ([DK_LABEL], {})

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
        dk_text = dk_items[0][1]
        if dk_text not in ordered:
            ordered.append(dk_text)
        else:
            ordered = [o for o in ordered if nrm_basic(o) != nrm_basic(dk_text)]
            ordered.append(dk_text)
    else:
        ordered.append(DK_LABEL)
        diag.added_dk_to_questions += 1

    key2idx: Dict[str, int] = {}
    text2idx: Dict[str, int] = {nrm_basic(txt): i + 1 for i, txt in enumerate(ordered)}
    for k, v in non_dk_items_sorted + dk_items:
        key2idx[k] = text2idx[nrm_basic(v)]
    
    key2idx["DK"] = text2idx[nrm_basic(ordered[-1])]
    return ordered, key2idx

RE_PREFIX_LETTER = re.compile(r"^\s*([A-J])\s*[\)\].:\-–—]\s*(.*)$", re.IGNORECASE)
RE_PREFIX_NUM    = re.compile(r"^\s*(\d+)\s*[\)\].:\-–—]\s*(.*)$", re.IGNORECASE)
RE_TRAIL_LETTER  = re.compile(r"^(.*?)[\s]*[\(\[]\s*([A-J])\s*[\)\]]\s*$", re.IGNORECASE)
RE_TRAIL_NUM     = re.compile(r"^(.*?)[\s]*[\(\[]\s*(\d+)\s*[\)\]]\s*$", re.IGNORECASE)

def extract_letter_or_number_hint(s: str) -> Tuple[Optional[str], Optional[int], str]:
    if not isinstance(s, str):
        s = str(s)
    m = RE_PREFIX_LETTER.match(s)
    if m: return m.group(1).upper(), None, m.group(2)
    m = RE_PREFIX_NUM.match(s)
    if m: return None, int(m.group(1)), m.group(2)
    m = RE_TRAIL_LETTER.match(s)
    if m: return m.group(2).upper(), None, m.group(1)
    m = RE_TRAIL_NUM.match(s)
    if m: return None, int(m.group(2)), m.group(1)
    return None, None, s

def parse_option_to_index(raw_val, options: List[str], diag: Diag) -> Optional[int]:
    if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
        return None
    s = str(raw_val).strip()
    if s == "" or s == "--":
        return None

    if is_dk_text(s) or nrm_basic(s) in {"dk", "idk"}:
        return len(options)

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
        diag.remap_gt_len_to_dk += 1
        return len(options)

    if LETTER_RE.match(s):
        idx = ord(s.upper()) - ord("A") + 1
        if 1 <= idx <= len(options):
            return idx

    if re.fullmatch(r"-?\d+", s):
        num = int(s)
        if 1 <= num <= len(options):
            return num
        if 0 <= num < len(options):
            return num + 1
        diag.remap_gt_len_to_dk += 1
        return len(options)

    ns = nrm_basic(s)
    for i, opt in enumerate(options, start=1):
        if nrm_basic(opt) == ns:
            return i

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

# Main Builder
def build_docs(data_root: Path, out_path: Path) -> None:
    diag = Diag()

    # 1) Load JSON (Source of Truth)
    ds_path = data_root / "kgain_dataset.json"
    with open(ds_path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    articles = kg.get("articles", []) if isinstance(kg, dict) else kg
    print(f"[INFO] Loaded {len(articles)} articles from JSON.")

    # 2) Load CSV (Annotations)
    csv_path = data_root / "kg_points_long.csv"
    hp = pd.read_csv(csv_path)

    hp["media"] = hp["media"].astype(str).str.strip().str.casefold().map(MEDIA_NORMALIZE).fillna("unknown")
    hp = hp[hp["media"].isin({"news", "abstract", "tweet"})].copy()
    hp["phase"] = hp["phase"].astype(str).str.strip().str.casefold()
    hp["annotator_id_int"] = hp["annotator_id"].map(coerce_annotator_id)
    hp["q_in_set"] = hp["q_in_set"].astype(int)
    hp["qs_id"] = hp["qs_id"].astype(int)
    
    # Filter valid rows
    hp = hp[hp["annotator_id_int"].notna() & hp["phase"].isin({"pre", "post"})].copy()
    hp["annotator_id_int"] = hp["annotator_id_int"].astype(int)

    # 3) PRE-INITIALIZE Dictionary with ALL 90 Documents
    docs_map: Dict[Tuple[str, str], Dict] = {}
    
    for art in articles:
        article_id = art.get("article_id") or art.get("article-id")
        contents = art.get("contents", {})
        
        for media in ["news", "abstract", "tweet"]:
            doc_key = (article_id, media)
            content_text = contents.get(media)
            docs_map[doc_key] = {
                "article-id": article_id,
                "content-type": media,
                "content": content_text,
                "human_annotations": []
            }

    print(f"[INFO] Initialized {len(docs_map)} document buckets (should be 90).")

    # 4) Fill in Annotations from CSV
    grouped = hp.groupby(["media", "qs_id", "annotator_id_int"], dropna=False)
    print(f"[INFO] Processing {len(grouped)} annotator sessions...")

    for (media, qs_id, annot_id), df in grouped:
        # qs_id ranges from 2 to 31.
        # Python lists are 0-indexed.
        # So: Article Index = qs_id - 2.
        article_idx = qs_id - 2 
        
        if article_idx < 0 or article_idx >= len(articles):
            continue

        article_obj = articles[article_idx]
        article_id = article_obj.get("article_id") or article_obj.get("article-id")
        json_qas = article_obj.get("qas", [])
        doc_key = (article_id, media)

        if doc_key not in docs_map:
            continue 

        qa_annots = []
        answers_by_qnum = {}
        for _, row in df.iterrows():
            qnum = row["q_in_set"]
            ph = row["phase"]
            ans = row["answer_option"]
            if qnum not in answers_by_qnum:
                answers_by_qnum[qnum] = {"pre": None, "post": None}
            if pd.notna(ans):
                answers_by_qnum[qnum][ph] = str(ans)

        for q_idx, qa_json in enumerate(json_qas):
            q_num_1based = q_idx + 1
            
            q_text = qa_json.get("question")
            opts_dict = qa_json.get("options", {})
            options_list, origkey2newidx = options_dict_to_list_with_dk_last(opts_dict, diag)
            dk_idx = len(options_list)

            correct_key = qa_json.get("answer")
            correct_idx = None
            if isinstance(correct_key, str) and correct_key in origkey2newidx:
                correct_idx = int(origkey2newidx[correct_key])
            else:
                if isinstance(correct_key, str):
                    ci = parse_option_to_index(correct_key, options_list, diag)
                    correct_idx = int(ci) if ci is not None else None
            
            correct_ans_str = None
            if isinstance(correct_idx, int) and 1 <= correct_idx <= len(options_list):
                correct_ans_str = options_list[correct_idx - 1]

            user_ans = answers_by_qnum.get(q_num_1based, {})
            pre_raw = user_ans.get("pre")
            post_raw = user_ans.get("post")

            pre_idx = parse_option_to_index(pre_raw, options_list, diag)
            post_idx = parse_option_to_index(post_raw, options_list, diag)

            if pre_idx is None:
                pre_idx = dk_idx
                diag.imputed_null_to_dk += 1
            if post_idx is None:
                post_idx = dk_idx
                diag.imputed_null_to_dk += 1

            qa_annots.append({
                "question_in_set": q_num_1based,
                "question-text": q_text,
                "options": options_list,
                "correct_option": correct_idx,
                "correct_answer": correct_ans_str,
                "human-answer-pre": int(pre_idx),
                "human-answer-post": int(post_idx),
            })

        docs_map[doc_key]["human_annotations"].append({
            "annotator_id": int(annot_id),
            "qa_annotations": qa_annots
        })

    # 5) Final Output & Check for Empty Docs
    out_rows = sorted(docs_map.values(), key=lambda r: (str(r["article-id"]), r["content-type"]))

    empty_docs = [r for r in out_rows if len(r["human_annotations"]) == 0]
    if empty_docs:
        print(f"\n[WARNING] The following {len(empty_docs)} documents STILL have 0 annotators:")
        for d in empty_docs:
            print(f"  - ID: {d['article-id']} | Media: {d['content-type']}")
    else:
        print(f"\n[SUCCESS] All {len(out_rows)} documents have human annotations.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Wrote {len(out_rows)} document records to {out_path}")
    media_counts = Counter([r["content-type"] for r in out_rows])
    print("       Documents by media:", dict(media_counts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".", help="Directory containing kgain_dataset.json and kg_points_long.csv")
    ap.add_argument("--out", type=str, default="kgain_annotated_dataset.json", help="Output JSON path")
    args = ap.parse_args()
    build_docs(Path(args.data_root).resolve(), Path(args.out).resolve())

if __name__ == "__main__":
    main()