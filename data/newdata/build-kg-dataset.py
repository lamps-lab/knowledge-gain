"""
Build KG_i,j,k datapoints from Qualtrics-style CSVs for three media types.
Now supports answer keys in CSV or JSON. If none is present, generates templates.

Input files (expected in the working directory):
  - a1.csv, a2.csv, a3.csv  -> media 'news'
  - b1.csv, b2.csv, b3.csv  -> media 'abstract'
  - c1.csv, c2.csv, c3.csv  -> media 'tweet'

Each CSV contains ~120 question columns organized as repeated blocks:
  [6 pre questions] -> [Timing x 4 columns] -> [6 post questions] -> [Timing] -> ...
for 10 question-sets per file (10 * 12 = 120 question columns).
The *second header row* (human-readable text) repeats for pre & post, which we use
to pair (pre, post) for the same question within a question-set.

Outputs:
  - kg_points.csv        : one row per (qs_id, media, annotator), i.e., KG_i,j,k
  - kg_points_long.csv   : per-question long format, includes pre/post answers
  - answer_key_template.json / answer_key_template.csv (only created if no key is found)

Answer key formats this script accepts:
  A) CSV:  answer_key.csv  with columns: ['qs_id','q_in_set','correct_option']
  B) JSON: answer_key.json with items shaped like:
     {
       "question-id": <int>,            # = 1000*qs_id + q_in_set
       "qs_id": <int>,                  # 1..30
       "q_in_set": <int>,               # 1..6
       "question-text": "<string>",
       "answer": "<string or number>"   # correct option; must match your data values
     }
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


# -----------------------------
# Config
# -----------------------------

MEDIA_PREFIX = {
    "a": "news",
    "b": "abstract",
    "c": "tweet",
}

# How many Qs inside a single question set (we see 6 in your header sample)
QS_SIZE = 6

# Timing cols between pre and post exist in raw exports but are ignored (we key off Q\d+ only)
TIMING_COLS = 4

# How many question sets per CSV (10 sets x 12 Q-columns total per set)
SETS_PER_FILE = 10

# Column label that carries the human-entered annotator ID in the 2nd header row (per your sample)
ANNOTATOR_TEXT_KEY = "Please enter your ID number"

# Fallback: if above isn't found, use these column candidates by first-row header names
ANNOTATOR_COL_CANDIDATES = ["QID291", "QID291_TEXT"]


# -----------------------------
# Helpers
# -----------------------------

def load_multiheader_csv(path: Path) -> pd.DataFrame:
    """Load Qualtrics-style export with 3 header rows into a 3-level MultiIndex columns DataFrame."""
    return pd.read_csv(path, header=[0, 1, 2], low_memory=False)


def find_annotator_column(df: pd.DataFrame) -> Tuple[int, Tuple[str, str, str]]:
    """Locate the annotator ID column via the 2nd-level header text; fallback to known variable names."""
    for col in df.columns:
        if isinstance(col, tuple) and len(col) >= 2:
            text = col[1]
            if isinstance(text, str) and ANNOTATOR_TEXT_KEY.lower() in text.lower():
                return 1, col
    for col in df.columns:
        if isinstance(col, tuple) and len(col) >= 1:
            var = col[0]
            if var in ANNOTATOR_COL_CANDIDATES:
                return 0, col
    raise ValueError("Could not find an annotator ID column. "
                     f"Looked for text '{ANNOTATOR_TEXT_KEY}' or {ANNOTATOR_COL_CANDIDATES}.")


def extract_question_cols_in_order(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Collect Q columns in original order by first-row labels like:
      "Q1", "Q1.", "Q1 True/False", "Q23. Easy MC", etc.
    Exclude technical "QID..." fields.
    """
    qcols: List[Tuple[str, str, str]] = []
    for col in df.columns:
        var = col[0]  # level-0 label
        if not isinstance(var, str):
            continue
        v = var.strip()
        if v.upper().startswith("QID"):
            continue
        # Accept any that start with Q<digits> (word boundary)
        if re.match(r"^Q\d+\b", v):
            qcols.append(col)
    return qcols


def group_into_question_sets(qcols: List[Tuple[str, str, str]]) -> List[Tuple[List, List]]:
    """
    Given ordered Q columns, pair them by set:
      [6 pre], [6 post].
    (Timing columns aren't in qcols because we only captured Q* columns.)
    """
    sets = []
    i = 0
    while i < len(qcols):
        pre = qcols[i:i+QS_SIZE]
        i += QS_SIZE
        post = qcols[i:i+QS_SIZE]
        i += QS_SIZE
        if len(pre) != QS_SIZE or len(post) != QS_SIZE:
            break
        sets.append((pre, post))
    return sets


def melt_answers_for_one_set(
    row: pd.Series,
    pre_cols: List[Tuple[str, str, str]],
    post_cols: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    """Return 2*QS_SIZE rows: (phase, q_in_set, question_text, answer_option)."""
    entries = []
    for phase, cols in [("pre", pre_cols), ("post", post_cols)]:
        for q_idx, col in enumerate(cols, start=1):
            qtext = col[1] if isinstance(col, tuple) and len(col) > 1 else col[0]
            ans = row[col]
            entries.append({
                "phase": phase,
                "q_in_set": q_idx,
                "question_text": qtext,
                "answer_option": ans
            })
    return pd.DataFrame(entries)


def parse_media_csv(path: Path, qs_offset: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_multiheader_csv(path)
    _, annot_col = find_annotator_column(df)
    annotator = df[annot_col].astype(str).str.strip()

    qcols = extract_question_cols_in_order(df)

    # --- Diagnostics ---
    print(f"[DEBUG] {path.name}: found {len(qcols)} Q columns")
    if len(qcols) <= 0:
        sample_level0 = list(dict.fromkeys(df.columns.get_level_values(0)))[:30]
        print("[DEBUG] level-0 header samples:", sample_level0)

    # Keep rows that have *any* answer among Q-columns
    data_mask = df[qcols].notna().any(axis=1) if qcols else pd.Series(False, index=df.index)

    # Be tolerant about the Finished flag shape in MultiIndex
    finished_mask = True
    try:
        finished_cols = [c for c in df.columns
                         if isinstance(c, tuple) and len(c) >= 1 and str(c[0]).strip() == "Finished"]
        if finished_cols:
            fm = None
            for c in finished_cols:
                colmask = (df[c] == 1)
                fm = colmask if fm is None else (fm | colmask)
            finished_mask = fm
    except Exception:
        finished_mask = True

    # Apply filtering
    df = df[data_mask & finished_mask].copy()
    annotator = annotator.loc[df.index]

    sets = group_into_question_sets(qcols)

    long_entries, set_entries = [], []
    fname = path.stem
    media_prefix = fname[0].lower()
    media = MEDIA_PREFIX.get(media_prefix, "unknown")
    batch_match = re.findall(r"\d+", fname)
    batch = int(batch_match[0]) if batch_match else None

    for ridx, row in df.iterrows():
        annot_id = str(annotator.loc[ridx]) if pd.notna(annotator.loc[ridx]) else f"{fname}_row{ridx}"
        for local_set_idx, (pre_cols, post_cols) in enumerate(sets, start=1):
            qs_id = qs_offset + local_set_idx

            qdf = melt_answers_for_one_set(row, pre_cols, post_cols)
            qdf.insert(0, "annotator_id", annot_id)
            qdf.insert(1, "media", media)
            qdf.insert(2, "file", fname)
            qdf.insert(3, "batch", batch)
            qdf.insert(4, "qs_id", qs_id)
            long_entries.append(qdf)

            pre_answers = qdf.loc[qdf["phase"]=="pre", "answer_option"].tolist()
            post_answers = qdf.loc[qdf["phase"]=="post", "answer_option"].tolist()

            set_entries.append({
                "annotator_id": annot_id,
                "media": media,
                "file": fname,
                "batch": batch,
                "qs_id": qs_id,
                "n_questions": QS_SIZE,
                "pre_answers": pre_answers,
                "post_answers": post_answers,
                "pre_correct": None,
                "post_correct": None,
                "knowledge_gain": None,
            })

    set_level_df = pd.DataFrame(set_entries)
    question_level_df = pd.concat(long_entries, ignore_index=True) if long_entries else pd.DataFrame(
        columns=["annotator_id","media","file","batch","qs_id","phase","q_in_set","question_text","answer_option"]
    )
    return set_level_df, question_level_df


# -----------------------------
# Answer-key handling
# -----------------------------

def normalized_match(a: pd.Series, b: pd.Series) -> pd.Series:
    """Case/space-insensitive equality on stringified values."""
    return (
        a.astype(str).str.strip().str.casefold()
        == b.astype(str).str.strip().str.casefold()
    )


def load_answer_key(in_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load an answer key from CSV or JSON.
    Returns a DataFrame with columns ['qs_id','q_in_set','correct_option'] or None.
    """
    csv_path = in_dir / "answer_key.csv"
    json_path = in_dir / "answer_key.json"

    if csv_path.exists():
        key_df = pd.read_csv(csv_path)
        if not {"qs_id","q_in_set","correct_option"}.issubset(key_df.columns):
            raise ValueError("answer_key.csv must have columns: qs_id,q_in_set,correct_option")
        key_df = key_df.loc[:, ["qs_id","q_in_set","correct_option"]].copy()
        key_df["q_in_set"] = key_df["q_in_set"].astype(int)
        return key_df

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        rows = []
        for it in items:
            qs_id = it.get("qs_id")
            q_in_set = it.get("q_in_set")
            correct_option = it.get("answer")
            if qs_id is None or q_in_set is None:
                qid = it.get("question-id")
                if isinstance(qid, int):
                    qs_id = qid // 1000
                    q_in_set = qid % 1000
            rows.append({"qs_id": qs_id, "q_in_set": q_in_set, "correct_option": correct_option})
        key_df = pd.DataFrame(rows)
        if key_df[["qs_id","q_in_set"]].isna().any().any():
            raise ValueError("answer_key.json items must include qs_id and q_in_set (or a parseable question-id).")
        key_df["q_in_set"] = key_df["q_in_set"].astype(int)
        return key_df

    return None


def export_answer_key_templates(out_dir: Path, q_df: pd.DataFrame) -> None:
    """
    When no key is found, emit:
      - answer_key_template.json (with fields requested)
      - answer_key_template.csv  (qs_id,q_in_set,question_text,correct_option)
    We use POST-phase question_text (identical to PRE for a set) to avoid duplicates.
    """
    base = (
        q_df[q_df["phase"]=="post"]
        .sort_values(["qs_id","q_in_set"])
        .drop_duplicates(["qs_id","q_in_set"])
        .loc[:, ["qs_id","q_in_set","question_text"]]
        .reset_index(drop=True)
    )
    # JSON template
    json_items = []
    for _, r in base.iterrows():
        qid = int(1000 * int(r["qs_id"]) + int(r["q_in_set"]))
        json_items.append({
            "question-id": qid,
            "qs_id": int(r["qs_id"]),
            "q_in_set": int(r["q_in_set"]),
            "question-text": str(r["question_text"]),
            "answer": ""   # <- fill with the correct option exactly as it appears in your data
        })
    with open(out_dir / "answer_key_template.json", "w", encoding="utf-8") as f:
        json.dump(json_items, f, ensure_ascii=False, indent=2)

    # CSV template
    csv_df = base.rename(columns={"question_text":"question_text"}).copy()
    csv_df["correct_option"] = ""
    csv_df.to_csv(out_dir / "answer_key_template.csv", index=False)

    print("[INFO] No answer key found. Created 'answer_key_template.json' and 'answer_key_template.csv'.")
    print("       Fill either one (JSON 'answer' or CSV 'correct_option') and rerun.")


def compute_correctness_from_key(
    set_df: pd.DataFrame, q_df: pd.DataFrame, key_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge an answer key to compute correctness per question and aggregate to knowledge gain.
    Matching is case/space-insensitive on stringified values.
    """
    key_df = key_df.copy()
    key_df["q_in_set"] = key_df["q_in_set"].astype(int)

    q = q_df.merge(key_df, on=["qs_id","q_in_set"], how="left")
    q["is_correct"] = normalized_match(q["answer_option"], q["correct_option"])

    agg = (
        q.groupby(["annotator_id","media","qs_id","phase"])["is_correct"]
        .sum()
        .unstack("phase")
        .fillna(0)
        .rename(columns={"pre":"pre_correct", "post":"post_correct"})
        .reset_index()
    )

    set_df2 = set_df.drop(columns=["pre_correct","post_correct","knowledge_gain"], errors="ignore")
    set_df2 = set_df2.merge(agg, on=["annotator_id","media","qs_id"], how="left")
    set_df2["pre_correct"] = set_df2["pre_correct"].fillna(0).astype(int)
    set_df2["post_correct"] = set_df2["post_correct"].fillna(0).astype(int)
    set_df2["knowledge_gain"] = set_df2["post_correct"] - set_df2["pre_correct"]
    return set_df2, q


# -----------------------------
# Build
# -----------------------------

def build_all(input_dir: str = ".", out_dir: str = ".") -> None:
    in_dir = Path(input_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = [
        "a1.csv","a2.csv","a3.csv",
        "b1.csv","b2.csv","b3.csv",
        "c1.csv","c2.csv","c3.csv",
    ]
    # Map batch digit to global qs_id offsets so *1 → 1..10, *2 → 11..20, *3 → 21..30
    offsets = {"1": 1, "2": 11, "3": 21}

    all_set, all_q = [], []

    for f in files:
        p = in_dir / f
        if not p.exists():
            print(f"[WARN] Missing file: {p}")
            continue
        batch_digit = re.findall(r"\d+", f)
        qs_offset = offsets.get(batch_digit[0], 1) if batch_digit else 1
        sdf, qdf = parse_media_csv(p, qs_offset=qs_offset)
        all_set.append(sdf.assign(source_file=f))
        all_q.append(qdf.assign(source_file=f))

    set_df = pd.concat(all_set, ignore_index=True) if all_set else pd.DataFrame()
    q_df = pd.concat(all_q, ignore_index=True) if all_q else pd.DataFrame()

    # If we have no questions at all, write empty and stop early (helps debugging)
    if q_df.empty or set_df.empty:
        print("[INFO] No question rows parsed; please verify input files and regex for Q-columns.")
        set_df.to_csv(out / "kg_points.csv", index=False)
        q_df.to_csv(out / "kg_points_long.csv", index=False)
        return

    # Load key (CSV or JSON); if none found, export templates using discovered questions
    key_df = load_answer_key(in_dir)
    if key_df is None:
        export_answer_key_templates(out, q_df)
        print("[INFO] Skipping correctness & KG computation until an answer key is provided.")
    else:
        set_df, q_df = compute_correctness_from_key(set_df, q_df, key_df)

    # Write outputs
    set_df.to_csv(out / "kg_points.csv", index=False)
    q_df.to_csv(out / "kg_points_long.csv", index=False)

    print(f"Wrote {len(set_df)} KG_i,j,k rows to {out/'kg_points.csv'}")
    print(f"Wrote {len(q_df)} per-question rows to {out/'kg_points_long.csv'}")


if __name__ == "__main__":
    build_all(".", ".")
