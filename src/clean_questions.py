#!/usr/bin/env python3
import argparse, json, re
from collections import defaultdict

# -----------------------------
# PATTERNS TO REMOVE (robust)
# -----------------------------
REMOVE_PHRASES = [
    r"according to the findings",
    r"according to findings",
    r"according to recent research",
    r"according to recent studies",
    r"according to the study",
    r"according to the study described",
    r"in the study mentioned",
    r"in the study described",
    r"based on the findings",
    r"based on the study",
    r"as described in the abstract",
    r"as mentioned in the study",
]

REMOVE_RE = re.compile("|".join(REMOVE_PHRASES), flags=re.IGNORECASE)

WH_WORDS = ("what", "which", "how", "why", "when", "where", "in which", "to what")


# -----------------------------
# CLEAN CORE FUNCTION
# -----------------------------
def clean_question(q: str):
    original = q

    if not q:
        return q, False, "empty"

    q = q.strip()

    # fix weird spacing artifacts early
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"\s+,", ",", q)
    q = re.sub(r",\s+", ", ", q)

    # remove boilerplate phrases
    q2 = REMOVE_RE.sub("", q)
    q2 = re.sub(r"\s+", " ", q2).strip()

    changed = (q2 != q)
    q = q2

    # fix leading punctuation artifacts (", which ...")
    q = re.sub(r"^,\s*", "", q)
    q = re.sub(r"^\s*[,;:]\s*", "", q)

    # detect if True/False style
    is_tf = any(x in q for x in ["True", "False"]) and len(q.split()) < 25

    # -----------------------------
    # punctuation logic
    # -----------------------------
    q = q.strip()

    # IMPORTANT: preserve existing terminal punctuation
    if q.endswith((".", "?", "!", ":")):
        return q, changed, "kept_punct"

    # True/False → statement
    if is_tf:
        q = q.rstrip("?:! ") + "."
        return q, True, "tf_fix"

    # WH-question → question mark
    lower = q.lower()

    if any(lower.startswith(w) for w in WH_WORDS):
        q = q.rstrip("?:! ") + "?"
        return q, True, "wh_question_fix"

    # default fallback → question mark if looks like question
    if "?" not in q and any(w in lower for w in WH_WORDS):
        q = q + "?"

    return q, changed, "generic_fix"


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log", default="clean_log.jsonl")
    args = ap.parse_args()

    total_questions = 0
    modified = 0
    flagged = 0

    reason_counts = defaultdict(int)

    with open(args.input) as f, \
         open(args.output, "w") as out, \
         open(args.log, "w") as log:

        for line in f:
            ex = json.loads(line)
            qas = ex.get("qa_annotations", [])

            for qa in qas:
                total_questions += 1

                old_q = qa.get("question-text") or qa.get("question_text") or ""

                new_q, changed, reason = clean_question(old_q)

                qa["question-text"] = new_q

                if changed:
                    modified += 1
                    reason_counts[reason] += 1

                    log.write(json.dumps({
                        "type": "changed",
                        "old": old_q,
                        "new": new_q,
                        "reason": reason
                    }) + "\n")
                else:
                    log.write(json.dumps({
                        "type": "cleaned_only",
                        "old": old_q,
                        "new": new_q
                    }) + "\n")

                # flag suspicious cases (empty / broken)
                if len(new_q.strip()) < 5:
                    flagged += 1

            out.write(json.dumps(ex) + "\n")

    print("Done.")
    print(f"Total questions: {total_questions}")
    print(f"Modified: {modified}")
    print(f"Flagged candidates: {flagged}")
    print(f"Output: {args.output}")
    print(f"Log: {args.log}")
    print("\nTop reasons:")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()