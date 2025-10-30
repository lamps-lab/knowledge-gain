import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore ---
cred = credentials.Certificate("../key/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

TARGET_EMAIL = "mjiang2@nd.edu"

# Global vote counters (across all questions) ---
global_vote_counts = {"a": 0, "b": 0, "c": 0}

# Per-sample result containers for each type ---
zero_votes      = {t: set() for t in ("a", "b", "c")}
less_than_two   = {t: set() for t in ("a", "b", "c")}

# Scan each paper/sample ---
for paper in db.collection('papers2').stream():
    sample_id = paper.id

    # Count of votes by this user within this sample, per type
    votes_by_type = {"a": 0, "b": 0, "c": 0}

    # Iterate all questions in this sample
    for qdoc in paper.reference.collection('kgainQuestions').stream():
        q = qdoc.to_dict()
        t = q.get("type")
        if t not in votes_by_type:
            continue

        # Did mjiang2@nd.edu vote here?
        if any(v.get("email") == TARGET_EMAIL for v in q.get("voters", [])):
            global_vote_counts[t] += 1      # global tally
            votes_by_type[t]  += 1          # per-sample tally

    # After scanning questions in this sample:
    for t in ("a", "b", "c"):
        if votes_by_type[t] == 0:
            zero_votes[t].add(sample_id)
        if votes_by_type[t] < 2:
            less_than_two[t].add(sample_id)

# Print results ---

print(f"=== Global vote counts for {TARGET_EMAIL} ===")
for qt in ("a", "b", "c"):
    print(f"Type '{qt}': {global_vote_counts[qt]} total votes")

for qt in ("a", "b", "c"):
    print(f"\n=== Samples with NO votes of type '{qt}' ===")
    for sid in sorted(zero_votes[qt]):
        print(f"- {sid}")

    print(f"\n=== Samples with LESS THAN TWO votes of type '{qt}' ===")
    for sid in sorted(less_than_two[qt]):
        print(f"- {sid}")
