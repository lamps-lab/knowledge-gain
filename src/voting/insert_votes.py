import json
import firebase_admin
from firebase_admin import credentials, firestore, auth

# --- Initialize Firestore/Admin SDK ---
cred = credentials.Certificate("../key/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Load the votes you want to cast ---
with open("votes2cast.json", "r") as f:
    votes_to_cast = json.load(f)

TARGET_EMAIL = "mjiang2@nd.edu"
WEIGHT = 1  # or whatever weight you assign per vote

# --- Look up your Auth UID once ---
user = auth.get_user_by_email(TARGET_EMAIL)
TARGET_UID = user.uid

# --- Create a Transaction object ---
transaction = db.transaction()

@firestore.transactional
def cast_vote_transaction(txn, q_ref):
    """Transaction handler that increments vote & adds you as a voter."""
    snapshot = q_ref.get(transaction=txn)
    if not snapshot.exists:
        raise RuntimeError(f"Question doc {q_ref.path} no longer exists")

    data = snapshot.to_dict()
    # Skip if already voted
    if any(v.get("email") == TARGET_EMAIL for v in data.get("voters", [])):
        return

    # Otherwise increment vote count and append to voters
    txn.update(q_ref, {
        "vote": firestore.Increment(1),
        "voters": firestore.ArrayUnion([{
            "email": TARGET_EMAIL,
            "uid": TARGET_UID,
            "weight": WEIGHT
        }])
    })

# --- Iterate and cast all votes ---
for sample_id, question_texts in votes_to_cast.items():
    coll_ref = db.collection("papers2").document(sample_id).collection("kgainQuestions")
    for qdoc in coll_ref.stream():
        data = qdoc.to_dict()
        if data.get("questionText") in question_texts:
            print(f"Casting vote on sample={sample_id!r}, questionText={data['questionText']!r}")
            # Note: pass the same Transaction object each time
            cast_vote_transaction(transaction, qdoc.reference)

print("All done!")
