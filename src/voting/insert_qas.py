import json
import firebase_admin
from firebase_admin import credentials, firestore

# --- Initialize Firestore/Admin SDK ---
cred = credentials.Certificate("../key/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# more_questions_2_replace.json
# "../data/d11-20-replacement-questions.json" document 11-28
# "../data/more_questions_2_replace.json" // document 28
# "../data/even-more-questions2add.json" document 24
# --- Load questions ---
with open("../data/more_questions_2_replace.json", "r") as f:
    questions = json.load(f)

# --- Insert questions ---
for q in questions:
    article_id = q["article_id"]
    doc_ref = db.collection("papers2").document(article_id).collection("kgainQuestions").document()

    # Prepare Firestore document
    question_data = {
        "questionText": q["replacement_question"],
        "original_question_id": q.get("original_question_id", ""),
        "type": q["type"],
        "options": q["options"],
        "correctAnswer": q["correctAnswer"],
        "vote": 0,
        "voters": []
    }

    # Upload to Firestore
    doc_ref.set(question_data)
    print(f"Inserted question into article {article_id}: {q['replacement_question']}")

print("All questions inserted successfully!")
