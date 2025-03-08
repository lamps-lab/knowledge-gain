import firebase_admin
from firebase_admin import credentials, firestore
import json

from bs4 import BeautifulSoup

cred = credentials.Certificate("key/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

articles = []
papers_ref = db.collection('papers')
papers = list(papers_ref.limit(10).stream())

def extract_text(html):
    # Extract plain text from an HTML string
    #   preserving newlines for <br> and <p> tags.
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Use "\n" as a separator to preserve line breaks and paragraphs.
        return soup.get_text(separator="\n", strip=True)
    return "N/A"

def map_question_type(q_type):
    # map Firestore type to qa_type string.
    mapping = {'a': 'TF', 'b': 'MC_easy', 'c': 'MC_hard'}
    return mapping.get(q_type, q_type)

article_index = 1
for paper in papers:
    paper_data = paper.to_dict()
    article_id = paper.id  # Use the same article id as in the DB

    contents = {
        "abstract": extract_text(paper_data.get("abstracthtml", "")),
        "news": extract_text(paper_data.get("newshtml", "")),
        "tweet": extract_text(paper_data.get("tweethtml", ""))
    }
    
    # Group QA documents by question type (a, b, c)
    questions_ref = papers_ref.document(paper.id).collection('kgainQuestions')
    groups = {'a': [], 'b': [], 'c': []}
    for qdoc in questions_ref.stream():
        qdata = qdoc.to_dict()
        qtype = qdata.get("type", "")
        if qtype in groups:
            groups[qtype].append(qdata)
    
    # For each type, sort by vote (desc) and select top 2
    qas = []
    for qtype, qlist in groups.items():
        qlist.sort(key=lambda x: x.get("vote", 0), reverse=True)
        for qdata in qlist[:2]:
            qa_entry = {
                "question": qdata.get("questionText", ""),
                "answer": qdata.get("correctAnswer", ""),
                "qa_type": map_question_type(qdata.get("type", "")),
                "source": qdata.get("source", ""),
                "evidence": qdata.get("evidence", ""),
                "vote": qdata.get("vote", 0)
            }
            qas.append(qa_entry)
    
    article = {
        "article_id": article_id,
        "contents": contents,
        "qas": qas
    }
    articles.append(article)

with open("kgain_dataset.json", "w") as f:
    json.dump({"articles": articles}, f, indent=2)

print("Generated dataset with", len(articles), "articles.")