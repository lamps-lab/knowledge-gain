import firebase_admin
from firebase_admin import credentials, firestore
import json
from bs4 import BeautifulSoup

cred = credentials.Certificate("../key/accountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def extract_text(html):
    if html:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    return "N/A"

def map_question_type(q_type):
    mapping = {'a': 'TF', 'b': 'MC_easy', 'c': 'MC_hard'}
    return mapping.get(q_type, q_type)

articles = []
papers_ref2 = db.collection('papers2')
papers_ref = db.collection('papers')

# pull 20 from papers2 and 10 from papers
papers0 = list(papers_ref2.limit(20).stream())
papers1 = list(papers_ref.limit(10).stream())
papers = papers0 + papers1

for paper in papers:
    paper_data = paper.to_dict()
    article_id = paper.id

    contents = {
        "abstract": extract_text(paper_data.get("abstracthtml", "")),
        "news": extract_text(paper_data.get("newshtml", "")),
        "tweet": extract_text(paper_data.get("tweethtml", "")),
    }

    groups = {'a': [], 'b': [], 'c': []}
    try:
        for qdoc in paper.reference.collection('kgainQuestions').stream():
            qdata = qdoc.to_dict() or {}
            qtype = qdata.get("type", "")
            if qtype in groups:
                groups[qtype].append(qdata)
    except Exception as e:
        print(f"[warn] Could not read kgainQuestions for {article_id}: {e}")

    # For each type, sort by vote (desc) and select top 2
    qas = []
    for qtype, qlist in groups.items():
        qlist.sort(key=lambda x: x.get("vote", 0), reverse=True)
        for qdata in qlist[:2]:
            qas.append({
                "question": qdata.get("questionText", ""),
                "answer": qdata.get("correctAnswer", ""),
                "qa_type": map_question_type(qdata.get("type", "")),
                "source": qdata.get("source", ""),
                "evidence": qdata.get("evidence", ""),
                "vote": qdata.get("vote", 0),
                "options": qdata.get("options", {}),
            })

    print(f"Article {article_id}: saved {len(qas)} Qs "
          f"(a={len(groups['a'])}, b={len(groups['b'])}, c={len(groups['c'])})")

    articles.append({
        "article_id": article_id,
        "contents": contents,
        "qas": qas
    })

with open("../data/kgain_dataset.json", "w") as f:
    json.dump({"articles": articles}, f, indent=2)

print("Generated dataset with", len(articles), "articles.")
