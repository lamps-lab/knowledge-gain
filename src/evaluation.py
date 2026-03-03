import json
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

def load_candidates(path="candidates.json"):
    recs = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    for rec in recs:
        ref = rec.get("news_article", "")
        if not ref:
            for c in rec.get("candidates", []):
                if c.get("source") == "original":
                    ref = c.get("text", "")
                    break
        if not ref.strip():
            continue

        for c in rec.get("candidates", []):
            if str(c.get("source","")).startswith("gen_"):
                pred = c.get("text","")
                if pred.strip():
                    rows.append({
                        "article_id": rec.get("article_id", 0),
                        "source": c.get("source"),
                        "reference": ref,
                        "prediction": pred,
                    })
    return rows

rows = load_candidates("candidates.json")
preds = [r["prediction"] for r in rows]
refs  = [r["reference"] for r in rows]

# ROUGE (F1)
sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
r1 = []; r2 = []; rL = []
for ref, pred in zip(refs, preds):
    s = sc.score(ref, pred)
    r1.append(s["rouge1"].fmeasure)
    r2.append(s["rouge2"].fmeasure)
    rL.append(s["rougeL"].fmeasure)

# BLEU
bleu = [
    sacrebleu.sentence_bleu(pred, [ref]).score
    for pred, ref in zip(preds, refs)
]

# BERTScore
P, R, F1 = bertscore(preds, refs, lang="en", rescale_with_baseline=True)

df = pd.DataFrame(rows)
df["rouge1_f1"] = r1
df["rouge2_f1"] = r2
df["rougeL_f1"] = rL
df["bleu"] = bleu
df["bertscore_f1"] = [float(x) for x in F1]

df.to_csv("metrics.csv", index=False)
print(df[["article_id","source","rouge1_f1","rouge2_f1","rougeL_f1","bleu","bertscore_f1"]].head())
print("\nCorpus BLEU:", bleu)
print("Mean ROUGE-1/2/L:", sum(r1)/len(r1), sum(r2)/len(r2), sum(rL)/len(rL))
print("Mean BERTScore F1:", sum(df["bertscore_f1"])/len(df))