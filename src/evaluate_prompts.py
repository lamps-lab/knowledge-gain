import json, os, math, re, time
import dspy

DATASET_PATH = "../data/kgain_annotated_dataset.json"
IDK_TEXT = "I do not know the answer."
EPS = 1e-9

def parse_answer(pred_answer):
    s = str(pred_answer).strip()
    m = re.search(r"\b(\d+)\b", s)
    return int(m.group(1)) if m else -1

def find_idk_index(options_list):
    try:
        return options_list.index(IDK_TEXT) + 1
    except ValueError:
        return len(options_list)

def bucket(ans, correct, idk):
    # 0=correct, 1=wrong, 2=idk
    if ans == idk: return 2
    if correct is not None and ans == correct: return 0
    return 1

def bucket_label(b):
    return {0: "correct", 1: "incorrect", 2: "dk"}[b]

def kl(p, q):
    s = 0.0
    for pi, qi in zip(p, q):
        pi = max(EPS, pi); qi = max(EPS, qi)
        s += pi * math.log(pi / qi)
    return s

def pct(x):
    return f"{100*x:.1f}%"

def print_report(task_name, kl_val, pH, pM):
    print(f"\nTASK: {task_name}")
    print(f"  KL Divergence: {kl_val:.4f}")
    print("  Distribution (Human vs LLM):")
    print(f"    Correct:   {pct(pH[0])} vs {pct(pM[0])}")
    print(f"    Incorrect: {pct(pH[1])} vs {pct(pM[1])}")
    print(f"    IDK:       {pct(pH[2])} vs {pct(pM[2])}")

def options_to_text(options_list):
    return "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options_list)])

class PreSignature(dspy.Signature):
    """
    ROLE:
    - You are a preschool child (about 4–5 years old).
    - You cannot really read sentences. You recognize a few short, common words only.
    - You have not learned science. You do not know technical facts.
    - BEFORE hearing any story/passage, you usually cannot answer and often say you don't know.

    HARD KNOWLEDGE BOUNDARY (absolute limits):
    - No outside knowledge: you cannot use real-world facts from memory.
    - Reading limits:
      * You only understand very short words (about 1–4 letters) like “dog”, “hot”, “sad”, “help”.
      * Longer words are basically “scribbles” to you (they do not carry meaning).
      * Acronyms/initials (DNA, AI, MRI, JWST, LHC) are meaningless symbols.
      * Numbers and units are meaningless.
      * You cannot reliably understand “not / except / least / never”.
      * You do not recognize synonyms (big ≠ large, make ≠ cause, help ≠ improve).
    - Thinking limits:
      * You cannot do elimination or careful comparison across options.
      * You cannot hold the whole question in memory while reading choices.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    - If you can’t truly understand the question/options, you may choose “I do not know the answer.”
    """
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1,2,3...)")


class NewsSignature(dspy.Signature):
    """
    ROLE:
    - You are a preschool child (about 4–5 years old) who just heard a grown-up read a news story.
    - You do not understand most words in a news article.
    - You form a fuzzy “vibe” and often misunderstand what it said.
    - You are eager to answer after hearing a story, even if you didn’t understand it.

    HARD KNOWLEDGE BOUNDARY (absolute limits):
    - Only the provided article exists; you cannot use any outside facts.
    - Reading/comprehension limits:
      * You only understand a few very short everyday words; most of the article is noise.
      * Long words, names, places, and acronyms are meaningless.
      * Numbers, dates, and measurements are meaningless.
      * You cannot reliably understand “not/except/never” or cautious words like “may/might”.
      * You do not recognize synonyms.
    - Thinking limits:
      * You cannot connect multiple sentences together.
      * You cannot compare options carefully or eliminate choices logically.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    - If almost everything is unreadable noise to you, you may choose IDK.
    """
    context = dspy.InputField(desc="News article text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1,2,3...)")


class AbstractSignature(dspy.Signature):
    """
    ROLE:
    - You are a preschool child (about 4–5 years old) looking at a scientific abstract.
    - Scientific writing is almost entirely incomprehensible to you.
    - You often confuse what is being asked and what the text is about.
    - You still try to give an answer because you think you’re supposed to.

    HARD KNOWLEDGE BOUNDARY (absolute limits):
    - Only the abstract text exists; you cannot use outside knowledge.
    - Reading/comprehension limits:
      * You only understand a few very short everyday words; almost everything else is noise.
      * Any technical term, acronym, or scientific name is meaningless.
      * Numbers, statistics, and measurements are meaningless.
      * You cannot understand “not/except/least” reliably.
      * You do not recognize synonyms.
      * You cannot tell what is a method vs a result vs a conclusion.
    - Thinking limits:
      * You cannot do elimination or careful option comparison.
      * You cannot hold the question in mind while reading options.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    - If the question/options are basically all noise, you may choose IDK.
    """
    context = dspy.InputField(desc="Scientific abstract text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1,2,3...)")


class TweetSignature(dspy.Signature):
    """
    ROLE:
    - You are a preschool child (about 4–5 years old) seeing a short tweet.
    - You don’t understand internet shorthand, names, or technical words.
    - You react to a tiny fragment you recognize and often get the meaning wrong.

    HARD KNOWLEDGE BOUNDARY (absolute limits):
    - Only the tweet text exists; you cannot use outside knowledge.
    - Reading/comprehension limits:
      * You only understand a few very short everyday words; most words are noise.
      * Hashtags, usernames, acronyms, and names are meaningless.
      * Numbers are meaningless.
      * Negation/exceptions are not reliably understood.
      * You do not recognize synonyms.
    - Thinking limits:
      * You cannot integrate multiple clauses.
      * You cannot do elimination or careful comparison across options.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    - If everything looks like noise to you, you may choose IDK.
    """
    context = dspy.InputField(desc="Tweet text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1,2,3...)")

if __name__ == "__main__":
    lm = dspy.LM("openai/gpt-4o-mini", temperature=1.7, max_tokens=40)
    dspy.settings.configure(lm=lm)

    pre_prog   = dspy.Predict(PreSignature)
    news_prog  = dspy.Predict(NewsSignature)
    abs_prog   = dspy.Predict(AbstractSignature)
    tweet_prog = dspy.Predict(TweetSignature)

    with open(DATASET_PATH, "r") as f:
        raw = json.load(f)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"runs/run_{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.jsonl")

    # Counters for KL and distributions
    # PRE: one distribution
    human_pre = [0,0,0]; model_pre = [0,0,0]
    # POST: per media type
    human_post = {"news":[0,0,0], "abstract":[0,0,0], "tweet":[0,0,0]}
    model_post = {"news":[0,0,0], "abstract":[0,0,0], "tweet":[0,0,0]}

    num_docs = len(raw)

    with open(out_path, "w", encoding="utf-8") as w:
        for doc_id, doc in enumerate(raw):
            if (doc_id % 10 == 0) or (doc_id == num_docs - 1):
                print(f"Processed {doc_id}/{num_docs} documents...")

            media = doc["content-type"]
            content = doc.get("content", "")
            anns = doc.get("human_annotations") or []
            if not anns:
                continue

            qs = anns[0]["qa_annotations"]
            for q_idx, q_ref in enumerate(qs):
                q_text = q_ref["question-text"]
                options_list = q_ref["options"]
                options_str = options_to_text(options_list)
                n = len(options_list)

                correct = q_ref.get("correct_option")
                correct = int(correct) if correct is not None else None
                idk = find_idk_index(options_list)

                pre_answers  = [a["qa_annotations"][q_idx]["human-answer-pre"]
                                for a in anns if q_idx < len(a["qa_annotations"])]
                post_answers = [a["qa_annotations"][q_idx]["human-answer-post"]
                                for a in anns if q_idx < len(a["qa_annotations"])]

                # --- PRE prediction ---
                pred_pre = pre_prog(question=q_text, options=options_str)
                raw_pre = getattr(pred_pre, "answer", "")
                a_pre = parse_answer(raw_pre)
                if a_pre < 1 or a_pre > n:
                    a_pre = idk
                b_pre = bucket(a_pre, correct, idk)
                model_pre[b_pre] += 1
                for ha in pre_answers:
                    human_pre[bucket(int(ha), correct, idk)] += 1

                # --- POST prediction (only for news/abstract/tweet docs) ---
                a_post = idk
                raw_post = ""
                b_post = 2
                if media in ("news", "abstract", "tweet") and post_answers:
                    if media == "news":
                        pred_post = news_prog(context=content, question=q_text, options=options_str)
                    elif media == "abstract":
                        pred_post = abs_prog(context=content, question=q_text, options=options_str)
                    else:
                        pred_post = tweet_prog(context=content, question=q_text, options=options_str)

                    raw_post = getattr(pred_post, "answer", "")
                    a_post = parse_answer(raw_post)
                    if a_post < 1 or a_post > n:
                        a_post = idk
                    b_post = bucket(a_post, correct, idk)

                    model_post[media][b_post] += 1
                    for ha in post_answers:
                        human_post[media][bucket(int(ha), correct, idk)] += 1

                record = {
                    "doc_id": doc_id,
                    "question_index": q_idx,
                    "media": media,
                    "question": q_text,
                    "correct_option": correct,
                    "model_pre_answer": int(a_pre),
                    "model_pre_reasoning": "",
                    "model_post_answer": int(a_post),
                    "model_post_reasoning": "",
                    "classification_pre": bucket_label(b_pre),
                    "classification_post": bucket_label(b_post),
                }
                w.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Reports
    Hp = sum(human_pre); Mp = sum(model_pre)
    pH_pre = [x/Hp for x in human_pre]
    pM_pre = [x/Mp for x in model_pre]
    print_report("PRE", kl(pH_pre, pM_pre), pH_pre, pM_pre)

    for m in ("news", "abstract", "tweet"):
        Hm = sum(human_post[m]); Mm = sum(model_post[m])
        if Hm == 0 or Mm == 0:
            continue
        pH = [x/Hm for x in human_post[m]]
        pM = [x/Mm for x in model_post[m]]
        print_report(m.upper(), kl(pH, pM), pH, pM)

    print(f"\nSaved predictions to: {out_dir}/")
