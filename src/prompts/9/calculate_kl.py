import json
import dspy
import math
import os
from collections import Counter, defaultdict

# Keep temperature consistent with optimization to maintain the error distribution
lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000, temperature=1.3)
dspy.settings.configure(lm=lm)

DATASET_PATH = "../../../data/kgain_annotated_dataset.json"

PROMPT_FILES = {
    "pre": "human_proxy_pre.json",
    "news": "human_proxy_news.json",
    "abstract": "human_proxy_abstract.json",
    "tweet": "human_proxy_tweet.json"
}

class PreSignature(dspy.Signature):
    """You are taking a test on a topic you know NOTHING about."""
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class NewsSignature(dspy.Signature):
    """You are a user who ONLY reads headlines."""
    context = dspy.InputField(desc="News article text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class AbstractSignature(dspy.Signature):
    """You are a layperson who DOES NOT UNDERSTAND the text."""
    context = dspy.InputField(desc="Scientific abstract text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class TweetSignature(dspy.Signature):
    """You are a literal-minded bot."""
    context = dspy.InputField(desc="Tweet text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

SIG_MAP = {
    "pre": PreSignature, "news": NewsSignature, 
    "abstract": AbstractSignature, "tweet": TweetSignature
}

def parse_llm_answer(pred_obj):
    """Extracts the integer index from the LLM prediction object."""
    try:
        if not pred_obj.answer: return -1
        # The optimizer might produce text like "Answer: 1", so we clean it aggressively
        txt = str(pred_obj.answer).strip().split('.')[0].split(' ')[0]
        # Remove any non-digit characters just in case
        txt = ''.join(filter(str.isdigit, txt))
        if not txt: return -1
        return int(txt)
    except:
        return -1 

def classify_answer(answer_idx, correct_idx, num_options):
    if answer_idx == -1: return 'dk'              # Parsing failure
    if answer_idx >= num_options: return 'dk'     # EQUALS or GREATER than last option
    if answer_idx == correct_idx: return 'correct'
    return 'incorrect'

def manual_kl_divergence(p_counts, q_counts):
    states = ['correct', 'incorrect', 'dk']
    total_p = sum(p_counts.values())
    total_q = sum(q_counts.values())
    
    if total_p == 0 or total_q == 0: return 0.0
    epsilon = 1e-9
    
    kl_sum = 0.0
    for s in states:
        prob_p = (p_counts[s] + epsilon) / (total_p + 3 * epsilon)
        prob_q = (q_counts[s] + epsilon) / (total_q + 3 * epsilon)
        kl_sum += prob_p * math.log(prob_p / prob_q)
    return kl_sum

def evaluate():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    # 1. Load Models
    models = {}
    print("[INFO] Loading Optimized Prompts...")
    for task_name, filename in PROMPT_FILES.items():
        if os.path.exists(filename):
            try:
                prog = dspy.Predict(SIG_MAP[task_name])
                prog.load(filename)
                models[task_name] = prog
                print(f"  [OK] Loaded {task_name.upper()} from {filename}")
            except Exception as e:
                print(f"  [FAIL] Error loading {filename}: {e}")
    
    human_stats = defaultdict(lambda: Counter())
    llm_stats = defaultdict(lambda: Counter())
    results_log = []

    print(f"\n[INFO] Running inference on {len(data)} documents with Temperature=0.7...")

    for doc_idx, doc in enumerate(data):
        media = doc['content-type']
        content = doc.get('content', "")
        
        if not doc['human_annotations']: continue
        
        reference_qs = doc['human_annotations'][0]['qa_annotations']
        
        for q_idx, q_ref in enumerate(reference_qs):
            q_text = q_ref['question-text']
            options_list = q_ref['options']
            correct_idx = q_ref['correct_option']
            options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options_list)])
            num_opts = len(options_list)

            if 'pre' in models:
                pred_pre = models['pre'](question=q_text, options=options_str)
                pre_ans_int = parse_llm_answer(pred_pre)
                llm_cls_pre = classify_answer(pre_ans_int, correct_idx, num_opts)
            else:
                pred_pre = None
                pre_ans_int = -1
                llm_cls_pre = 'dk'
            
            if media in models:
                pred_post = models[media](context=content, question=q_text, options=options_str)
                post_ans_int = parse_llm_answer(pred_post)
                llm_cls_post = classify_answer(post_ans_int, correct_idx, num_opts)
            else:
                pred_post = None
                post_ans_int = -1
                llm_cls_post = 'dk'

            results_log.append({
                "doc_id": doc_idx,
                "media": media,
                "question": q_text,
                "correct_option": correct_idx,
                "model_pre_answer": pre_ans_int,
                "model_pre_reasoning": getattr(pred_pre, 'reasoning', '') if pred_pre else '',
                "model_post_answer": post_ans_int,
                "model_post_reasoning": getattr(pred_post, 'reasoning', '') if pred_post else '',
                "classification_pre": llm_cls_pre,
                "classification_post": llm_cls_post
            })

            for annotator in doc['human_annotations']:
                if q_idx < len(annotator['qa_annotations']):
                    qa = annotator['qa_annotations'][q_idx]
                    
                    # Human
                    h_cls_pre = classify_answer(qa['human-answer-pre'], correct_idx, num_opts)
                    human_stats['pre'][h_cls_pre] += 1
                    
                    h_cls_post = classify_answer(qa['human-answer-post'], correct_idx, num_opts)
                    human_stats[media][h_cls_post] += 1
                    
                    # LLM (Weighted by human count)
                    llm_stats['pre'][llm_cls_pre] += 1
                    if media in models:
                        llm_stats[media][llm_cls_post] += 1

        if (doc_idx + 1) % 10 == 0:
            print(f"    Processed {doc_idx + 1} docs...")

    # Save Log
    with open("evaluation_results_dump.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\n[INFO] Saved detailed responses to 'evaluation_results_dump.json'")

    # Print Table
    print("\n" + "="*50)
    print("FINAL KL DIVERGENCE RESULTS (LOWER IS BETTER)")
    print("Metric: Accuracy Distribution [Correct, Incorrect, IDK]")
    print("="*50)
    
    tasks = ['pre', 'news', 'abstract', 'tweet']
    
    for task in tasks:
        h_dist = human_stats[task]
        l_dist = llm_stats[task]
        
        kl_score = manual_kl_divergence(h_dist, l_dist)
        h_total = sum(h_dist.values())
        l_total = sum(l_dist.values())
        
        def fmt(d, t, k): return (d[k]/t)*100 if t>0 else 0
        
        print(f"\nTASK: {task.upper()}")
        print(f"  KL Divergence: {kl_score:.4f}")
        print(f"  Distribution (Human vs LLM):")
        print(f"    Correct:   {fmt(h_dist, h_total, 'correct'):.1f}% vs {fmt(l_dist, l_total, 'correct'):.1f}%")
        print(f"    Incorrect: {fmt(h_dist, h_total, 'incorrect'):.1f}% vs {fmt(l_dist, l_total, 'incorrect'):.1f}%")
        print(f"    IDK:       {fmt(h_dist, h_total, 'dk'):.1f}% vs {fmt(l_dist, l_total, 'dk'):.1f}%")

if __name__ == "__main__":
    evaluate()