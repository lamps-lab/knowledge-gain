import json
import dspy
import math
from collections import Counter, defaultdict
import os

# Teacher model (Temp=0) for consistent evaluation
lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000, temperature=0.0)
dspy.settings.configure(lm=lm)

DATASET_PATH = "../data/kgain_annotated_dataset.json"

PROMPT_FILES = {
    "pre": "human_proxy_pre.json",
    "news": "human_proxy_news.json",
    "abstract": "human_proxy_abstract.json",
    "tweet": "human_proxy_tweet.json"
}

# DSPy Definitions (MUST MATCH OPTIMIZER EXACTLY)

# The optimized prompts were built on these exact signatures.
# If we change field names here, the loading will fail or be meaningless.

class PreSignature(dspy.Signature):
    """Answer based on intuition and gut feeling."""
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class BaseSignature(dspy.Signature):
    """
    You are a participant in a study. Answer the multiple-choice question.
    """
    context = dspy.InputField(desc="Background text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

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
    # 1. Load Data
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at {DATASET_PATH}")
        return
        
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
        
    # 2. Load Models
    models = {}
    print("[INFO] Loading Optimized Prompts...")
    
    try:
        # A. Pre-Knowledge Model
        pre_model = dspy.Predict(PreSignature)
        pre_model.load(PROMPT_FILES['pre'])
        models['pre'] = pre_model
        print(f"  Loaded PRE.")

        # B. Post-Knowledge Models
        for m in ['news', 'abstract', 'tweet']:
            post_model = dspy.Predict(BaseSignature)
            post_model.load(PROMPT_FILES[m])
            models[m] = post_model
            print(f"  Loaded {m.upper()}.")
            
    except Exception as e:
        print(f"[ERROR] Could not load prompt files. {e}")
        return

    # initialize counters & logging
    human_stats = defaultdict(lambda: Counter())
    llm_stats = defaultdict(lambda: Counter())
    
    results_log = []

    print(f"[INFO] Running inference on {len(data)} documents...")

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
            
            # INFERENCE
            
            # A. Pre-Knowledge Inference
            pred_pre = models['pre'](question=q_text, options=options_str)
            pre_ans_int = parse_llm_answer(pred_pre)
            llm_cls_pre = classify_answer(pre_ans_int, correct_idx, len(options_list))
            
            # B. Post-Knowledge Inference
            if media in models:
                pred_post = models[media](context=content, question=q_text, options=options_str)
                post_ans_int = parse_llm_answer(pred_post)
                llm_cls_post = classify_answer(post_ans_int, correct_idx, len(options_list))
                
                # Capture reasoning if available
                post_reasoning = getattr(pred_post, 'reasoning', '')
            else:
                pred_post = None
                post_ans_int = -1
                llm_cls_post = 'dk'
                post_reasoning = ""

            #  save details to log
            results_log.append({
                "doc_id": doc_idx,
                "media": media,
                "question": q_text,
                "correct_option": correct_idx,
                "model_pre_answer": pre_ans_int,
                "model_pre_reasoning": getattr(pred_pre, 'reasoning', ''),
                "model_post_answer": post_ans_int,
                "model_post_reasoning": post_reasoning,
                "classification_pre": llm_cls_pre,
                "classification_post": llm_cls_post
            })

            # AGGREGATE STATS
            for annotator in doc['human_annotations']:
                if q_idx < len(annotator['qa_annotations']):
                    qa = annotator['qa_annotations'][q_idx]
                    
                    # Human Stats
                    h_cls_pre = classify_answer(qa['human-answer-pre'], correct_idx, len(options_list))
                    human_stats['pre'][h_cls_pre] += 1
                    
                    h_cls_post = classify_answer(qa['human-answer-post'], correct_idx, len(options_list))
                    human_stats[media][h_cls_post] += 1
                    
                    # LLM Stats (Weighted by human count)
                    llm_stats['pre'][llm_cls_pre] += 1
                    if media in models:
                        llm_stats[media][llm_cls_post] += 1

        if (doc_idx + 1) % 10 == 0:
            print(f"    Processed {doc_idx + 1} docs...")

    
    with open("evaluation_results_dump.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\n[INFO] Saved detailed responses to 'evaluation_results_dump.json'")

    # 5. Results Printout
    print("\n" + "="*50)
    print("FINAL KL DIVERGENCE RESULTS (LOWER IS BETTER)")
    print("Metric: Accuracy Distribution [Correct, Incorrect, IDK]")
    print("="*50)
    
    tasks = ['pre', 'news', 'abstract', 'tweet']
    
    for task in tasks:
        h_dist = human_stats[task]
        l_dist = llm_stats[task]
        
        kl_score = manual_kl_divergence(h_dist, l_dist)
        
        print(f"\nTASK: {task.upper()}")
        print(f"  KL Divergence: {kl_score:.4f}")
        
        h_total = sum(h_dist.values())
        l_total = sum(l_dist.values())
        
        def get_pct(d, t, k): return (d[k]/t)*100 if t>0 else 0
        
        print(f"  Distribution (Human vs LLM):")
        print(f"    Correct:   {get_pct(h_dist, h_total, 'correct'):.1f}% vs {get_pct(l_dist, l_total, 'correct'):.1f}%")
        print(f"    Incorrect: {get_pct(h_dist, h_total, 'incorrect'):.1f}% vs {get_pct(l_dist, l_total, 'incorrect'):.1f}%")
        print(f"    IDK:       {get_pct(h_dist, h_total, 'dk'):.1f}% vs {get_pct(l_dist, l_total, 'dk'):.1f}%")
if __name__ == "__main__":
    evaluate()