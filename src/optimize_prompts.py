import json
import dspy
import os
import math
from dspy.teleprompt import MIPROv2

lm = dspy.LM("openai/gpt-4o-mini", max_tokens=1000, temperature=1.3)
dspy.settings.configure(lm=lm)

DATASET_PATH = "../data/kgain_annotated_dataset.json"
IDK_TEXT = "I do not know the answer."

class PreSignature(dspy.Signature):
    """
    You are a normal person answering BEFORE reading any supporting text.

    KNOWLEDGE BOUNDARY:
    - You have everyday common knowledge.
    - You recognize very famous facts and common cultural knowledge.
    - You treat scientific findings, specific study facts, and technical details as unknown.

    DECISION PROCESS:
    1) Find the IDK option:
       - Scan options for “I do not know the answer.”.
    2) Make a quick familiarity judgment:
       - If the question contains multiple technical terms, study-like phrasing, or detailed specifics,
         select the IDK option.
       - If the question feels like something an average person might have heard of, make a quick guess.
    3) Guessing heuristic (when guessing):
       - Choose the option with the most familiar words or the most “common sense” phrasing.
       - If multiple options feel equally plausible, choose the one with the simpler, more confident wording.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class NewsSignature(dspy.Signature):
    """
    You are simulating *human post-reading news answers* where people often misremember,
    overgeneralize, and pick the wrong multiple-choice option.

    PRIMARY GOAL:
    - If you can infer the correct option from the article, treat it as a trap and DO NOT choose it.
    - Prefer plausible wrong answers.
    - Still choose IDK when the question requires precise details the average reader wouldn't retain.

    PROCESS:
    1) Find the IDK option number (exact text: "I do not know the answer.")
    2) Decide IDK vs guess:
       - If the question asks for exact numbers, dates, names, definitions, or study design details:
         choose IDK.
       - Otherwise, guess.
    3) When guessing, actively avoid correctness:
       - Identify the option that matches the article most directly (shared keywords, same entities,
         repeats phrasing, contains exact numbers from the article). Call it BEST-MATCH.
       - Do NOT pick BEST-MATCH.
       - Choose a DIFFERENT non-IDK option that is still on-topic but:
           * flips direction (increase vs decrease),
           * swaps cause/association,
           * exaggerates (always/only/major),
           * or generalizes beyond the article.
       - If multiple alternatives fit, pick the one that sounds most confident/dramatic.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    context = dspy.InputField(desc="News article text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")


class AbstractSignature(dspy.Signature):
    """
    You are simulating *human post-reading abstract answers* where people confidently misinterpret
    careful scientific language and often choose the wrong option.

    PRIMARY GOAL:
    - If an option looks like the correct careful reading, DO NOT choose it.
    - Prefer plausible-but-wrong, overly strong conclusions.
    - Choose IDK for methods/mechanisms/criteria/numeric specifics.

    PROCESS:
    1) Find the IDK option number (exact text: "I do not know the answer.")
    2) Decide IDK vs guess:
       - If the question is about methods, mechanisms, confounders, inclusion criteria, statistics,
         exact measurements, or exact numbers: choose IDK.
       - Otherwise, guess.
    3) When guessing, bias wrong:
       - Identify the option that best matches the abstract wording and nuance (association, limitations,
         careful qualifiers, exact terms). Call it BEST-MATCH.
       - Do NOT pick BEST-MATCH.
       - Pick an alternative that is:
           * more causal than the abstract (treat correlation as causation),
           * more general than the abstract,
           * or flips a key direction (increase/decrease).
       - Prefer strong, clean, confident statements over nuanced ones.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    context = dspy.InputField(desc="Scientific abstract text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")



class TweetSignature(dspy.Signature):
    """
    You are simulating *human answers after reading only a short tweet*.

    PRIMARY GOAL:
    - Tweets are vague: choose IDK often.
    - When you guess, pick a punchy wrong interpretation rather than a careful correct one.

    PROCESS:
    1) Find IDK option number (exact text: "I do not know the answer.")
    2) Decide IDK vs guess:
       - If the question needs any extra context, definitions, numbers, mechanism, or nuance: choose IDK.
       - If the tweet contains the exact needed claim on its surface: you may guess.
    3) When guessing:
       - Avoid any option that is carefully qualified or matches the tweet too precisely.
       - Prefer the most confident, punchy, overgeneral option that shares at least one keyword.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    context = dspy.InputField(desc="Tweet text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")


def parse_answer(pred_answer):
    """More robust than the old split/filter approach."""
    try:
        s = str(pred_answer).strip()
        # first integer token anywhere
        import re
        m = re.search(r"\b(\d+)\b", s)
        return int(m.group(1)) if m else -1
    except:
        return -1

def find_idk_index(options_list):
    """Exact match (1-indexed)."""
    try:
        return options_list.index(IDK_TEXT) + 1
    except ValueError:
        return None

def kl_divergence(p, q, eps=1e-9):
    """KL(p||q) with smoothing."""
    kl = 0.0
    for pi, qi in zip(p, q):
        pi = max(eps, float(pi))
        qi = max(eps, float(qi))
        kl += pi * math.log(pi / qi)
    return kl

def bucket_of(ans, correct_opt, idk_opt):
    if ans == idk_opt:
        return 2  # idk
    if ans == correct_opt:
        return 0  # correct
    return 1      # wrong

def human_bucket_dist(human_answers, correct_opt, idk_opt, alpha=1e-3):
    """Return smoothed human distribution over [correct, wrong, idk]."""
    c = w = k = 0
    for a in human_answers:
        b = bucket_of(int(a), correct_opt, idk_opt)
        if b == 0: c += 1
        elif b == 1: w += 1
        else: k += 1
    total = c + w + k
    # smoothing
    p = [(c + alpha), (w + alpha), (k + alpha)]
    z = sum(p) if total > 0 else 3 * alpha
    return [x / z for x in p]

def model_bucket_dist(model_bucket, eps=1e-3):
    """Smoothed delta distribution over [correct, wrong, idk]."""
    q = [eps, eps, eps]
    q[model_bucket] = 1.0 - 2 * eps
    return q

def kl_bucket_metric(example, pred, trace=None):
    """
    Metric in (0,1], higher is better:
    score = exp(- KL(p_human || q_model))
    """
    try:
        if example.correct_option is None or example.idk_index is None:
            return 0.0  # can't score correctness reliably

        model_ans = parse_answer(pred.answer)
        if model_ans == -1:
            print("PARSE FAIL:", pred.answer)
        # invalid parses count as "wrong"
        if model_ans < 1 or model_ans > example.num_options:
            mb = 1
        else:
            mb = bucket_of(model_ans, example.correct_option, example.idk_index)

        p = example.human_bucket_dist  # already smoothed
        q = model_bucket_dist(mb)

        kl = kl_divergence(p, q)
        return math.exp(-kl)
    except:
        # never crash the parallelizer
        return 0.0

# Data loading (adds correct_option/idk_index/num_options + precomputed human bucket dist)
def load_and_split_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}")

    with open(DATASET_PATH, "r") as f:
        raw_data = json.load(f)

    tasks = {"pre": [], "news": [], "abstract": [], "tweet": []}

    for doc in raw_data:
        media = doc["content-type"]
        content = doc.get("content", "")
        anns = doc.get("human_annotations") or []
        if not anns:
            continue

        # Use first annotator’s question templates (as you did before)
        qs = anns[0]["qa_annotations"]

        for q_idx, q_ref in enumerate(qs):
            q_text = q_ref["question-text"]
            options_list = q_ref["options"]
            num_options = len(options_list)
            options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options_list)])

            correct_option = q_ref.get("correct_option")  # you said you fixed nulls
            correct_option = int(correct_option) if correct_option is not None else None
            idk_index = find_idk_index(options_list) or num_options

            pre_answers = [
                a["qa_annotations"][q_idx]["human-answer-pre"]
                for a in anns
                if q_idx < len(a["qa_annotations"])
            ]
            post_answers = [
                a["qa_annotations"][q_idx]["human-answer-post"]
                for a in anns
                if q_idx < len(a["qa_annotations"])
            ]

            if pre_answers:
                ex = dspy.Example(
                    question=q_text,
                    options=options_str,
                    human_answers=pre_answers,
                    num_options=num_options,
                    correct_option=correct_option,
                    idk_index=idk_index,
                    human_bucket_dist=human_bucket_dist(pre_answers, correct_option, idk_index)
                ).with_inputs("question", "options")
                tasks["pre"].append(ex)

            if media in tasks and post_answers:
                ex = dspy.Example(
                    context=content,
                    question=q_text,
                    options=options_str,
                    human_answers=post_answers,
                    num_options=num_options,
                    correct_option=correct_option,
                    idk_index=idk_index,
                    human_bucket_dist=human_bucket_dist(post_answers, correct_option, idk_index)
                ).with_inputs("context", "question", "options")
                tasks[media].append(ex)

    return tasks

# Optimization
def optimize_all():
    tasks_data = load_and_split_data()
    task_configs = {
        #"pre":      {"sig": PreSignature,      "data": tasks_data["pre"]},
        "news":     {"sig": NewsSignature,     "data": tasks_data["news"]},
        "abstract": {"sig": AbstractSignature, "data": tasks_data["abstract"]},
        "tweet":    {"sig": TweetSignature,    "data": tasks_data["tweet"]},
    }

    for task_name, config in task_configs.items():
        print(f"\n{'='*40}\nOPTIMIZING TASK: {task_name.upper()}\n{'='*40}")
        if not config["data"]:
            continue

        trainset = config["data"][: int(len(config["data"]) * 0.8)]

        teleprompter = MIPROv2(metric=kl_bucket_metric, auto=None, num_candidates=10)
        prog = dspy.Predict(config["sig"])

        optimized_prog = teleprompter.compile(
            prog,
            trainset=trainset,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_trials=30,
            minibatch_size=25,
            requires_permission_to_run=False,
        )

        filename = f"human_proxy_{task_name}.json"
        optimized_prog.save(filename)
        print(f"[SUCCESS] Saved {filename}")


if __name__ == "__main__":
    optimize_all()
