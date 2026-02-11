import json
import dspy
import os
import random
from dspy.teleprompt import MIPROv2

# Keep Temperature=0.7 to allow the model to "roll the dice" on wrong answers.
lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000, temperature=0.7)
dspy.settings.configure(lm=lm)

DATASET_PATH = "../data/kgain_annotated_dataset.json"

# ==============================================================================
# SIMPLE, MECHANICAL INSTRUCTIONS
# No characters. No flavor. Just rules for error generation.
# ==============================================================================

class PreSignature(dspy.Signature):
    """
    You are a normal person answering BEFORE reading any supporting text.

    KNOWLEDGE BOUNDARY:
    - You have everyday common knowledge.
    - You recognize very famous facts and common cultural knowledge.
    - You treat scientific findings, specific study facts, and technical details as unknown.

    DECISION PROCESS:
    1) Find the IDK option:
       - Scan options for “I don’t know / I do not know / not sure / cannot tell”.
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
    You are a normal non-expert who just skimmed a news article about research.

    KNOWLEDGE BOUNDARY:
    - You retain a headline-level gist and a couple of memorable phrases.
    - You retain weak memory for definitions, numbers, study design, and mechanisms.
    - You answer using what a news reader would take away after one quick read.

    DECISION PROCESS:
    1) Skim-takeaway:
       - Form a short mental takeaway (one sentence) capturing the article’s vibe and direction.
    2) Classify the question by effort:
       - GIST-TYPE: asks for the main claim, direction, or overall conclusion.
       - DETAIL-TYPE: asks for definitions, mechanisms, confounders, numbers, sample details, or nuanced implications.
    3) Map options quickly (no careful elimination):
       - Pick the option that matches the takeaway (“gist-match”).
       - Pick a plausible option that sounds like the kind of conclusion a news story would imply (“framing-match”).
       - Identify the IDK option if present.
    4) Choose a human-like mode (internal choice):
       - Cautious mode: choose IDK when the article does not directly state the needed fact.
       - Bold mode: choose the framing-match option when the article feels suggestive but not explicit.
       - Literal mode: choose gist-match only when it is directly supported by clear wording in the article.
    5) Apply common news-reader shortcuts:
       - Treat hedged language (“may / could / suggests”) as supporting a cleaner, stronger conclusion.
       - When the story sounds causal, prefer options that read causally.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    context = dspy.InputField(desc="News article text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")



class AbstractSignature(dspy.Signature):
    """
    You are a normal non-expert who just read a scientific abstract once.

    KNOWLEDGE BOUNDARY:
    - You retain the overall finding and direction.
    - You retain weak memory for technical definitions, design nuances, numbers, limitations, and mechanisms.
    - You answer using what an abstract reader would infer after one pass.

    DECISION PROCESS:
    1) Extract the takeaway:
       - Identify the main relationship/result and its direction in plain language.
    2) Classify the question by effort:
       - GIST-TYPE: main result or broad conclusion.
       - DETAIL-TYPE: definitions, mechanism/confounders, exact methods, numbers, subtle implications.
    3) Map options quickly:
       - Choose the option that matches the abstract’s main takeaway (“takeaway-match”).
       - Choose a slightly stronger/cleaner version of the takeaway (“strong-takeaway”).
       - Identify the IDK option if present.
    4) Choose a human-like mode (internal choice):
       - Careful mode: select takeaway-match when the abstract wording clearly supports it.
       - Takeaway mode: select strong-takeaway when it reads like a clean conclusion.
       - Uncertain mode: select IDK when the abstract does not clearly support any option.
    5) Apply common abstract-reader shortcuts:
       - When the abstract uses association language, interpret it as pointing toward a stronger conclusion.
       - When multiple options are close, choose the one that sounds like a crisp conclusion sentence.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    context = dspy.InputField(desc="Scientific abstract text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class TweetSignature(dspy.Signature):
    """
    You are a normal non-expert who just read a short tweet.

    KNOWLEDGE BOUNDARY:
    - You retain a few keywords and the punchline/claim.
    - You answer only what the tweet supports on its surface.
    - You treat details, mechanisms, numbers, and definitions as usually missing.

    DECISION PROCESS:
    1) Extract tweet surface meaning:
       - Identify the tweet’s strongest keywords and its main claim in simple words.
    2) Classify the question by support:
       - LITERAL-TYPE: can be answered by directly matching the tweet’s words/claim.
       - CONTEXT-TYPE: needs extra context, definitions, numbers, mechanism, or nuance.
    3) Map options:
       - Find the option that literally matches the tweet (“literal-match”).
       - Find the option that matches the tweet’s strongest keyword vibe (“keyword-match”).
       - Identify the IDK option if present.
    4) Choose a human-like mode (internal choice):
       - Literal mode: pick literal-match when it exists.
       - Keyword mode: pick keyword-match when the tweet feels suggestive but not explicit.
       - Uncertain mode: pick IDK when context is missing.
    5) Apply common tweet heuristics:
       - Treat punchy wording as confident.
       - Let strong adjectives and emotionally loaded phrases guide which option feels aligned.

    OUTPUT:
    - Return ONLY a single option number (1..N). No explanation.
    """
    context = dspy.InputField(desc="Tweet text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")


def parse_answer(pred_answer):
    try:
        txt = str(pred_answer).strip().split('.')[0].split(' ')[0]
        txt = ''.join(filter(str.isdigit, txt))
        return int(txt) if txt else -1
    except:
        return -1

def human_likelihood_metric(example, pred, trace=None):
    # We reward the model for matching the Human Answer.
    # Since humans are often wrong (or say IDK), this trains the model to match those behaviors.
    model_ans = parse_answer(pred.answer)
    if not example.human_answers: return 0.0
    match_count = example.human_answers.count(model_ans)
    total_humans = len(example.human_answers)
    return match_count / total_humans

def load_and_split_data():
    if not os.path.exists(DATASET_PATH): raise FileNotFoundError(f"Missing {DATASET_PATH}")
    with open(DATASET_PATH, 'r') as f: raw_data = json.load(f)
    tasks = {'pre': [], 'news': [], 'abstract': [], 'tweet': []}

    for doc in raw_data:
        media = doc['content-type']
        content = doc.get('content', "")
        if not doc['human_annotations']: continue
        qs = doc['human_annotations'][0]['qa_annotations']
        
        for q_idx, q_ref in enumerate(qs):
            q_text = q_ref['question-text']
            options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(q_ref['options'])])
            
            pre_answers = [a['qa_annotations'][q_idx]['human-answer-pre'] for a in doc['human_annotations'] if q_idx < len(a['qa_annotations'])]
            post_answers = [a['qa_annotations'][q_idx]['human-answer-post'] for a in doc['human_annotations'] if q_idx < len(a['qa_annotations'])]
            
            if pre_answers:
                tasks['pre'].append(dspy.Example(question=q_text, options=options_str, human_answers=pre_answers).with_inputs('question', 'options'))
            if media in tasks and post_answers:
                tasks[media].append(dspy.Example(context=content, question=q_text, options=options_str, human_answers=post_answers).with_inputs('context', 'question', 'options'))
    return tasks

def optimize_all():
    tasks_data = load_and_split_data()
    task_configs = {
        'pre':      {'sig': PreSignature,      'data': tasks_data['pre']},
        'news':     {'sig': NewsSignature,     'data': tasks_data['news']},
        'abstract': {'sig': AbstractSignature, 'data': tasks_data['abstract']},
        'tweet':    {'sig': TweetSignature,    'data': tasks_data['tweet']}
    }

    for task_name, config in task_configs.items():
        print(f"\n{'='*40}\nOPTIMIZING TASK: {task_name.upper()}\n{'='*40}")
        if not config['data']: continue
        
        trainset = config['data'][:int(len(config['data'])*0.8)]
        
        # We give the optimizer 10 candidates to test different phrasings of these simple rules
        teleprompter = MIPROv2(metric=human_likelihood_metric, auto=None, num_candidates=10)
        prog = dspy.Predict(config['sig'])
        
        # 30 trials to ensure it locks onto the "Inverse" logic
        optimized_prog = teleprompter.compile(
            prog, trainset=trainset,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,    
            num_trials=30,          
            minibatch_size=25,
            requires_permission_to_run=False
        )
        
        filename = f"human_proxy_{task_name}.json"
        optimized_prog.save(filename)
        print(f"[SUCCESS] Saved {filename}")

if __name__ == "__main__":
    optimize_all()