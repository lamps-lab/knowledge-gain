import json
import dspy
import os
import random
from dspy.teleprompt import MIPROv2

lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000, temperature=0.7)
dspy.settings.configure(lm=lm)

DATASET_PATH = "../data/kgain_annotated_dataset.json"

# Signatures optimized for HUMAN ERROR
class PreSignature(dspy.Signature):
    """
    You are a regular "average person" with NO special knowledge. 
    Answer the question based ONLY on your intuition and gut feeling. 
    If a regular person wouldn't know this fact off the top of their head, you DO NOT KNOW the answer.
    """
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class NewsSignature(dspy.Signature):
    """
    You are a distracted reader skimming the news. 
    You often miss details or misinterpret the main point.
    Do NOT simply say "I don't know" if you are unsure. Instead, make a GUESS based on the headlines or keywords that stand out to you, even if that leads to a wrong answer.
    """
    context = dspy.InputField(desc="News article text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class AbstractSignature(dspy.Signature):
    """
    You are a layperson trying to read a complex scientific abstract.
    You understand very little of the jargon.
    However, you don't want to admit ignorance. Try to GUESS the answer based on words that look similar to the options. 
    You are likely to pick an answer just because it repeats a word from the text.
    """
    context = dspy.InputField(desc="Scientific abstract text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

class TweetSignature(dspy.Signature):
    """
    You are scrolling through social media quickly.
    You rely on quick heuristics and gut feelings.
    If the tweet is confusing, you might just guess the most controversial or obvious option. Only say "I don't know" if it is completely impossible to guess.
    """
    context = dspy.InputField(desc="Tweet text")
    question = dspy.InputField()
    options = dspy.InputField()
    answer = dspy.OutputField(desc="A single number (1, 2, 3...)")

# The Alignment Metric
def parse_answer(pred_answer):
    try:
        txt = str(pred_answer).strip().split('.')[0].split(' ')[0]
        txt = ''.join(filter(str.isdigit, txt))
        return int(txt) if txt else -1
    except:
        return -1

def human_agreement_metric(example, pred, trace=None):
    """
    Score = Proportion of humans that the model agreed with.
    """
    model_ans = parse_answer(pred.answer)
    
    if not example.human_answers:
        return 0.0
        
    match_count = example.human_answers.count(model_ans)
    total_humans = len(example.human_answers)
    
    return match_count / total_humans

# Data Loading
def load_and_split_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Cannot find dataset at {DATASET_PATH}")

    with open(DATASET_PATH, 'r') as f:
        raw_data = json.load(f)

    tasks = {'pre': [], 'news': [], 'abstract': [], 'tweet': []}

    for doc in raw_data:
        media = doc['content-type']
        content = doc.get('content', "")
        
        if not doc['human_annotations']: continue
        
        qs = doc['human_annotations'][0]['qa_annotations']
        
        for q_idx, q_ref in enumerate(qs):
            q_text = q_ref['question-text']
            options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(q_ref['options'])])
            
            pre_answers = []
            post_answers = []
            
            for annotator in doc['human_annotations']:
                if q_idx < len(annotator['qa_annotations']):
                    qa = annotator['qa_annotations'][q_idx]
                    pre_answers.append(qa['human-answer-pre'])
                    post_answers.append(qa['human-answer-post'])
            
            # --- PRE EXAMPLE ---
            tasks['pre'].append(dspy.Example(
                question=q_text,
                options=options_str,
                human_answers=pre_answers
            ).with_inputs('question', 'options'))

            # --- POST EXAMPLE ---
            if media in tasks:
                tasks[media].append(dspy.Example(
                    context=content,
                    question=q_text,
                    options=options_str,
                    human_answers=post_answers
                ).with_inputs('context', 'question', 'options'))

    return tasks

# Optimization Loop
def optimize_all():
    tasks_data = load_and_split_data()
    
    task_configs = {
        'pre':      {'sig': PreSignature,      'data': tasks_data['pre']},
        'news':     {'sig': NewsSignature,     'data': tasks_data['news']},
        'abstract': {'sig': AbstractSignature, 'data': tasks_data['abstract']},
        'tweet':    {'sig': TweetSignature,    'data': tasks_data['tweet']}
    }

    for task_name, config in task_configs.items():
        print(f"\n{'='*60}")
        print(f"OPTIMIZING TASK: {task_name.upper()}")
        print(f"{'='*60}")
        
        dataset = config['data']
        total_len = len(dataset)
        
        if total_len == 0:
            print(f"No data for {task_name}, skipping.")
            continue

        train_size = int(total_len * 0.8)
        random.shuffle(dataset)
        trainset = dataset[:train_size]
        
        print(f"Total Examples: {total_len}")
        print(f"Optimization Set: {len(trainset)}")

        teleprompter = MIPROv2(
            metric=human_agreement_metric, 
            auto=None, 
            num_candidates=7
        )
        
        prog = dspy.Predict(config['sig'])
        
        print(f"Starting optimization (generating 7 instruction candidates, running 20 trials)...")
        
        optimized_prog = teleprompter.compile(
            prog,
            trainset=trainset,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,    
            num_trials=20,          
            minibatch_size=25
        )
        
        filename = f"human_proxy_{task_name}.json"
        optimized_prog.save(filename)
        print(f"[SUCCESS] Saved optimized prompt to {filename}")
        
        try:
            print(f"--- Learned Instruction for {task_name} ---")
            print(optimized_prog.predictors()[0].signature.instructions)
        except:
            pass

if __name__ == "__main__":
    optimize_all()