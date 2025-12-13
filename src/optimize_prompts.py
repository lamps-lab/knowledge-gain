import json
import dspy
import random
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, COPRO
from collections import Counter

# -------------------------------------------------------------------------
# 1. Configuration & Setup
# -------------------------------------------------------------------------

# Configure your LLM (LLMSim)
# Ideally use GPT-4o or GPT-4-turbo for high-fidelity simulation
lm = dspy.LM('openai/gpt-4o-mini', max_tokens=1000)
dspy.settings.configure(lm=lm)

DATASET_PATH = "../data/kgain_annotated_dataset.json"

# -------------------------------------------------------------------------
# 2. Data Loading & Parsing
# -------------------------------------------------------------------------
def load_and_split_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    tasks = {
        "pre": [],
        "news": [],
        "abstract": [],
        "tweet": []
    }

    for doc in data:
        media_type = doc['content-type'] # news, abstract, tweet
        content = doc['content']
        
        for annotator in doc['human_annotations']:
            # We treat every annotator's response as a distinct training example
            for qa in annotator['qa_annotations']:
                
                # Format options as a string for the LLM
                options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(qa['options'])])
                
                # --- Task 1: Pre-Knowledge (No Content) ---
                # We want to predict 'human-answer-pre'
                tasks["pre"].append(dspy.Example(
                    question=qa['question-text'],
                    options=options_str,
                    # The label we want to mimic:
                    human_choice=str(qa['human-answer-pre']) 
                ).with_inputs("question", "options"))

                # --- Task 2/3/4: Post-Knowledge (With Content) ---
                # We want to predict 'human-answer-post'
                if media_type in tasks:
                    tasks[media_type].append(dspy.Example(
                        context=content,
                        question=qa['question-text'],
                        options=options_str,
                        # The label we want to mimic:
                        human_choice=str(qa['human-answer-post'])
                    ).with_inputs("context", "question", "options"))

    # Shuffle and create splits (Train / Dev)
    splits = {}
    for key, examples in tasks.items():
        random.shuffle(examples)
        # Use a smaller subset for dev to save costs during optimization
        train_size = int(len(examples) * 0.6)
        splits[key] = {
            "train": examples[:train_size],
            "dev": examples[train_size:]
        }
        print(f"Task [{key}]: {len(examples)} total examples ({len(splits[key]['train'])} train, {len(splits[key]['dev'])} dev)")
    
    return splits

# -------------------------------------------------------------------------
# 3. Define DSPy Signatures (The "Prompt Templates")
# -------------------------------------------------------------------------

class PreKnowledgeWorker(dspy.Signature):
    """
    You are a participant in a knowledge study. 
    Answer the question based strictly on your current intuition and prior knowledge. 
    If you do not know, you must select the 'I do not know' option.
    Return only the number of the selected option.
    """
    question = dspy.InputField()
    options = dspy.InputField(desc="Numbered list of options")
    answer = dspy.OutputField(desc="The number of the selected option (e.g., '1', '2', '3')")

class PostKnowledgeWorker(dspy.Signature):
    """
    You are a participant in a knowledge study.
    Read the provided text carefully. Answer the question based on the text and your prior knowledge.
    If the answer is not in the text and you do not know, select 'I do not know'.
    Return only the number of the selected option.
    """
    context = dspy.InputField(desc="The article text to read")
    question = dspy.InputField()
    options = dspy.InputField(desc="Numbered list of options")
    answer = dspy.OutputField(desc="The number of the selected option (e.g., '1', '2', '3')")

# -------------------------------------------------------------------------
# 4. Define the Modules
# -------------------------------------------------------------------------

class HumanSimulatorPre(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought allows the LLM to "reason" like a human before choosing
        self.prog = dspy.ChainOfThought(PreKnowledgeWorker)
    
    def forward(self, question, options):
        return self.prog(question=question, options=options)

class HumanSimulatorPost(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(PostKnowledgeWorker)
    
    def forward(self, context, question, options):
        return self.prog(context=context, question=question, options=options)

# -------------------------------------------------------------------------
# 5. Define the Alignment Metric
# -------------------------------------------------------------------------

def mimicry_metric(gold, pred, trace=None):
    """
    Rewards the LLM if it selects the EXACT same option index as the human.
    This forces the LLM to model human errors and 'IDK' tendencies.
    """
    # Normalize strings (remove periods, spaces)
    pred_clean = pred.answer.strip().split('.')[0].split(' ')[0]
    gold_clean = gold.human_choice.strip()
    
    return pred_clean == gold_clean

# -------------------------------------------------------------------------
# 6. Optimization Loop
# -------------------------------------------------------------------------

def optimize_prompts():
    all_data = load_and_split_data(DATASET_PATH)
    
    # We use BootstrapFewShotWithRandomSearch.
    # It acts as an evolution strategy: it tries different combinations of 
    # few-shot examples (human demos) to find the set that maximizes the metric.
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=mimicry_metric, 
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidate_programs=15, # Increase this for better results (e.g. 10-20)
        num_threads=4
    )

    optimized_programs = {}

    print("\n" + "="*60)
    print("OPTIMIZING PROMPT 1: PRE-KNOWLEDGE (PRIORS)")
    print("="*60)
    
    pre_optimizer = teleprompter.compile(
        student=HumanSimulatorPre(),
        trainset=all_data['pre']['train'][:50], # limiting training set for speed
        valset=all_data['pre']['dev'][:20]
    )
    pre_optimizer.save("human_pre.json")
    optimized_programs['pre'] = pre_optimizer

    # --- Post-Knowledge Optimizations ---
    
    for media in ['news', 'abstract', 'tweet']:
        print("\n" + "="*60)
        print(f"OPTIMIZING PROMPT FOR: {media.upper()}")
        print("="*60)
        
        post_optimizer = teleprompter.compile(
            student=HumanSimulatorPost(),
            trainset=all_data[media]['train'][:50],
            valset=all_data[media]['dev'][:20]
        )
        post_optimizer.save(f"human_post_{media}.json")
        optimized_programs[media] = post_optimizer

    print("\n[DONE] All prompts optimized and saved to JSON.")

if __name__ == "__main__":
    optimize_prompts()