import sys
import argparse
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import torch

from peft import LoraConfig, get_peft_model, TaskType

parser = argparse.ArgumentParser(description='Finetune Qwen2')
parser.add_argument('directory', help='Path to the output directory')
parser.add_argument('model', help="model id to finetune from HuggingFace, i.e. 'Qwen/Qwen2.5-3B-Instruct'")
args = parser.parse_args()
file_path = args.directory
model_id = args.model

("Number of GPUs available:", torch.cuda.device_count())

# Load pre-split datasets.
with open("../data/train_dataset.json", "r") as f:
    train_samples = json.load(f)
with open("../data/test_dataset.json", "r") as f:
    test_samples = json.load(f)

print(f"number of training schmackles: {len(train_samples)}")
print(f"number of test schmackles: {len(test_samples)}")
# convert list of dicts into dict of lists.
def dict_from_samples(samples):
    return {k: [sample[k] for sample in samples] for k in samples[0]}

train_dataset = Dataset.from_dict(dict_from_samples(train_samples))
test_dataset  = Dataset.from_dict(dict_from_samples(test_samples))

#model_id = "Qwen/Qwen2.5-3B-Instruct"
#"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
def custom_collator(features):
    allowed_keys = ["input_ids", "attention_mask", "labels"]
    batch = {}
    for key in allowed_keys:
        arr = []
        for i, f in enumerate(features):
            if key not in f:
                #print(f"Example {i} missing key: {key}")
                continue
            item = f[key] if torch.is_tensor(f[key]) else torch.tensor(f[key], dtype=torch.long)
            if item.dim() == 0:
                #print(f"Example {i} key '{key}' is scalar, unsqueezing")
                item = item.unsqueeze(0)
            arr.append(item)
        #print(f"Batch key '{key}' shapes:", [a.shape for a in arr])
        batch[key] = torch.stack(arr)
    return batch

class DebugTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        #print("Training step inputs:")
        #for k, v in inputs.items():
        #    if torch.is_tensor(v):
        #        print(f"  {k}: {v.shape}")
        #    else:
        #        print(f"  {k}: {type(v)}")
        # Proceed with standard training_step.
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):#num_items_in_batch):
        outputs = model(**inputs)
        loss = outputs.loss
        # Ensure loss is at least 1-dimensional.
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return (loss, outputs) if return_outputs else loss

# Modify the preprocess function to include explicit instructions.
def preprocess_function(examples):
    instruction = (
        "You are a helpful assistant that answers multiple choice questions given different types of content. "
        "Output must be a JSON object with the keys: 'question', 'options', and 'correct_answer'."
    )
    # build prompt with question, context & answer.
    texts = [
        f"{instruction}\n\nquestion: {q}\ncontent: {c}\noptions: {o}\ncorrect answer: {a}"
        for q, c, o, a in zip(examples["question"], examples["content"], examples["options"], examples["answer"])
    ]
    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length", return_token_type_ids=False)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=list(train_dataset.features.keys()))
test_dataset  = test_dataset.map(preprocess_function, batched=True, remove_columns=list(test_dataset.features.keys()))

#print(train_dataset[0])
#exit()

#"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id)
print(model)

# In LoRA, the model's weight matrix W is adapted as:
#   W = W0 + ΔW, where ΔW = A * B,
# with A ∈ ℝ^(d×r) and B ∈ ℝ^(r×k), and r is the rank.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # specify the task type
    inference_mode=False,          # training mode
    r=4,                         # LoRA rank (r)
    lora_alpha=8,               # scaling factor α
    lora_dropout=0.1,            # dropout probability for LoRA layers
    target_modules=["q_proj", "v_proj"]  # target modules for LoRA injection (adjust as needed)
)

model = get_peft_model(model, lora_config)
print("LoRA parameters:")
model.print_trainable_parameters()

#"../models/finetuned_qwen_"
# 6. DATA COLLATOR: For causal language modeling (mlm=False)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. TRAINING SETUP
training_args = TrainingArguments(
    output_dir=file_path,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    #save_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.15,
    logging_steps=10,
    #predict_with_generate=True,
    fp16=torch.cuda.is_available()
)

trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=custom_collator,
    tokenizer=tokenizer,
)

# 8. FINETUNE THE MODEL & SAVE
trainer.train()
trainer.save_model(file_path)
