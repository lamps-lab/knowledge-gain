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

# Load pre-split datasets.
with open("../data/train_dataset.json", "r") as f:
    train_samples = json.load(f)
with open("../data/test_dataset.json", "r") as f:
    test_samples = json.load(f)

# convert list of dicts into dict of lists.
def dict_from_samples(samples):
    return {k: [sample[k] for sample in samples] for k in samples[0]}

train_dataset = Dataset.from_dict(dict_from_samples(train_samples))
test_dataset  = Dataset.from_dict(dict_from_samples(test_samples))

model_id = "Qwen/Qwen2.5-7B-Instruct"
#"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print(model)

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
    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=list(train_dataset.features.keys()))
test_dataset  = test_dataset.map(preprocess_function, batched=True, remove_columns=list(test_dataset.features.keys()))

# 6. DATA COLLATOR: For causal language modeling (mlm=False)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# 7. TRAINING SETUP
training_args = TrainingArguments(
    output_dir="../models/finetuned_qwen",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    #save_strategy="epoch",
    learning_rate=5e-6,
    weight_decay=0.1,
    logging_steps=10,
    #predict_with_generate=True,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 8. FINETUNE THE MODEL & SAVE
trainer.train()
trainer.save_model("../models/finetuned_qwen")
