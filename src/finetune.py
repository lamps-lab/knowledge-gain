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

# Helper: convert list of dicts into dict of lists.
def dict_from_samples(samples):
    return {k: [sample[k] for sample in samples] for k in samples[0]}

train_dataset = Dataset.from_dict(dict_from_samples(train_samples))
test_dataset  = Dataset.from_dict(dict_from_samples(test_samples))

model_id = "Qwen/Qwen2.5-7B-Instruct"
#"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print(model)

# build prompt with question, context & answer.
def preprocess_function(examples):
    # Each training example is a full sequence:
    # "question: {question}\ncontext: {content}\nanswer: {answer}"
    texts = [
        f"question: {q}\ncontext: {c}\nanswer: {a}" 
        for q, c, a in zip(examples["question"], examples["content"], examples["answer"])
    ]
    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    # For causal LM, the labels are the same as input_ids.
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

exit()
# 9. TESTING THE MODEL
# Evaluate the model on the test dataset.
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Generate predictions using trainer.predict().
test_output = trainer.predict(test_dataset)
predicted_ids = test_output.predictions
decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
print("\nSample predictions from test dataset:")
for i, text in enumerate(decoded_preds[:5]):
    print(f"Sample {i+1}: {text}")

# Alternatively, generate output for a single test sample using model.generate.
sample = test_dataset[0]
input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
generated_ids = model.generate(input_ids, max_length=512, do_sample=False)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nSingle sample prompt:")
print(tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
print("Generated output:")
print(generated_text)
