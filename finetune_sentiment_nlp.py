import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import ast
import spacy
import numpy as np
import json

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate

# ------------------------------
# 1. CONFIG
# ------------------------------
DATA_PATH = "ARP_dataset_fixed_Sentiment.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_DIR = "./finbert-finetuned"
TEST_SIZE_INTERNAL = 0.2   # test size *within* the 70%
HOLDOUT_RATIO = 0.3        # 30% held out for final evaluation
NUM_EPOCHS = 4
BATCH_SIZE = 16
LR = 2e-5
WINDOW = 6
RANDOM_SEED = 42

# ------------------------------
# 2. LOAD AND FLATTEN DATA
# ------------------------------
print("Loading and flattening data...")
df = pd.read_csv(DATA_PATH)

def parse_entities(row):
    entities_raw = ast.literal_eval(row["Entities"])
    entities_clean = []
    for e in entities_raw:
        if e[1] == '' or e[1] is None:  
            score = 0.0  # Assign neutral
        else:
            score = float(e[1])
        entities_clean.append({"name": e[0], "score": score})
    return entities_clean

df["parsed_entities"] = df.apply(parse_entities, axis=1)

flat_data = []
for _, row in df.iterrows():
    for ent in row["parsed_entities"]:
        flat_data.append({
            "sentence": row["Sentence"],
            "entity": ent["name"],
            "score": ent["score"]
        })

flat_df = pd.DataFrame(flat_data)

# ------------------------------
# 3. MAP SCORE TO LABEL INDEX
# ------------------------------
unique_scores = sorted(flat_df["score"].unique())
score_to_label = {s: i for i, s in enumerate(unique_scores)}
flat_df["label"] = flat_df["score"].map(score_to_label)

# Save score-label mapping
with open("score_to_label.json", "w") as f:
    json.dump(score_to_label, f)

print(f"Label mapping saved: {score_to_label}")

# ------------------------------
# 4. CONTEXT SLICING
# ------------------------------
print("Extracting context windows...")
nlp = spacy.load("en_core_web_sm")

def get_context(sentence: str, aspect: str, window: int = WINDOW) -> str:
    doc = nlp(sentence)
    tokens = aspect.split()
    length = len(tokens)
    for i in range(len(doc) - length + 1):
        span = doc[i : i + length]
        if span.text.lower() == aspect.lower():
            start = max(0, i - window)
            end = min(len(doc), i + length + window)
            return doc[start:end].text
    return sentence

flat_df["context"] = flat_df.apply(lambda r: get_context(r["sentence"], r["entity"]), axis=1)

# ------------------------------
# 5. SPLIT DATA (70/30 THEN 80/20)
# ------------------------------
print("Splitting data (70% for fine-tuning, 30% for holdout)...")
finetune_df, holdout_df = train_test_split(
    flat_df, test_size=HOLDOUT_RATIO, random_state=RANDOM_SEED, shuffle=True
)

# Save holdout set for use in all future evaluation
holdout_df.to_csv("holdout_eval_set.csv", index=False)
print(f"Saved {len(holdout_df)} samples for final evaluation to holdout_eval_set.csv")

# Convert 70% to Hugging Face Dataset and split internally 80/20
dataset = Dataset.from_pandas(finetune_df[["context", "label"]])
dataset = dataset.train_test_split(test_size=TEST_SIZE_INTERNAL, seed=RANDOM_SEED)

# ------------------------------
# 6. TOKENIZE
# ------------------------------
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["context"], padding="max_length", truncation=True)

tokenized_ds = dataset.map(tokenize, batched=True)

# ------------------------------
# 7. MODEL + METRICS
# ------------------------------
print("Loading model and metrics...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(score_to_label),
    ignore_mismatched_sizes=True
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

# ------------------------------
# 8. TRAINING
# ------------------------------
print("Starting training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
