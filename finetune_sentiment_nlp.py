import warnings
warnings.filterwarnings("ignore")

import os
import shutil
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
    EarlyStoppingCallback
)
import evaluate

# ------------------------------
# 1. CONFIG
# ------------------------------
DATA_PATH = "ARP_dataset_fixed_Sentiment.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_DIR = "./finbert-finetuned"
LOG_DIR = "./logs"
TEST_SIZE_INTERNAL = 0.2   # fraction for internal validation
HOLDOUT_RATIO = 0.3        # fraction held out for final evaluation
NUM_EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-5                 # reduced base learning rate
WEIGHT_DECAY = 0.05       # increased weight decay for regularization
WARMUP_RATIO = 0.1        # fraction of total steps for warmup
WINDOW = 6
RANDOM_SEED = 42

# ------------------------------
# Clear previous outputs
# ------------------------------
for path in [OUTPUT_DIR, LOG_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)

# ------------------------------
# 2. LOAD DATA + CREATE HOLDOUT
# ------------------------------
print("Loading data and creating holdout set...")
df = pd.read_csv(DATA_PATH)

df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED)
holdout_size = int(len(df_shuffled) * HOLDOUT_RATIO)
holdout_df = df_shuffled.iloc[:holdout_size]
train_df_orig = df_shuffled.iloc[holdout_size:]

holdout_df.to_csv("holdout_eval_set.csv", index=False)
print(f"Saved {len(holdout_df)} holdout samples to holdout_eval_set.csv")

# ------------------------------
# 3. PARSE + FILTER TRAINING DATA
# ------------------------------
print("Parsing and filtering training data...")
def parse_entities(row):
    raw = ast.literal_eval(row.get("Entities", "[]"))
    cleaned = []
    for ent, score_raw in raw:
        if not ent or score_raw == '' or score_raw is None:
            continue
        try:
            score = float(score_raw)
        except ValueError:
            continue
        cleaned.append({"name": ent, "score": score})
    return cleaned

train_df_orig["parsed_entities"] = train_df_orig.apply(parse_entities, axis=1)
train_df_filtered = train_df_orig[train_df_orig["parsed_entities"].map(len) > 0].reset_index(drop=True)

# ------------------------------
# 4. FLATTEN TRAINING SET
# ------------------------------
print("Flattening training data...")
flat_records = []
for _, row in train_df_filtered.iterrows():
    for e in row["parsed_entities"]:
        flat_records.append({
            "sentence": row["Sentence"],
            "entity": e["name"],
            "score": e["score"]
        })
flat_df = pd.DataFrame(flat_records)

# ------------------------------
# 5. MAP SCORE TO LABEL
# ------------------------------
unique_scores = sorted(flat_df["score"].unique())
score_to_label = {s: i for i, s in enumerate(unique_scores)}
flat_df["label"] = flat_df["score"].map(score_to_label)

print("\n--- CLASS DISTRIBUTION ---")
counts = flat_df["label"].value_counts().sort_index()
for score, idx in score_to_label.items():
    print(f"Score {score:+.2f} (Label {idx}): {counts.get(idx,0)} samples")

with open("score_to_label.json", "w") as fp:
    json.dump(score_to_label, fp)
print(f"Label mapping saved: {score_to_label}")

# ------------------------------
# 6. CONTEXT SLICING
# ------------------------------
print("Extracting context windows...")
nlp = spacy.load("en_core_web_sm")

def get_context(sentence, aspect, window=WINDOW):
    doc = nlp(sentence)
    toks = aspect.split()
    m = len(toks)
    for i in range(len(doc) - m + 1):
        span = doc[i : i + m]
        if span.text.lower() == aspect.lower():
            start = max(i - window, 0)
            end = min(i + m + window, len(doc))
            return doc[start:end].text
    return sentence

flat_df["context"] = flat_df.apply(lambda r: get_context(r["sentence"], r["entity"]), axis=1)

# ------------------------------
# 7. PREPARE HF DATASET
# ------------------------------
print("Preparing Hugging Face dataset...")
dataset = Dataset.from_pandas(flat_df[["context", "label"]])
dataset = dataset.train_test_split(test_size=TEST_SIZE_INTERNAL, seed=RANDOM_SEED)

# ------------------------------
# 8. TOKENIZE
# ------------------------------
print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(batch["context"], padding="max_length", truncation=True)
tokenized_ds = dataset.map(tokenize, batched=True)

# ------------------------------
# 9. CALCULATE WARMUP STEPS
# ------------------------------
train_steps_per_epoch = (len(tokenized_ds["train"]) + BATCH_SIZE - 1) // BATCH_SIZE
total_training_steps = train_steps_per_epoch * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_training_steps)
print(f"Total steps: {total_training_steps}, Warmup steps: {warmup_steps}")

# ------------------------------
# 10. MODEL + TRAINER
# ------------------------------
print("Loading model and setting up Trainer...")
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

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=LR,
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    load_best_model_at_end=True,
    logging_dir=LOG_DIR
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ------------------------------
# 11. DEBUG PREDICTIONS
# ------------------------------
print("\n--- DEBUG: Sample Predictions ---")
preds = trainer.predict(tokenized_ds["test"])
for i in range(min(10, len(preds.predictions))):
    true = preds.label_ids[i]
    pred = np.argmax(preds.predictions[i])
    print(f"Sample {i+1}: True={true}, Pred={pred}, Logits={preds.predictions[i]}")