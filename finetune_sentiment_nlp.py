import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import re
from difflib import SequenceMatcher

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

# ---------------------------------
# 0) ENTITY LIST & FUZZY HELPERS
# ---------------------------------
ENTITY_LIST = [
    "Federal Reserve", "Interest Rates", "Inflation", "Employment", "Unemployment", "GDP", "Trade", "Congress",
    "Monetary Policy", "Financial Stability", "Price Stability", "Regulatory Implementation", "Pandemic",
    "Asset Runoff", "Reinvestment", "Money Market", "Bond Market", "Equity Markets", "Financial Markets",
    "Repo Markets", "Fiscal Policy", "Balance Sheet", "Reserves", "Digital Dollar", "Foreign Currencies",
    "Federal Funds", "Demand", "Securities", "War", "Finance", "Debt", "Mortgage", "Maturity", "Credit",
    "Labor Market", "Auction", "Press Conference", "Banking System", "Uncertain", "Development",
    "Economic Outlook", "Countries"
]

# Synonyms/aliases -> canonical entity
_MANUAL_ALIASES = {
    r"\bfed(?:eral)?\s*reserve\b": "Federal Reserve",
    r"\bfed\b": "Federal Reserve",
    r"\bfomc\b": "Federal Reserve",
    r"\bquantitative\s+tightening\b": "Asset Runoff",
    r"\bqt\b": "Asset Runoff",
    r"\bbalance\s+sheet\s+runoff\b": "Asset Runoff",
    r"\bgross\s+domestic\s+product\b": "GDP",
    r"\bcpi\b": "Inflation",
    r"\bpce\b": "Inflation",
    r"\binflation\s+target\b": "Price Stability",
    r"\bunemployment\s+rate\b": "Unemployment",
    r"\bjobless(?:ness|\s+rate)?\b": "Unemployment",
    r"\btreasury\s+market\b": "Bond Market",
    r"\bequities\b": "Equity Markets",
    r"\bmoney\s+markets?\b": "Money Market",
    r"\brepurchase\s+agreements?\b": "Repo Markets",
    r"\bdigital\s+(?:currency|usd|dollar)\b": "Digital Dollar",
}

_DEF_WINDOW = 6
_DEF_MIN_RATIO = 0.68

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _seqsim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _canonicalize(aspect: str) -> str:
    """Map aspect to a canonical entity (aliases first, else best fuzzy hit from ENTITY_LIST)."""
    norm = _normalize(aspect)
    for pat, canon in _MANUAL_ALIASES.items():
        if re.search(pat, norm):
            return canon
    best = (0.0, aspect)
    for ent in ENTITY_LIST:
        sc = _seqsim(_normalize(ent), norm)
        if sc > best[0]:
            best = (sc, ent)
    return best[1] if best[0] >= 0.75 else aspect

# ------------------------------
# 1. CONFIG
# ------------------------------
DATA_PATH = "ARP_dataset_fixed_Sentiment.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"
OUTPUT_DIR = "./finbert-finetuned"
LOG_DIR = "./logs"
TEST_SIZE_INTERNAL = 0.2   # validation split inside the training 90%
HOLDOUT_RATIO = 0.10       # <<< 10% held out for final evaluation
NUM_EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
WINDOW = _DEF_WINDOW
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
print("Loading data and creating 10% holdout set...")
df = pd.read_csv(DATA_PATH)

df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED)
holdout_size = int(len(df_shuffled) * HOLDOUT_RATIO)
holdout_df = df_shuffled.iloc[:holdout_size]
train_df_orig = df_shuffled.iloc[holdout_size:]

# Save holdout in original CSV format (including empty tuples)
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
            continue  # skip empties for training
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
records = []
for _, row in train_df_filtered.iterrows():
    for e in row["parsed_entities"]:
        records.append({
            "sentence": row["Sentence"],
            "entity": e["name"],
            "score": e["score"]
        })
flat_df = pd.DataFrame(records)

# ------------------------------
# 5. MAP SCORES -> CLASS LABELS (7 classes)
# ------------------------------
unique_scores = sorted(flat_df["score"].unique())  # e.g., [-1,-0.66,-0.33,0,0.33,0.66,1]
score_to_label = {s: i for i, s in enumerate(unique_scores)}
flat_df["label"] = flat_df["score"].map(score_to_label).astype(int)

print("\n--- CLASS DISTRIBUTION ---")
label_counts = flat_df["label"].value_counts().sort_index()
for s, idx in score_to_label.items():
    print(f"Score {s:+.2f} (Label {idx}): {label_counts.get(idx, 0)} samples")

with open("score_to_label.json", "w") as fp:
    json.dump(score_to_label, fp)
print(f"Label mapping saved: {score_to_label}")

# ------------------------------
# 6. CONTEXT SLICING (FUZZY)
# ------------------------------
print("Extracting context windows (fuzzy matching)...")
nlp = spacy.load("en_core_web_sm")

def get_context(sentence: str, aspect: str, window: int = WINDOW, min_ratio: float = _DEF_MIN_RATIO) -> str:
    doc = nlp(sentence)
    aspect = _canonicalize(aspect)
    norm_aspect = _normalize(aspect)
    asp_len = max(1, len(norm_aspect.split()))

    best = (0.0, 0, 0)
    for n in range(max(1, asp_len - 1), asp_len + 2):
        for i in range(0, len(doc) - n + 1):
            span = doc[i:i+n]
            score = _seqsim(_normalize(span.text), norm_aspect)
            if score > best[0]:
                best = (score, i, i+n)

    score, i, j = best
    if score >= min_ratio:
        start = max(0, i - window)
        end = min(len(doc), j + window)
        return doc[start:end].text
    return sentence

flat_df["context"] = flat_df.apply(lambda r: get_context(r["sentence"], r["entity"]), axis=1)

# ------------------------------
# 7. HF DATASET (Classification)
# ------------------------------
print("Preparing HF dataset (classification)...")
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
# 9. TRAINER SETUP (Classification)
# ------------------------------
print("Loading classification model...")
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
        "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

steps_per_epoch = (len(tokenized_ds["train"]) + BATCH_SIZE - 1) // BATCH_SIZE
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

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

print("Starting classification training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# ------------------------------
# 10. DEBUG PREDICTIONS
# ------------------------------
print("\n--- DEBUG: Sample Classification Predictions ---")
with open("score_to_label.json", "r") as fp:
    s2l = json.load(fp)
l2s = {v: float(k) for k, v in s2l.items()}

pred_output = trainer.predict(tokenized_ds["test"])
logits = pred_output.predictions
labels = pred_output.label_ids
for i in range(min(10, len(labels))):
    pred_idx = int(np.argmax(logits[i]))
    print(
        f"Sample {i+1}: TrueLabel={labels[i]} TrueScore={l2s.get(int(labels[i]), None)} "
        f"PredLabel={pred_idx} PredScore={l2s.get(pred_idx, None)} "
        f"Logits={np.array2string(logits[i], precision=3)}"
    )
