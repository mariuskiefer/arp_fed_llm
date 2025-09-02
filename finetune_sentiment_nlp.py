import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import re
import json
import ast
import random
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rapidfuzz import fuzz, process

import spacy
from datasets import Dataset
from sklearn.model_selection import train_test_split

import evaluate
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput


# Config
DATA_PATH = "data/ARP_dataset_fixed_Sentiment.csv"
CUSTOM_ENTITIES_PATH = "custom_entities.json"
OUTPUT_DIR = "./finbert-finetuned"
LOG_DIR = "./logs"

TEST_SIZE_INTERNAL = 0.2
HOLDOUT_RATIO = 0.30

# hyperparameters 
NUM_EPOCHS = 12  
BATCH_SIZE = 16
LEARNING_RATE = 1e-5  
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.15  
WINDOW = 8

# Fuzzy / context
MIN_ALIAS_RATIO = 0.80
MIN_SPAN_RATIO = 0.68

# Tokenization
MAX_LENGTH = 160

MODEL_NAME = "yiyanghkust/finbert-tone"
RANDOM_SEED = 42
USE_ENTITY_PREFIX = True

# Ordinal threshold tuning
ENABLE_GLOBAL_THRESHOLD_TUNING = True
THRESH_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LAST_BEST_THRESHOLD = 0.50

# Sample weighting
ENABLE_SAMPLE_WEIGHTING = True
UNCERTAINTY_WEIGHT_CLIP = (0.30, 1.00)
CLASS_WEIGHT_MODE = "inverse_freq"


# Load custom entities
with open(CUSTOM_ENTITIES_PATH, "r") as fh:
    CUSTOM_ENTITIES = json.load(fh)

# Small helpers
def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# Build alias -> canonical map
ALIAS_TO_CANON = {}
CANONICALS = list(CUSTOM_ENTITIES.keys())
for canon, aliases in CUSTOM_ENTITIES.items():
    for alias in aliases:
        ALIAS_TO_CANON[normalize_text(alias)] = canon
ALIAS_KEYS = list(ALIAS_TO_CANON.keys())

def token_ratio(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a, b) / 100.0

@lru_cache(maxsize=4096)
def canonicalize_entity(aspect: str):
    raw = aspect or ""
    norm = normalize_text(raw)
    if not norm:
        return raw, None, 0.0
    if norm in ALIAS_TO_CANON:
        return ALIAS_TO_CANON[norm], norm, 1.0

    best_alias, best_score, _ = process.extractOne(norm, ALIAS_KEYS, scorer=fuzz.token_set_ratio)
    best_score = (best_score or 0.0) / 100.0
    if best_alias and best_score >= MIN_ALIAS_RATIO:
        return ALIAS_TO_CANON[best_alias], best_alias, best_score

    best_score_c, best_canon = 0.0, None
    for canon in CANONICALS:
        sc = token_ratio(norm, normalize_text(canon))
        if sc > best_score_c:
            best_score_c, best_canon = sc, canon
    if best_score_c >= MIN_ALIAS_RATIO:
        return best_canon, None, best_score_c
    return raw, None, best_score


# Context extraction
NLP = spacy.blank("en")

def best_span_match(sentence: str, target_norm: str, window: int):
    doc = NLP(sentence)
    asp_len = max(1, len(target_norm.split()))
    best = {"score": 0.0, "i": 0, "j": 0, "span": "", "context": sentence}
    for n in range(max(1, asp_len - 1), asp_len + 3):
        for i in range(0, len(doc) - n + 1):
            j = i + n
            span_text = doc[i:j].text
            sc = token_ratio(normalize_text(span_text), target_norm)
            if sc > best["score"]:
                best.update({"score": sc, "i": i, "j": j, "span": span_text})
    if best["score"] >= MIN_SPAN_RATIO:
        start = max(0, best["i"] - window)
        end = min(len(doc), best["j"] + window)
        best["context"] = doc[start:end].text
    else:
        best["context"] = sentence
    return best

def get_context_for_entity(sentence: str, raw_entity: str, window: int):
    canon, matched_alias, conf = canonicalize_entity(raw_entity)
    target = matched_alias if matched_alias else normalize_text(canon)
    info = {
        "raw_entity": raw_entity,
        "canonical": canon,
        "alias_used": matched_alias,
        "alias_conf": conf,
        "span_score": 0.0,
        "span_text": "",
        "context": sentence,
    }
    span = best_span_match(sentence, target, window)
    info.update({"span_score": span["score"], "span_text": span["span"], "context": span["context"]})
    return info

# Data loading & holdout
def load_data(data_path: str, holdout_ratio: float, seed: int):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    holdout_size = int(len(df) * holdout_ratio)
    holdout_df = df.iloc[:holdout_size].copy()
    train_df_orig = df.iloc[holdout_size:].copy()
    holdout_df.to_csv("data/holdout_eval_set.csv", index=False)
    return train_df_orig, holdout_df

# Parse entities & flatten
def parse_entity_list(row):
    raw = row.get("Entities", "[]")
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = []
    out = []
    for tup in parsed:
        if not isinstance(tup, (list, tuple)) or len(tup) < 2:
            continue
        ent, score_raw = tup[0], tup[1]
        if not ent or score_raw == '' or score_raw is None:
            continue
        try:
            score = float(score_raw)
        except Exception:
            continue
        out.append({"name": ent, "score": score})
    return out

def build_training_pairs(train_df_orig: pd.DataFrame):
    df = train_df_orig.copy()
    df["parsed_entities"] = df.apply(parse_entity_list, axis=1)
    df = df[df["parsed_entities"].map(len) > 0].reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        for e in row["parsed_entities"]:
            rows.append({
                "sentence": row["Sentence"],
                "entity": e["name"],
                "score": e["score"],
                "source": row.get("Document_ID", row.get("Source", row.get("File", "")))
            })
    return pd.DataFrame(rows)


# Labels
def build_label_map(flat_df: pd.DataFrame):
    unique_scores = sorted(flat_df["score"].unique())
    score_to_label = {s: i for i, s in enumerate(unique_scores)}
    flat_df["label"] = flat_df["score"].map(score_to_label).astype(int)
    with open("score_to_label.json", "w") as fp:
        json.dump(score_to_label, fp)
    return score_to_label

# Context assembly
def assemble_contexts(df_in: pd.DataFrame, window: int):
    rows = []
    for _, r in df_in.iterrows():
        ctx = get_context_for_entity(r["sentence"], r["entity"], window)
        base = r.to_dict()
        rows.append({
            **base,
            "canonical_entity": ctx["canonical"],
            "alias_used": ctx["alias_used"],
            "alias_conf": ctx["alias_conf"],
            "span_score": ctx["span_score"],
            "span_text": ctx["span_text"],
            "context": ctx["context"],
        })
    return pd.DataFrame(rows)


# HF dataset & tokenization
def build_dataset(df_with_context: pd.DataFrame, test_size: float, seed: int):
    assert "label" in df_with_context.columns, "Expected 'label' column"
    if USE_ENTITY_PREFIX:
        texts = [
            f"[E] {ce} [/E] {ctx}" if isinstance(ce, str) and ce else ctx
            for ce, ctx in zip(df_with_context.get("canonical_entity", [""]*len(df_with_context)),
                               df_with_context["context"])
        ]
    else:
        texts = df_with_context["context"].tolist()

    weights = (
        df_with_context["sample_weight"].to_numpy().astype(np.float32)
        if "sample_weight" in df_with_context.columns
        else np.ones(len(df_with_context), dtype=np.float32)
    )

    df = pd.DataFrame({"text": texts, "labels": df_with_context["label"].astype(int), "weight": weights})
    dset = Dataset.from_pandas(df[["text", "labels", "weight"]])
    
    # Use sklearn for stratified split, HF datasets stratification has issues
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(dset))
    labels = df_with_context["label"].values
    
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=seed, 
        stratify=labels
    )
    
    return {
        "train": dset.select(train_idx),
        "test": dset.select(test_idx)
    }

def tokenize_dataset(dset, tokenizer):
    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    # Handle both DatasetDict and regular dict formats
    if hasattr(dset, 'map'):
        # Original HuggingFace DatasetDict format
        cols = dset["train"].column_names
        remove_cols = ["text"] if "text" in cols else None
        return dset.map(tok, batched=True, remove_columns=remove_cols)
    else:
        # Our custom dict format
        cols = dset["train"].column_names
        remove_cols = ["text"] if "text" in cols else None
        return {
            "train": dset["train"].map(tok, batched=True, remove_columns=remove_cols),
            "test": dset["test"].map(tok, batched=True, remove_columns=remove_cols)
        }


# Ordinal CORAL head
class BertForOrdinalCORAL(nn.Module):
    def __init__(self, encoder: AutoModel, hidden_size: int, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.bert = encoder
        self.num_labels = num_labels
        self.num_thresholds = num_labels - 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_thresholds)
        
        # Initialize classifier weights with smaller values to prevent extreme logits
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, K-1)

        loss = None
        if labels is not None:
            thresholds = torch.arange(self.num_thresholds, device=logits.device).unsqueeze(0)  # (1,K-1)
            targets = (labels.unsqueeze(1) > thresholds).float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        return SequenceClassifierOutput(loss=loss, logits=logits)

def build_ordinal_model(model_name: str, num_labels: int):
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size
    return BertForOrdinalCORAL(encoder=encoder, hidden_size=hidden_size, num_labels=num_labels, dropout=0.2)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def preds_from_logits(logits, threshold: float):
    with torch.no_grad():
        probs = torch.sigmoid(torch.tensor(logits))
        return (probs > threshold).sum(dim=1).cpu().numpy().astype(int)

def find_best_threshold(logits, labels, grid):
    best_t, best_f1 = 0.50, -1.0
    for t in grid:
        preds = preds_from_logits(logits, t)
        f1_val = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        if f1_val > best_f1:
            best_t, best_f1 = float(t), float(f1_val)
    return best_t, best_f1

def make_class_weights(labels: np.ndarray, num_labels: int, mode: str = "inverse_freq"):
    if mode is None:
        return np.ones(num_labels, dtype=np.float32)
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / counts if mode == "inverse_freq" else np.ones_like(counts)
    w = w * (num_labels / w.sum())
    return w.astype(np.float32)

def add_sample_weights(ctx_df: pd.DataFrame, num_labels: int):
    class_w = make_class_weights(ctx_df["label"].to_numpy(), num_labels, CLASS_WEIGHT_MODE)
    if "alias_conf" in ctx_df.columns and "span_score" in ctx_df.columns:
        unc = np.minimum(ctx_df["alias_conf"].to_numpy(), ctx_df["span_score"].to_numpy())
        lo, hi = UNCERTAINTY_WEIGHT_CLIP
        unc = np.clip(unc, lo, hi)
    else:
        unc = np.ones(len(ctx_df), dtype=np.float32)
    sw = class_w[ctx_df["label"].to_numpy()] * unc
    out = ctx_df.copy()
    out["sample_weight"] = sw.astype(np.float32)
    return out

# Use validation loss for model selection
def compute_metrics(eval_pred):
    global LAST_BEST_THRESHOLD
    logits, labels = eval_pred
    if ENABLE_GLOBAL_THRESHOLD_TUNING:
        LAST_BEST_THRESHOLD, _ = find_best_threshold(logits, labels, THRESH_GRID)
        preds = preds_from_logits(logits, LAST_BEST_THRESHOLD)
    else:
        preds = preds_from_logits(logits, 0.50)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)
        outputs = model(**inputs, labels=None)
        logits = outputs.logits  # (B, K-1)

        thresholds = torch.arange(logits.size(1), device=logits.device).unsqueeze(0)  # (1, K-1)
        targets = (labels.long().unsqueeze(1) > thresholds).float()
        per_thr = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, K-1)
        per_ex = per_thr.mean(dim=1)

        if weights is not None:
            weights = weights.to(per_ex.dtype)
            per_ex = per_ex * weights

        loss = per_ex.mean()
        return (loss, outputs) if return_outputs else loss


# Cleanup helpers
def cleanup_checkpoints(output_dir: str):
    """Remove all checkpoint folders but keep the final model files"""
    if not os.path.exists(output_dir):
        return
    
    for item in os.listdir(output_dir):
        path = os.path.join(output_dir, item)
        if os.path.isdir(path) and item.startswith("checkpoint"):
            shutil.rmtree(path)
            print(f"Removed checkpoint: {item}")


# Training
def train_model(flat_df: pd.DataFrame, score_to_label: dict):
    if "label" not in flat_df.columns:
        flat_df["label"] = flat_df["score"].map(score_to_label).astype(int)

    print(f"Training with fixed hyperparameters:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Window size: {WINDOW}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")

    ctx_df = assemble_contexts(flat_df[["sentence","entity","score","label"]], WINDOW)
    if ENABLE_SAMPLE_WEIGHTING:
        ctx_df = add_sample_weights(ctx_df, num_labels=len(score_to_label))

    dset = build_dataset(ctx_df, test_size=TEST_SIZE_INTERNAL, seed=RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenize_dataset(dset, tokenizer)

    print(f"Dataset sizes:")
    print(f"  Training: {len(tokenized['train'])}")
    print(f"  Validation: {len(tokenized['test'])}")

    steps_per_epoch = (len(tokenized["train"]) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    model = build_ordinal_model(MODEL_NAME, num_labels=len(score_to_label))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",  
        logging_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",  
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,  
        save_safetensors=True,
        report_to=[],
        logging_dir=LOG_DIR,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=2,  
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=4)]  
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    
    # Clean up any temporary checkpoints, keeping only the final model
    cleanup_checkpoints(OUTPUT_DIR)
    
    # Save the optimal threshold
    with open("best_threshold.json", "w") as f:
        json.dump({"threshold": LAST_BEST_THRESHOLD}, f)
    
    print(f"Training completed. Best model saved to {OUTPUT_DIR}")
    print(f"Optimal threshold: {LAST_BEST_THRESHOLD:.4f} (saved to best_threshold.json)")

    # Preview predictions
    print(f"\nSample predictions on validation set:")
    with open("score_to_label.json", "r") as fp:
        s2l = json.load(fp)
    l2s = {v: float(k) for k, v in s2l.items()}

    pred_output = trainer.predict(tokenized["test"])
    logits = pred_output.predictions
    labels = pred_output.label_ids
    
    # Check logit magnitudes
    print(f"Logit statistics:")
    print(f"  Mean: {np.mean(logits):.4f}")
    print(f"  Std: {np.std(logits):.4f}")
    print(f"  Min: {np.min(logits):.4f}")
    print(f"  Max: {np.max(logits):.4f}")
    
    preds = preds_from_logits(logits, LAST_BEST_THRESHOLD)
    for i in range(min(10, len(labels))):
        tl, pl = int(labels[i]), int(preds[i])
        print(f"  #{i+1:02d}  true={tl} ({l2s.get(tl):.2f})   pred={pl} ({l2s.get(pl):.2f})")
    return trainer

# -----------------
# Main
# -----------------
def main():
    set_seed_everywhere(RANDOM_SEED)
    
    # Clean up previous runs
    for path in [OUTPUT_DIR, LOG_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("Loading and preparing data...")
    train_df_orig, _ = load_data(DATA_PATH, HOLDOUT_RATIO, RANDOM_SEED)
    flat_df = build_training_pairs(train_df_orig)
    score_to_label = build_label_map(flat_df)
    
    print(f"Data statistics:")
    print(f"  Training examples: {len(flat_df)}")
    print(f"  Unique scores: {len(score_to_label)}")
    print(f"  Score distribution: {flat_df['score'].value_counts().sort_index()}")

    trainer = train_model(flat_df, score_to_label)

if __name__ == "__main__":
    main()