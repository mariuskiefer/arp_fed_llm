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
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid

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

# -----------------
# Config
# -----------------
DATA_PATH = "ARP_dataset_fixed_Sentiment.csv"
CUSTOM_ENTITIES_PATH = "custom_entities.json"
OUTPUT_DIR = "./finbert-finetuned"
LOG_DIR = "./logs"

TEST_SIZE_INTERNAL = 0.2
HOLDOUT_RATIO = 0.10

NUM_EPOCHS = 10
BATCH_SIZE = 16
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1

# CV (quick sweep)
CV_SUBSET_FRACTION = 0.3
CV_NUM_EPOCHS = 2
CV_PARAM_GRID = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "WINDOW": [4, 6, 8],
}
CV_RANDOM_STATE = 42

# Fuzzy / context
MIN_ALIAS_RATIO = 0.80
MIN_SPAN_RATIO = 0.68

# Tokenization
MAX_LENGTH = 160

MODEL_NAME = "yiyanghkust/finbert-tone"
RANDOM_SEED = 42
USE_ENTITY_PREFIX = True

# Ordinal threshold tuning (Branch A)
ENABLE_GLOBAL_THRESHOLD_TUNING = True
THRESH_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LAST_BEST_THRESHOLD = 0.50

# Sample weighting (Branch B)
ENABLE_SAMPLE_WEIGHTING = True
UNCERTAINTY_WEIGHT_CLIP = (0.30, 1.00)      # clip(min(alias_conf, span_score))
CLASS_WEIGHT_MODE = "inverse_freq"          # or None

# -----------------
# Load custom entities (simple read)
# -----------------
with open(CUSTOM_ENTITIES_PATH, "r") as fh:
    CUSTOM_ENTITIES = json.load(fh)

# -----------------
# Small helpers
# -----------------
# Lowercase + strip punctuation + squeeze spaces
def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# Fix all RNGs so runs are repeatable
def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

# Build alias -> canonical map once
ALIAS_TO_CANON = {}
CANONICALS = list(CUSTOM_ENTITIES.keys())
for canon, aliases in CUSTOM_ENTITIES.items():
    for alias in aliases:
        ALIAS_TO_CANON[normalize_text(alias)] = canon
ALIAS_KEYS = list(ALIAS_TO_CANON.keys())

# String similarity using rapidfuzz token-set ratio normalized to [0,1]
def token_ratio(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a, b) / 100.0

# Map raw aspect to canonical; return (canonical, matched_alias, confidence)
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

# -----------------
# Context extraction (just spaCy tokenizer; no heavy pipeline)
# -----------------
NLP = spacy.blank("en")

# Slide a small window over tokens and pick the span that best matches the target
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

# Given a sentence + raw aspect, produce local context and matching metadata
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

# -----------------
# Data loading & holdout
# -----------------
# Shuffle, carve out a holdout split, keep the rest for model dev
def load_data(data_path: str, holdout_ratio: float, seed: int):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    holdout_size = int(len(df) * holdout_ratio)
    holdout_df = df.iloc[:holdout_size].copy()
    train_df_orig = df.iloc[holdout_size:].copy()
    holdout_df.to_csv("holdout_eval_set.csv", index=False)
    return train_df_orig, holdout_df

# -----------------
# Parse entities & flatten
# -----------------
# Parse the Entities column (list of (entity, score)) and clean it up
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

# Explode rows into (sentence, entity, score) tuples to train on
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

# -----------------
# Labels
# -----------------
# Freeze the ordinal ladder and map to 0..K-1
def build_label_map(flat_df: pd.DataFrame):
    unique_scores = sorted(flat_df["score"].unique())
    score_to_label = {s: i for i, s in enumerate(unique_scores)}
    flat_df["label"] = flat_df["score"].map(score_to_label).astype(int)
    with open("score_to_label.json", "w") as fp:
        json.dump(score_to_label, fp)
    return score_to_label

# -----------------
# Context assembly
# -----------------
# Attach canonical entity + a trimmed context window to each example
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

# -----------------
# HF dataset & tokenization
# -----------------
# Make a HF Dataset with text/labels/weights, plus a clean train/test split
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
    return dset.train_test_split(test_size=test_size, seed=seed)

# Tokenize to fixed MAX_LENGTH and drop raw text (labels/weights stay)
def tokenize_dataset(dset, tokenizer):
    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    cols = dset["train"].column_names
    remove_cols = ["text"] if "text" in cols else None
    return dset.map(tok, batched=True, remove_columns=remove_cols)

# -----------------
# Ordinal CORAL head, weighting, metrics
# -----------------
class BertForOrdinalCORAL(nn.Module):
    # BERT encoder + CORAL ordinal head (predicts K-1 thresholds with BCE-with-logits)
    def __init__(self, encoder: AutoModel, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = encoder
        self.num_labels = num_labels
        self.num_thresholds = num_labels - 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_thresholds)

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

# Load the FinBERT encoder and bolt on the CORAL head
def build_ordinal_model(model_name: str, num_labels: int):
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size
    dropout = encoder.config.hidden_dropout_prob
    return BertForOrdinalCORAL(encoder=encoder, hidden_size=hidden_size, num_labels=num_labels, dropout=dropout)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Turn K-1 logits into a single ordinal label via a global threshold
def preds_from_logits(logits, threshold: float):
    with torch.no_grad():
        probs = torch.sigmoid(torch.tensor(logits))
        return (probs > threshold).sum(dim=1).cpu().numpy().astype(int)

# Grid-search a single probability threshold to maximize macro-F1 on the dev split
def find_best_threshold(logits, labels, grid):
    best_t, best_f1 = 0.50, -1.0
    for t in grid:
        preds = preds_from_logits(logits, t)
        f1_val = f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        if f1_val > best_f1:
            best_t, best_f1 = float(t), float(f1_val)
    return best_t, best_f1

# Class weights for imbalance; normalize mean weight ~1
def make_class_weights(labels: np.ndarray, num_labels: int, mode: str = "inverse_freq"):
    if mode is None:
        return np.ones(num_labels, dtype=np.float32)
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / counts if mode == "inverse_freq" else np.ones_like(counts)
    w = w * (num_labels / w.sum())
    return w.astype(np.float32)

# Combine class weights with a light uncertainty signal from fuzzy matching
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

# Eval hook: optionally tune the global threshold, then report accuracy + macro-F1
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
    # Trainer override where I apply per-example weights to the CORAL loss
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

# -----------------
# Trainer factory
# -----------------
# Build a Trainer that only keeps the best checkpoint by eval_macro_f1
def create_trainer(
    num_labels,
    tokenized_ds,
    tokenizer,
    out_dir,
    lr,
    weight_decay,
    warmup_ratio,
    batch_size,
    num_epochs,
    log_dir,
    early_stop_patience=3,
    save_strategy="epoch",
    load_best=True,
):
    steps_per_epoch = (len(tokenized_ds["train"]) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    model = build_ordinal_model(MODEL_NAME, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy=save_strategy,
        logging_strategy="epoch",
        learning_rate=lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        save_safetensors=True,
        report_to=[],          # keep logs off-disk for CV
        logging_dir=log_dir,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stop_patience)] if load_best else []
    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

# -----------------
# CV sweep
# -----------------
# Take a stratified slice of the data (by label) so CV is quick but balanced
def stratified_sample(df_with_labels: pd.DataFrame, frac: float, seed: int):
    frac = max(0.05, min(frac, 1.0))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-frac, random_state=seed)
    y = df_with_labels["label"].values
    idx = np.arange(len(df_with_labels))
    keep_idx, _ = next(sss.split(idx, y))
    return df_with_labels.iloc[keep_idx].reset_index(drop=True)

# Remove empty "cv_*" dirs that Trainer may create under OUTPUT_DIR
def prune_empty_cv_dirs(base_dir: str):
    if not os.path.isdir(base_dir): return
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("cv_"):
            if all(len(files)==0 for _,_,files in os.walk(path)):
                shutil.rmtree(path, ignore_errors=True)

# Tiny grid over (lr, window) on a stratified subset; pick the best macro-F1
def run_cv_sweep(base_flat_df: pd.DataFrame, score_to_label: dict, param_grid: dict, subset_fraction: float, seed: int):
    print("\n[CV] preparing subset and parameter grid...")
    flat_df = base_flat_df.copy()
    if "label" not in flat_df.columns:
        flat_df["label"] = flat_df["score"].map(score_to_label).astype(int)
    sub_df = stratified_sample(flat_df, subset_fraction, seed)
    print(f"[CV] subset size: {len(sub_df)} / {len(flat_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results, grid = [], list(ParameterGrid(param_grid))

    for i, params in enumerate(grid, 1):
        lr = float(params["learning_rate"]); window = int(params["WINDOW"])
        tag = f"cv_lr{lr}_w{window}"
        print(f"[CV {i}/{len(grid)}] training {tag} ...")

        ctx_df = assemble_contexts(sub_df[["sentence","entity","score","label"]], window)
        if ENABLE_SAMPLE_WEIGHTING:
            ctx_df = add_sample_weights(ctx_df, num_labels=len(score_to_label))

        dset = build_dataset(ctx_df, test_size=TEST_SIZE_INTERNAL, seed=seed)
        tokenized = tokenize_dataset(dset, tokenizer)

        trainer = create_trainer(
            num_labels=len(score_to_label),
            tokenized_ds=tokenized,
            tokenizer=tokenizer,
            out_dir=os.path.join(OUTPUT_DIR, tag),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            batch_size=BATCH_SIZE,
            num_epochs=CV_NUM_EPOCHS,
            log_dir=LOG_DIR,
            early_stop_patience=2,
            save_strategy="no",
            load_best=False
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        macro_f1 = float(eval_metrics.get("eval_macro_f1", eval_metrics.get("macro_f1", 0.0)))
        acc = float(eval_metrics.get("eval_accuracy", eval_metrics.get("accuracy", 0.0)))
        print(f"[CV {i}/{len(grid)}] {tag}: macroF1={macro_f1:.4f}  acc={acc:.4f}")
        results.append({"learning_rate": lr, "WINDOW": window, "macro_f1": macro_f1, "accuracy": acc})

    prune_empty_cv_dirs(OUTPUT_DIR)

    cv_df = pd.DataFrame(results).sort_values(["macro_f1","accuracy"], ascending=[False, False])
    cv_df.to_csv("cv_results.csv", index=False)
    best = cv_df.iloc[0].to_dict()
    with open("best_params.json", "w") as fp:
        json.dump(best, fp, indent=2)
    print(f"[CV] best: {best}")
    return best

# -----------------
# Full training
# -----------------
# Train on the full set using the best CV params and save the single best model
def train_full_model(flat_df: pd.DataFrame, score_to_label: dict, lr: float, window: int):
    if "label" not in flat_df.columns:
        flat_df["label"] = flat_df["score"].map(score_to_label).astype(int)

    ctx_df = assemble_contexts(flat_df[["sentence","entity","score","label"]], window)
    if ENABLE_SAMPLE_WEIGHTING:
        ctx_df = add_sample_weights(ctx_df, num_labels=len(score_to_label))

    dset = build_dataset(ctx_df, test_size=TEST_SIZE_INTERNAL, seed=RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenize_dataset(dset, tokenizer)

    trainer = create_trainer(
        num_labels=len(score_to_label),
        tokenized_ds=tokenized,
        tokenizer=tokenizer,
        out_dir=OUTPUT_DIR,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        log_dir=LOG_DIR,
        early_stop_patience=3,
        save_strategy="epoch",
        load_best=True
    )

    print("\n[TRAIN] starting full training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)  # best only
    print("[TRAIN] best model saved.")

    print("\n[PREVIEW] sample predictions (threshold tuned = {:.2f})".format(LAST_BEST_THRESHOLD))
    with open("score_to_label.json", "r") as fp:
        s2l = json.load(fp)
    l2s = {v: float(k) for k, v in s2l.items()}

    pred_output = trainer.predict(tokenized["test"])
    logits = pred_output.predictions
    labels = pred_output.label_ids
    preds = preds_from_logits(logits, LAST_BEST_THRESHOLD if ENABLE_GLOBAL_THRESHOLD_TUNING else 0.50)
    for i in range(min(10, len(labels))):
        tl, pl = int(labels[i]), int(preds[i])
        print(f"  #{i+1:02d}  true={tl} ({l2s.get(tl)})   pred={pl} ({l2s.get(pl)})")
    return trainer

# -----------------
# Tiny cleanup helpers for artifacts we don't need to keep
# -----------------
# remove a file if it exists
def rm(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        pass
    except PermissionError as e:
        print(f"[WARN] Could not delete {path}: {e}")

def cleanup_cv_files():
    rm("cv_results.csv")
    rm("best_params.json")
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.startswith("checkpoint"):
                rm(os.path.join(OUTPUT_DIR, fname))

# -----------------
# Main
# -----------------
# End-to-end: load, prep, CV, train-best, preview, cleanup CV artifacts
def main():
    set_seed_everywhere(RANDOM_SEED)

    for path in [OUTPUT_DIR, LOG_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)

    train_df_orig, _ = load_data(DATA_PATH, HOLDOUT_RATIO, RANDOM_SEED)
    flat_df = build_training_pairs(train_df_orig)
    score_to_label = build_label_map(flat_df)

    best = run_cv_sweep(
        base_flat_df=flat_df,
        score_to_label=score_to_label,
        param_grid=CV_PARAM_GRID,
        subset_fraction=CV_SUBSET_FRACTION,
        seed=CV_RANDOM_STATE
    )
    best_lr = float(best["learning_rate"]); best_window = int(best["WINDOW"])
    _ = train_full_model(flat_df, score_to_label, best_lr, best_window)

    # remove CV json/csv artifacts after the full run
    cleanup_cv_files()

if __name__ == "__main__":
    main()
