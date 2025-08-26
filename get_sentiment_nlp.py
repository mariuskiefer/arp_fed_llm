import warnings
warnings.filterwarnings("ignore")

# make sure the right model is used: Include debug output for terminal


import os
import re
import json
from typing import List, Tuple

import numpy as np
import torch
import spacy
from rapidfuzz import fuzz, process
from transformers import AutoTokenizer, AutoModel

# ------------------------------
# Config & paths
# ------------------------------
FINETUNED_DIR = "./finbert-finetuned"         # must match trainer OUTPUT_DIR
BASE_MODEL_NAME = "yiyanghkust/finbert-tone"  # encoder/tokenizer source
CUSTOM_ENTITIES_PATH = "custom_entities.json"
DEF_WINDOW = 8
DEF_MIN_RATIO = 0.68
DECISION_THRESHOLD = 0.65

# ------------------------------
# Load label mapping (score<->label) saved by the trainer
# ------------------------------
with open("score_to_label.json", "r") as fp:
    SCORE2LABEL = json.load(fp)           # {"-1.0":0, "-0.66":1, ...}
SCORE2LABEL = {float(k): int(v) for k, v in SCORE2LABEL.items()}
LABEL2SCORE = {v: k for k, v in SCORE2LABEL.items()}   # idx -> score float

# ------------------------------
# Load custom entities and build alias map
# ------------------------------
with open(CUSTOM_ENTITIES_PATH, "r") as fh:
    CUSTOM_ENTITIES = json.load(fh)
ALIAS_TO_CANON = {}
for canon, aliases in CUSTOM_ENTITIES.items():
    for alias in aliases:
        ALIAS_TO_CANON[alias.lower().strip()] = canon
ALIAS_KEYS = list(ALIAS_TO_CANON.keys())
CANONICALS = list(CUSTOM_ENTITIES.keys())

# ------------------------------
# Tokenizer and spaCy (tokenizer-only)
# ------------------------------
def _pick_tokenizer_source():
    has_tok = any(os.path.exists(os.path.join(FINETUNED_DIR, f))
                  for f in ["tokenizer.json", "vocab.txt", "spiece.model"])
    return FINETUNED_DIR if has_tok else BASE_MODEL_NAME

TOKENIZER = AutoTokenizer.from_pretrained(_pick_tokenizer_source())
NLP = spacy.blank("en")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Small text helpers
# ------------------------------
# normalize_text: lowercase + strip punctuation + squeeze spaces
def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# token_ratio: fuzzy similarity in [0,1] using token-set ratio
def token_ratio(a: str, b: str) -> float:
    return (fuzz.token_set_ratio(a, b) or 0.0) / 100.0

# canonicalize_entity: map raw aspect to canonical name via aliases or nearest canonical
def canonicalize_entity(aspect: str):
    raw = aspect or ""
    norm = normalize_text(raw)
    if not norm:
        return raw, None, 0.0
    if norm in ALIAS_TO_CANON:
        return ALIAS_TO_CANON[norm], norm, 1.0

    best_alias, best_score, _ = process.extractOne(norm, ALIAS_KEYS, scorer=fuzz.token_set_ratio)
    best_score = (best_score or 0.0) / 100.0
    if best_alias and best_score >= 0.80:
        return ALIAS_TO_CANON[best_alias], best_alias, best_score

    best_c, best_sc = None, 0.0
    for c in CANONICALS:
        sc = token_ratio(norm, normalize_text(c))
        if sc > best_sc:
            best_sc, best_c = sc, c
    if best_sc >= 0.80:
        return best_c, None, best_sc
    return raw, None, best_score

# best_span_match: slide a small window; pick the span most similar to the target alias
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
    if best["score"] >= DEF_MIN_RATIO:
        start = max(0, best["i"] - window)
        end = min(len(doc), best["j"] + window)
        best["context"] = doc[start:end].text
    else:
        best["context"] = sentence
    return best

# build_context_window: canonicalize + slice a local context
def build_context_window(sentence: str, aspect: str, window: int = DEF_WINDOW) -> str:
    canon, alias, _ = canonicalize_entity(aspect)
    target = alias if alias else normalize_text(canon)
    span = best_span_match(sentence, target, window)
    return span["context"]

# ------------------------------
# CORAL model (same head as trainer) + loader
# ------------------------------
class BertForOrdinalCORAL(torch.nn.Module):
    # BERT encoder + CORAL ordinal head (predicts K-1 thresholds with BCE-with-logits at train time)
    def __init__(self, encoder: AutoModel, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = encoder
        self.num_labels = num_labels
        self.num_thresholds = num_labels - 1
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, self.num_thresholds)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, K-1)
        return logits

# load_state_dict_from_dir: handle safetensors or torch .bin
def load_state_dict_from_dir(model_dir: str) -> dict:
    safep = os.path.join(model_dir, "model.safetensors")
    alt_safe = os.path.join(model_dir, "pytorch_model.safetensors")
    binp = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safep):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("Found model.safetensors but safetensors is not installed. pip install safetensors") from e
        return load_file(safep)
    if os.path.exists(alt_safe):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("Found pytorch_model.safetensors but safetensors is not installed. pip install safetensors") from e
        return load_file(alt_safe)
    if os.path.exists(binp):
        return torch.load(binp, map_location="cpu")
    raise FileNotFoundError(
        f"No fine-tuned weights found in {model_dir}. "
        "Expected one of: model.safetensors, pytorch_model.safetensors, pytorch_model.bin"
    )

# load_coral_model: build encoder from base model (correct vocab/config), then load fine-tuned state dict
def load_coral_model(model_dir: str, num_labels: int) -> BertForOrdinalCORAL:
    enc = AutoModel.from_pretrained(BASE_MODEL_NAME)
    hidden = enc.config.hidden_size
    drop = enc.config.hidden_dropout_prob
    model = BertForOrdinalCORAL(enc, hidden, num_labels, drop)

    state = load_state_dict_from_dir(model_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Optional: you can print or log missing/unexpected here if you want to sanity-check

    model.eval().to(DEVICE)
    return model

# ------------------------------
# Instantiate model (K = number of ordinal classes)
# ------------------------------
NUM_LABELS = len(LABEL2SCORE)  # 7 for your ladder
MODEL = load_coral_model(FINETUNED_DIR, NUM_LABELS)

# preds_from_logits: apply the global threshold to map K-1 logits → label id
def preds_from_logits(logits: np.ndarray, threshold: float) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))    # sigmoid
    return (probs > threshold).sum(axis=1).astype(int)

# ------------------------------
# Public API expected by main.py (name unchanged)
# ------------------------------
def extract_nlp_sentiment(data: List[Tuple[str, List[str]]]):
    # data: [(sentence, [entity1, entity2, ...]), ...]
    # returns: [{"sentence": s, "entities": [{"name": e, "sentiment": float|''}, ...]}, ...]
    out = []
    threshold = DECISION_THRESHOLD

    for sentence, entities in data:
        names = list(entities) if entities else []

        # slice context per entity
        snippets, valid_idx = [], []
        for k, name in enumerate(names):
            if not name:
                snippets.append(None)
                continue
            ctx = build_context_window(sentence, name, DEF_WINDOW)
            snippets.append(ctx)
            valid_idx.append(k)

        # run inference only on valid contexts
        sentiments = ["" for _ in names]
        valid_texts = [snippets[i] for i in valid_idx]
        if valid_texts:
            enc = TOKENIZER(valid_texts, padding=True, truncation=True, max_length=160, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = MODEL(**enc).cpu().numpy()  # shape: (B, K-1)
            pred_labels = preds_from_logits(logits, threshold)
            pred_scores = [float(LABEL2SCORE.get(int(i), 0.0)) for i in pred_labels]
            for pos, sc in zip(valid_idx, pred_scores):
                sentiments[pos] = sc

        ents_out = [{"name": n if n else "", "sentiment": s if n else ""} for n, s in zip(names, sentiments)]
        out.append({"sentence": sentence, "entities": ents_out})
    return out
