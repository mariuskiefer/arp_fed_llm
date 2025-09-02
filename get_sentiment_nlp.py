import warnings
warnings.filterwarnings("ignore")

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

try:
    with open("best_threshold.json", "r") as f:
        DECISION_THRESHOLD = json.load(f)["threshold"]
    print(f"[DEBUG] Loaded optimal threshold: {DECISION_THRESHOLD}")
except FileNotFoundError:
    DECISION_THRESHOLD = 0.50  # fallback
    print(f"[DEBUG] Threshold file not found, using fallback: {DECISION_THRESHOLD}")


print(f"[DEBUG] Config loaded:")
print(f"  FINETUNED_DIR: {FINETUNED_DIR}")
print(f"  BASE_MODEL_NAME: {BASE_MODEL_NAME}")
print(f"  DEF_WINDOW: {DEF_WINDOW}")
print(f"  DECISION_THRESHOLD: {DECISION_THRESHOLD}")

# ------------------------------
# Load label mapping (score<->label) saved by the trainer
# ------------------------------
try:
    with open("score_to_label.json", "r") as fp:
        SCORE2LABEL = json.load(fp)           # {"-1.0":0, "-0.66":1, ...}
    SCORE2LABEL = {float(k): int(v) for k, v in SCORE2LABEL.items()}
    LABEL2SCORE = {v: k for k, v in SCORE2LABEL.items()}   # idx -> score float
    print(f"[DEBUG] Label mapping loaded: {len(SCORE2LABEL)} mappings")
    print(f"  Score ranges: {min(SCORE2LABEL.keys()):.2f} to {max(SCORE2LABEL.keys()):.2f}")
    print(f"  Label indices: {sorted(LABEL2SCORE.keys())}")
except Exception as e:
    print(f"[ERROR] Failed to load score_to_label.json: {e}")
    raise

# ------------------------------
# Load custom entities and build alias map
# ------------------------------
try:
    with open(CUSTOM_ENTITIES_PATH, "r") as fh:
        CUSTOM_ENTITIES = json.load(fh)
    ALIAS_TO_CANON = {}
    for canon, aliases in CUSTOM_ENTITIES.items():
        for alias in aliases:
            ALIAS_TO_CANON[alias.lower().strip()] = canon
    ALIAS_KEYS = list(ALIAS_TO_CANON.keys())
    CANONICALS = list(CUSTOM_ENTITIES.keys())
    print(f"[DEBUG] Custom entities loaded: {len(CUSTOM_ENTITIES)} canonical entities")
    print(f"  Total aliases: {len(ALIAS_KEYS)}")
    print(f"  Sample canonicals: {list(CANONICALS)[:3]}")
except Exception as e:
    print(f"[ERROR] Failed to load custom entities: {e}")
    raise

# ------------------------------
# Tokenizer and spaCy (tokenizer-only)
# ------------------------------
def _pick_tokenizer_source():
    has_tok = any(os.path.exists(os.path.join(FINETUNED_DIR, f))
                  for f in ["tokenizer.json", "vocab.txt", "spiece.model"])
    source = FINETUNED_DIR if has_tok else BASE_MODEL_NAME
    print(f"[DEBUG] Tokenizer source selected: {source}")
    print(f"  Fine-tuned tokenizer files found: {has_tok}")
    return source

TOKENIZER = AutoTokenizer.from_pretrained(_pick_tokenizer_source())
NLP = spacy.blank("en")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Device: {DEVICE}")

# ------------------------------
# Small text helpers
# ------------------------------
def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def token_ratio(a: str, b: str) -> float:
    return (fuzz.token_set_ratio(a, b) or 0.0) / 100.0

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

def build_context_window(sentence: str, aspect: str, window: int = DEF_WINDOW) -> str:
    canon, alias, _ = canonicalize_entity(aspect)
    target = alias if alias else normalize_text(canon)
    span = best_span_match(sentence, target, window)
    return span["context"]

# ------------------------------
# CORAL model (same head as trainer) + loader
# ------------------------------
class BertForOrdinalCORAL(torch.nn.Module):
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

def load_state_dict_from_dir(model_dir: str) -> dict:
    print(f"[DEBUG] Looking for model files in: {model_dir}")
    
    # Check what files exist
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"[DEBUG] Files found: {files}")
    else:
        print(f"[ERROR] Model directory does not exist: {model_dir}")
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    safep = os.path.join(model_dir, "model.safetensors")
    alt_safe = os.path.join(model_dir, "pytorch_model.safetensors")
    binp = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safep):
        print(f"[DEBUG] Loading from model.safetensors")
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("Found model.safetensors but safetensors is not installed. pip install safetensors") from e
        return load_file(safep)
    elif os.path.exists(alt_safe):
        print(f"[DEBUG] Loading from pytorch_model.safetensors")
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("Found pytorch_model.safetensors but safetensors is not installed. pip install safetensors") from e
        return load_file(alt_safe)
    elif os.path.exists(binp):
        print(f"[DEBUG] Loading from pytorch_model.bin")
        return torch.load(binp, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No fine-tuned weights found in {model_dir}. "
            f"Expected one of: model.safetensors, pytorch_model.safetensors, pytorch_model.bin"
        )

def load_coral_model(model_dir: str, num_labels: int) -> BertForOrdinalCORAL:
    print(f"[DEBUG] Building CORAL model with {num_labels} labels")
    
    # Load base encoder
    print(f"[DEBUG] Loading base encoder from: {BASE_MODEL_NAME}")
    enc = AutoModel.from_pretrained(BASE_MODEL_NAME)
    hidden = enc.config.hidden_size
    drop = enc.config.hidden_dropout_prob
    print(f"[DEBUG] Encoder config - hidden_size: {hidden}, dropout: {drop}")
    
    # Create CORAL model
    model = BertForOrdinalCORAL(enc, hidden, num_labels, drop)
    print(f"[DEBUG] CORAL model created with {model.num_thresholds} thresholds")
    
    # Load fine-tuned weights
    state = load_state_dict_from_dir(model_dir)
    print(f"[DEBUG] State dict loaded, keys: {len(state)} total")
    print(f"[DEBUG] Sample state dict keys: {list(state.keys())[:3]}")
    
    # Check if this looks like fine-tuned weights vs base weights
    classifier_keys = [k for k in state.keys() if 'classifier' in k]
    print(f"[DEBUG] Classifier layer keys found: {classifier_keys}")
    
    if classifier_keys:
        sample_classifier_weight = state[classifier_keys[0]]
        print(f"[DEBUG] Sample classifier weights: {sample_classifier_weight.flatten()[:5]}")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[DEBUG] Missing keys when loading state: {missing}")
    if unexpected:
        print(f"[DEBUG] Unexpected keys when loading state: {unexpected}")
    
    model.eval().to(DEVICE)
    print(f"[DEBUG] Model loaded and moved to {DEVICE}")
    
    # Verify model is in eval mode and check a sample parameter
    first_param = next(model.parameters())
    print(f"[DEBUG] Model training mode: {model.training}")
    print(f"[DEBUG] Sample model parameter: {first_param.flatten()[:3]}")
    
    return model

# ------------------------------
# Instantiate model (K = number of ordinal classes)
# ------------------------------
NUM_LABELS = len(LABEL2SCORE)
print(f"[DEBUG] Loading model with {NUM_LABELS} labels")
MODEL = load_coral_model(FINETUNED_DIR, NUM_LABELS)

def preds_from_logits(logits: np.ndarray, threshold: float) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))    # sigmoid
    return (probs > threshold).sum(axis=1).astype(int)

# ------------------------------
# Public API expected by main.py (name unchanged)
# ------------------------------
def extract_nlp_sentiment(data: List[Tuple[str, List[str]]]):
    print(f"[DEBUG] extract_nlp_sentiment called with {len(data)} sentences")
    out = []
    threshold = DECISION_THRESHOLD
    print(f"[DEBUG] Using threshold: {threshold}")

    total_entities = sum(len(entities) if entities else 0 for _, entities in data)
    print(f"[DEBUG] Total entities to process: {total_entities}")

    for sent_idx, (sentence, entities) in enumerate(data):
        names = list(entities) if entities else []
        print(f"[DEBUG] Sentence {sent_idx+1}/{len(data)}: {len(names)} entities")
        
        if sent_idx < 2:  # Debug first 2 sentences in detail
            print(f"[DEBUG]   Sentence text: '{sentence[:100]}...'")
            print(f"[DEBUG]   Entities: {names}")

        # slice context per entity
        snippets, valid_idx = [], []
        for k, name in enumerate(names):
            if not name:
                snippets.append(None)
                continue
            
            # Build context window 
            ctx = build_context_window(sentence, name, DEF_WINDOW)
            
            # Get canonical entity (same process as training)
            canonical, alias_used, conf = canonicalize_entity(name)
            
            # Format with entity prefix to match training
            formatted_ctx = f"[E] {canonical} [/E] {ctx}"
            snippets.append(formatted_ctx)
            valid_idx.append(k)
            
            if sent_idx < 2 and k < 2:  # Debug first 2 entities of first 2 sentences
                print(f"[DEBUG]     Entity {k+1}: '{name}' -> canonical: '{canonical}' (conf: {conf:.3f})")
                print(f"[DEBUG]     Context: '{ctx[:80]}...'")
                print(f"[DEBUG]     Formatted: '{formatted_ctx[:100]}...'")

        # run inference only on valid contexts
        sentiments = ["" for _ in names]
        valid_texts = [snippets[i] for i in valid_idx if snippets[i] is not None]
        
        if valid_texts:
            print(f"[DEBUG]   Running inference on {len(valid_texts)} valid contexts")
            
            # Tokenize
            enc = TOKENIZER(valid_texts, padding=True, truncation=True, max_length=160, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            
            if sent_idx < 2:  # Debug tokenization for first 2 sentences
                print(f"[DEBUG]   Tokenized input shapes: {[(k, v.shape) for k, v in enc.items()]}")
                sample_tokens = TOKENIZER.convert_ids_to_tokens(enc['input_ids'][0])
                print(f"[DEBUG]   Sample tokens: {sample_tokens[:10]}...")
            
            # Model inference
            with torch.no_grad():
                logits = MODEL(**enc).cpu().numpy()  # shape: (B, K-1)
            
            print(f"[DEBUG]   Model output logits shape: {logits.shape}")
            if sent_idx < 2:
                print(f"[DEBUG]   Sample logits: {logits[0]}")
            
            # Convert logits to predictions
            pred_labels = preds_from_logits(logits, threshold)
            pred_scores = [float(LABEL2SCORE.get(int(i), 0.0)) for i in pred_labels]
            
            if sent_idx < 2:
                print(f"[DEBUG]   Predicted labels: {pred_labels}")
                print(f"[DEBUG]   Predicted scores: {pred_scores}")
            
            # Map back to original positions
            for pos, sc in zip(valid_idx, pred_scores):
                sentiments[pos] = sc
        else:
            print(f"[DEBUG]   No valid contexts for sentence {sent_idx+1}")

        ents_out = [{"name": n if n else "", "sentiment": s if n else ""} for n, s in zip(names, sentiments)]
        out.append({"sentence": sentence, "entities": ents_out})

    print(f"[DEBUG] extract_nlp_sentiment completed: {len(out)} sentence results")
    return out