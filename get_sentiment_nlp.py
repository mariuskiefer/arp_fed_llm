# get_sentiment_nlp.py
# Uses the finetuned 7-class classifier. Same fuzzy context window + aliases.
# Input (from main.py): List[Tuple[str, List[str]]]
# Output: [{"sentence": str, "entities": [{"name": str, "sentiment": float|''}, ...]}, ...]

import re
import json
from difflib import SequenceMatcher

import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------
# ENTITY LIST + ALIASES (same as trainer)
# ------------------------------
ENTITY_LIST = [
    "Federal Reserve", "Interest Rates", "Inflation", "Employment", "Unemployment", "GDP", "Trade", "Congress",
    "Monetary Policy", "Financial Stability", "Price Stability", "Regulatory Implementation", "Pandemic",
    "Asset Runoff", "Reinvestment", "Money Market", "Bond Market", "Equity Markets", "Financial Markets",
    "Repo Markets", "Fiscal Policy", "Balance Sheet", "Reserves", "Digital Dollar", "Foreign Currencies",
    "Federal Funds", "Demand", "Securities", "War", "Finance", "Debt", "Mortgage", "Maturity", "Credit",
    "Labor Market", "Auction", "Press Conference", "Banking System", "Uncertain", "Development",
    "Economic Outlook", "Countries"
]

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
# Load finetuned model + mapping
# ------------------------------
_FINETUNED_DIR = "./finbert-finetuned"  # must match training OUTPUT_DIR
_TOKENIZER = AutoTokenizer.from_pretrained(_FINETUNED_DIR)
_MODEL = AutoModelForSequenceClassification.from_pretrained(_FINETUNED_DIR)

# label<->score map (written by trainer)
with open("score_to_label.json", "r") as fp:
    _S2L = json.load(fp)  # e.g., {"-1.0":0, "-0.66":1, ...} or numbers as strings
# normalize keys to float
_S2L = {float(k): int(v) for k, v in _S2L.items()}
_L2S = {v: k for k, v in _S2L.items()}  # idx -> score

# ------------------------------
# spaCy for context slicing
# ------------------------------
_NLP = spacy.load("en_core_web_sm")

def _get_context(sentence: str, aspect: str, window: int = _DEF_WINDOW, min_ratio: float = _DEF_MIN_RATIO) -> str:
    doc = _NLP(sentence)
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

# ------------------------------
# Public API expected by main.py
# ------------------------------
def extract_nlp_sentiment(data):
    """
    data: List[Tuple[str, List[str]]]
    returns: [
      {
        "sentence": str,
        "entities": [
          {"name": str, "sentiment": float|''},
          ...
        ]
      },
      ...
    ]
    """
    out = []
    for sentence, entities in data:
        names = list(entities) if entities else []
        # Build contexts per entity
        snippets = []
        idx_map = []  # keep indices of valid (non-empty) names
        for k, name in enumerate(names):
            if not name:
                snippets.append(None)
                continue
            ctx = _get_context(sentence, name)
            snippets.append(ctx)
            idx_map.append(k)

        # Tokenize only valid snippets
        valid_snippets = [snippets[k] for k in idx_map]
        sentiments = ["" for _ in names]

        if valid_snippets:
            enc = _TOKENIZER(valid_snippets, padding=True, truncation=True, return_tensors="pt")
            with _MODEL.device:
                pass  # no-op; rely on default device
            with __import__("torch").no_grad():
                logits = _MODEL(**enc).logits.cpu().numpy()
            pred_idx = np.argmax(logits, axis=-1)
            # map class index -> score value (float)
            pred_scores = [float(_L2S.get(int(i), 0.0)) for i in pred_idx]
            for pos, sc in zip(idx_map, pred_scores):
                sentiments[pos] = sc  # write back to the correct entity slot

        # Build output structure
        ents_out = []
        for name, sc in zip(names, sentiments):
            if not name:
                ents_out.append({"name": "", "sentiment": ""})
            else:
                ents_out.append({"name": name, "sentiment": sc})
        out.append({"sentence": sentence, "entities": ents_out})
    return out
