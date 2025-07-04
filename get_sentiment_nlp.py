import warnings
warnings.filterwarnings("ignore")

import spacy
from transformers import pipeline

# load spaCy model for context slicing
nlp = spacy.load("en_core_web_sm")

# load FinBERT for financial-domain sentiment
finabsa = pipeline(
    "text-classification",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone",
    return_all_scores=True
)

def get_context(sentence: str, aspect: str, window: int = 8) -> str:
    """
    Extracts a context window of up to `window` tokens on each side of `aspect` in `sentence`.
    Falls back to the full sentence if the aspect span isn't found verbatim.
    """
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


def extract_nlp_sentiment(data):
    """
    data: List[Tuple[str, List[str]]]
    returns: [
      {
        "sentence": str,
        "entities": [
          {"name": str, "sentiment": {"label": str, "score": float}},
          ...
        ]
      },
      ...
    ]
    Uses FinBERT on a sliced context around each aspect, quantizes scores in steps of 1/3 (0.33) using exact fractions.
    """
    output = []
    for sentence, entities in data:
        entity_results = []
        for ent in entities:
            snippet = get_context(sentence, ent)
            try:
                scores = finabsa(snippet)[0]
            except Exception:
                entity_results.append({
                    "name": ent,
                    "sentiment": {"label": "neutral", "score": 0.0}
                })
                continue

            # choose the label with highest confidence
            best = max(scores, key=lambda x: x.get("score", 0.0))
            raw_label = best.get("label", "neutral").lower()
            conf = best.get("score", 0.0)
            if raw_label.startswith("pos"):
                sign, label = 1, "positive"
            elif raw_label.startswith("neg"):
                sign, label = -1, "negative"
            else:
                sign, label = 0, "neutral"
            raw_score = sign * conf
            # quantize to nearest 1/3 step exactly
            quant_score = round(raw_score * 3) / 3
            # clamp to [-1,1]
            quant_score = max(-1.0, min(1.0, quant_score))

            entity_results.append({
                "name": ent,
                "sentiment": {"label": label, "score": round(quant_score, 2)}
            })

        output.append({"sentence": sentence, "entities": entity_results})
    return output
