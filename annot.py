import spacy
import fitz  # PyMuPDF
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- Custom Domain-Specific Entities ---
CUSTOM_ENTITIES = {
    "Federal Reserve": ["Federal Reserve", "the Fed", "FOMC", "Committee", "central bank"],
    "Interest Rates": ["interest rates", "policy rate", "federal funds rate", "rate hike", "rate cut"],
    "Inflation": ["inflation", "CPI", "PCE", "core inflation", "price pressures", "price level"],
    "Employment": ["unemployment", "employment", "labor market", "job gains", "wage growth", "job creation"],
    "GDP": ["GDP", "gross domestic product", "economic growth", "growth rate", "output"],
    "Trade": ["tariffs", "trade policy", "imports", "exports", "trade deficit"],
    "Congress": ["Congress", "the Hill", "lawmakers", "fiscal policy", "legislation"],
    "Monetary Policy": ["monetary policy", "tightening", "easing", "quantitative easing", "QE", "QT"],
    "Countries": ["China", "U.S.", "United States", "Eurozone", "Europe", "Japan"]
}

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text().replace('\n', ' ')
        text += page_text
    return text

# --- Keyword-based Entity Matching ---
def extract_custom_entities(sentence, entity_dict):
    found = set()
    for label, keywords in entity_dict.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', sentence, re.IGNORECASE):
                found.add(label)
    return list(found)

# --- Get Sentiment Score Using FinBERT ---
def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()

    # FinBERT label order: [negative, neutral, positive]
    sentiment_score = -1.0 * probs[0] + 0.0 * probs[1] + 1.0 * probs[2]
    return round(float(sentiment_score), 3)  # rounded for readability

# --- Main Pipeline ---
def main(pdf_path, output_json_path):
    # Load NLP and FinBERT
    print("Loading models...")
    nlp = spacy.load("en_core_web_sm")  # for sentence splitting
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    # Extract and process text
    print("Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path)
    doc = nlp(raw_text)

    output = []
    print("Processing sentences...")
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        matched_entities = extract_custom_entities(sentence_text, CUSTOM_ENTITIES)

        if matched_entities:
            sentiment_score = get_sentiment_score(sentence_text, tokenizer, model)
            entities = [{"name": e, "sentiment": sentiment_score} for e in matched_entities]
        else:
            entities = [{"name": "", "sentiment": ""}]

        output.append({
            "sentence": sentence_text,
            "entities": entities
        })

    # Save results
    print(f"Saving to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Processed {len(output)} sentences.")

# --- Example usage ---
if __name__ == "__main__":
    main("fed_conf.pdf", "fed_sentences_with_entities.json")
