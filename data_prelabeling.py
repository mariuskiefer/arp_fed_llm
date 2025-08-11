import spacy
import fitz  # PyMuPDF
import json
import re

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
========

#TODO:
# - Improve Custom Entities
# - Fix sentiment score allocation


# --- Custom Domain-Specific Entities ---
CUSTOM_ENTITIES = {
    "Federal Reserve": ["Federal Reserve", "the Fed", "FOMC", "Committee", "central bank"],
    "Interest Rates": ["interest rates", "policy rate", "federal funds rate", "rate hike", "rate cut"],
    "Inflation": ["inflation", "CPI", "PCE", "core inflation", "price pressures", "price level"],
    "Employment": ["employment", "job gains", "wage growth", "job creation", "labor force participation"],
    "Unemployment": ["unemployment", "job losses", "layoffs", "jobless claims", "unemployment rate"],
    "GDP": ["GDP", "gross domestic product", "economic growth", "growth rate", "output"],
    "Trade": ["tariffs", "trade policy", "imports", "exports", "trade deficit"],
    "Congress": ["Congress", "the Hill", "lawmakers", "fiscal policy", "legislation"],
    "Monetary Policy": ["monetary policy", "tightening", "easing", "quantitative easing", "QE", "QT"],
    "Countries": ["China", "U.S.", "United States", "Eurozone", "Europe", "Japan"],
    "Economy": ["economy", "economic conditions", "economic activity", "economic outlook", "economic recovery"]
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
    found = {}  # Changed to dict to store entity and its context
    for label, keywords in entity_dict.items():
        for kw in keywords:
            match = re.search(r'\b' + re.escape(kw) + r'\b', sentence, re.IGNORECASE)
            if match:
                # Get the context around the keyword (5 words before and after)
                start = max(0, match.start() - 50)
                end = min(len(sentence), match.end() + 50)
                context = sentence[start:end]
                found[label] = context
    return found


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
def process_pdf(pdf_path, nlp, tokenizer, model):
    # Extract and process text
    print(f"Processing {pdf_path}...")
    raw_text = extract_text_from_pdf(pdf_path)
    doc = nlp(raw_text)
    
========
# --- Sentence Splitting + Entity Tagging ---
def split_sentences_and_entities(text, nlp):
    doc = nlp(text)

    output = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        matched_entities = extract_custom_entities(sentence_text, CUSTOM_ENTITIES)


        if matched_entities:
            entities = []
            for entity, context in matched_entities.items():
                sentiment_score = get_sentiment_score(context, tokenizer, model)
                entities.append({"name": entity, "sentiment": sentiment_score})
        else:
            entities = [{"name": "", "sentiment": ""}]

        output.append({
            "sentence": sentence_text,
            "entities": entities,
            "document_id": os.path.basename(pdf_path)

        output.append({
            "sentence": sentence_text,
            "entities": [{"name": e, "sentiment": None} for e in matched_entities]

        })
    
    return output

def main(folder_path, output_json_path):
    # Load NLP and FinBERT
    print("Loading models...")
    nlp = spacy.load("en_core_web_sm")  # for sentence splitting
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    all_output = []
    # Process all PDFs in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            output = process_pdf(pdf_path, nlp, tokenizer, model)
            all_output.extend(output)


    # Save results
    print(f"Saving to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Processed {len(all_output)} sentences from {len(os.listdir(folder_path)) - 1} documents.")

# --- Example usage ---
if __name__ == "__main__":
    main("arp_pdfs", "fed_sentences_with_entities.json")

    return output

# --- Save to JSON ---
def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --- Main Pipeline ---
def main(pdf_path, output_json_path):
    nlp = spacy.load("en_core_web_sm")  # only used for sentence splitting
    raw_text = extract_text_from_pdf(pdf_path)
    annotated_data = split_sentences_and_entities(raw_text, nlp)
    save_to_json(annotated_data, output_json_path)
    print(f"Processed {len(annotated_data)} sentences.")

# Example usage
main("fed_conf.pdf", "fed_sentences_with_entities.json")

