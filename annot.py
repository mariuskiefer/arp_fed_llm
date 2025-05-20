import spacy
import fitz  # PyMuPDF
import json
import re

#TODO:
# - Improve Custom Entities
# - Fix sentiment score allocation

# comment to check if test branch works


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

# --- Sentence Splitting + Entity Tagging ---
def split_sentences_and_entities(text, nlp):
    doc = nlp(text)
    output = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        matched_entities = extract_custom_entities(sentence_text, CUSTOM_ENTITIES)
        output.append({
            "sentence": sentence_text,
            "entities": [{"name": e, "sentiment": None} for e in matched_entities]
        })

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
