
#%%
import spacy
from transformers import pipeline
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import fitz

#%% --- SentenceEmbedding & Sentiment Model ---
nlp = spacy.load("en_core_web_sm")
sent_model = SentenceTransformer("all-MiniLM-L6-v2")


#%% --- PDF Text Extraction ---
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text().replace('\n', ' ')
        text += page_text
    return text

#%% --- PDF Text Extraction with the sentence before and after ---
def extract_sentences_with_context(text, window=1):
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents]
    sent_contexts = []

    for i in range(len(sents)):
        left = sents[i - window] if i - window >= 0 else ""
        right = sents[i + window] if i + window < len(sents) else ""
        combined = f"{left} {sents[i]} {right}".strip()
        sent_contexts.append((sents[i], combined))
    
    return sent_contexts

#%% --- Custom Entities & Defination ---
CUSTOM_ENTITIES = [
    "Interest rate","inflation","FOMC", "quantitative policy", "balance sheet", "CPI", "PCE",
    "unemployment", "labor market", "wage growth", "financial stability", "forward guidance",
    "economic outlook", "GDP growth", "Macroeconomic policy", "central bank", "Federal Reserve", "Interest Rates", "Inflation",
    "Employment", "GDP", "Trade", "Congress", "Monetary Policy"
]

def extract_entities(text):
    return CUSTOM_ENTITIES

#%% --- Get Sentiment Score Using FinBERT ---
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_score(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
    sentiment_score = -1.0 * probs[0] + 0.0 * probs[1] + 1.0 * probs[2]
    return round(float(sentiment_score), 3)

#%% --- Sentence Classification and Sentiment Scoring ---
def match_labels(sent_contexts, labels, top_k=3, threshold=0.4):
    label_embeddings = sent_model.encode(labels, convert_to_tensor=True)
    results = []

    for original_sent, context_sent in sent_contexts:
        sent_vec = sent_model.encode(context_sent, convert_to_tensor=True)
        sims = util.cos_sim(sent_vec, label_embeddings)[0]
        top_indices = sims.topk(k=top_k).indices

        matched = []
        for idx in top_indices:
            similarity = sims[idx].item()
            if similarity >= threshold:
                matched.append({"entities": labels[idx], "similarity": round(similarity, 3)})
    
        # Sentiment analysis using FinBERT
        score = get_sentiment_score(context_sent, tokenizer, model)       
    
        results.append({
            "sentence": original_sent,
            "context": context_sent,
            "entities": matched,
            "Sentiment": round(score, 3)
        })
    
    return results

#%% --- Main Pipeline ---
def label_sentences_from_pdf(pdf_path, output_path="output_sentence(before&after).json"):
    text = extract_text(pdf_path)
    sent_contexts = extract_sentences_with_context(text)
    labels = extract_entities(text)
    results = match_labels(sent_contexts, labels)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Done.")

#%% --- Example usage ---
pdf_path = "fed_conf.pdf"
label_sentences_from_pdf(pdf_path)



