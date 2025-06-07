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

#%% --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text().replace('\n', ' ')
        text += page_text
    return text

#%% --- SentenceEmbedding & Sentiment Model ---
sent_model = SentenceTransformer('all-MiniLM-L6-v2')  
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def extract_phrases(sentence):
    doc = nlp(sentence)
    return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

#%% --- Custom Entities & Defination ---
CUSTOM_ENTITIES = {
    "Federal Reserve": "Federal Reserve is the central bank of the United States, responsible for conducting monetary policy, regulating banks, and maintaining financial stability",
    "FOMC": "FOMC isthe Federal Open Market Committee that sets U.S. monetary policy",
    "Quantitative policy": "Quantitative policy is a monetary policy involving changes in the central bank’s balance sheet, including asset purchases or reductions",
    "Balance sheet": "Balance sheet is the statement of financial position of the central bank or financial institution",
    "CPI": "Consumer Price Index CPI, a measure of inflation based on consumer goods prices",
    "PCE": "Personal Consumption Expenditures index PCE, a key inflation measure watched by the Fed",
    "Financial stability": "Financial stability is the resilience of the financial system to shocks and stress",
    "Forward guidance": "Forward guidance is a central bank's communication about future monetary policy intentions",
    "Economic outlook": "Economic outlook is the projected performance of the economy over a future period",
    "Macroeconomic policy": "Macroeconomic policy is the policies aimed at influencing broad economic factors like growth and inflation",
    "Interest Rates": "Interest Rates is the cost of borrowing money, often set by the central bank",
    "Inflation": "Inflation is the rate at which the general price level of goods and services rises",
    "Employment": "Employment is the state of having paid work, used to measure economic activity",
    "GDP": "Gross Domestic Product GPD, the total output of goods and services in an economy",
    "Trade": "Trade is the exchange of goods and services across borders or within an economy",
    "Congress": "Congress is the U.S. legislative body which influences fiscal and regulatory policy",
    "Monetary Policy": "Monetary Policy is the process by which a central bank manages money supply and interest rates"
}

entities_embeddings = {entities: sent_model.encode(desc, convert_to_tensor=True) for entities, desc in CUSTOM_ENTITIES.items()}

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
def classify_and_score(sentence):
    # Encode the entire sentence for semantic similarity
    sent_emb = sent_model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarity between the sentence and all predefined entity embeddings
    similarities = {
        entities: util.cos_sim(sent_emb, emb).item()
        for entities, emb in entities_embeddings.items()
    }

    # Keep the top 3 most relevant entity labels
    top_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]

    # Sentiment analysis using FinBERT
    score = get_sentiment_score(sentence, tokenizer, model)
    
    return {
        #"sentence": sentence,
        "top_entities": [
            {"entities": lbl, "similarity": round(sim, 3)} for lbl, sim in top_entities
        ],
        "sentiment": round(score, 3)
    }


#%% --- Main Pipeline ---
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    results = []
    for sent in sentences:
        entry = {
            "sentence": sent,
            "results": classify_and_score(sent)
        }
        results.append(entry)

    return results


#%% --- Example usage ---
pdf_path = "fed_conf.pdf" 
output = process_pdf(pdf_path)

with open("output_financial_analysis_sentence.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"✅ Done.")
