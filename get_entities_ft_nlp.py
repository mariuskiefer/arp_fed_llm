#%%
import os
import torch
from typing import List
from transformers import BertTokenizerFast, BertForSequenceClassification

#%%

ENTITY_LIST = [
    "Federal Reserve", "Interest Rates", "Inflation", "Employment", "Unemployment", "GDP", "Trade", "Congress", 
    "Monetary Policy", "Financial Stability", "Price Stability", "Regulatory Implementation", "Pandemic", 
    "Asset Runoff", "Reinvestment", "Money Market", "Bond Market", "Equity Markets", "Financial Markets", 
    "Repo Markets", "Fiscal Policy", "Balance Sheet", "Reserves", "Digital Dollar", "Foreign Currencies", 
    "Federal Funds", "Demand", "Securities", "War", "Finance", "Debt", "Mortgage", "Maturity", "Credit", 
    "Labor Market", "Auction", "Press Conference", "Banking System", "Uncertain", "Development", 
    "Economic Outlook", "Countries"
]

def extract_ft_entities(sentences: List[str], threshold: float = 0.5):
    #model_path = "/Users/yuan/Desktop/VisualStudio/Term3/ARP/FineTune_Traditional/arp_fed_llm/MicroF1_0.85"
    
    base_dir = os.path.dirname(__file__) 
    model_path = os.path.join(base_dir, "MicroF1_0.85")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        encodings = tokenizer(sentences, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        logits = model(**encodings).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        for sent, prob_vec in zip(sentences, probs):
            pred_entities = [ENTITY_LIST[i] for i, p in enumerate(prob_vec) if p >= threshold]
            results.append({"sentence": sent, "entities": pred_entities})

    return results
