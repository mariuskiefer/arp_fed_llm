import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
import spacy
from collections import defaultdict

'''
# read data
# df = pd.read_csv("ARP_PreLabel.csv")

# change string to Python list, remove sentiment score
def extract_entity_texts(entity_str): 
    try:
        entity_list = ast.literal_eval(entity_str)
        return [e[0] for e in entity_list if isinstance(e, tuple) and len(e) > 0]
    except:
        return []

df["Entity_Texts"] = df["Entities"].apply(extract_entity_texts)

#train_df, test_df = train_test_split(df, test_size=0.2, random_state=38)
'''

# --- Load Embedding Model ---
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Custom Entities & Definition ---
CUSTOM_ENTITIES = {
   "Interest Rates": "Interest Rates are the cost of borrowing money or the return on savings, typically influenced by a central bank to control inflation and stimulate or cool the economy.",
  "Inflation": "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power.",
  "Employment": "Employment refers to the condition where individuals engage in paid work, serving as a key indicator of economic activity.",
  "Unemployment": "Unemployment occurs when individuals who are actively seeking work are unable to find jobs, reflecting slack in the labor market.",
  "GDP": "Gross Domestic Product (GDP) is the total market value of all final goods and services produced within a country over a specific period.",
  "Trade": "Trade is the exchange of goods and services across borders or within regions, influencing balance of payments and economic growth.",
  "Congress": "Congress is the legislative branch of the U.S. government responsible for enacting laws, approving budgets, and overseeing fiscal policy.",
  "Monetary Policy": "Monetary Policy is the process by which a central bank manages money supply and interest rates to achieve macroeconomic objectives like price stability and employment.",
  "Financial Stability": "Financial Stability denotes a condition in which the financial system operates efficiently and is resilient to economic shocks.",
  "Price Stability": "Price Stability is a macroeconomic goal aimed at maintaining a low and stable rate of inflation.",
  "Regulatory Implementation": "Regulatory Implementation refers to the enactment and enforcement of rules that govern financial institutions and markets.",
  "Pandemic": "A pandemic is a global disease outbreak that can cause economic disruptions, impacting labor markets, supply chains, and consumption.",
  "Asset Runoff": "Asset Runoff refers to the natural reduction of a central bank's balance sheet as maturing securities are not reinvested.",
  "Reinvestment": "Reinvestment is the allocation of profits or returns back into assets, typically used for growth or debt reduction.",
  "Money Market": "The Money Market is a segment of the financial market for short-term borrowing and lending, typically under one year.",
  "Bond Market": "The Bond Market is where investors trade debt securities, such as government and corporate bonds, to finance operations or projects.",
  "Equity Markets": "Equity Markets are venues where shares of publicly held companies are issued and traded.",
  "Financial Markets": "Financial Markets are broad arenas where individuals and institutions trade financial securities and derivatives.",
  "Repo Markets": "Repo Markets involve repurchase agreements where securities are sold with the promise to repurchase them, used for short-term funding.",
  "Fiscal Policy": "Fiscal Policy involves government spending and taxation to influence economic conditions.",
  "Balance Sheet": "A Balance Sheet is a financial statement summarizing an entity’s assets, liabilities, and equity at a given point in time.",
  "Reserves": "Reserves are funds held by banks to meet regulatory requirements or settle transactions, often stored at a central bank.",
  "Digital Dollar": "Digital Dollar refers to a proposed central bank digital currency (CBDC) issued by the Federal Reserve.",
  "Foreign Currencies": "Foreign Currencies are legal tenders issued by countries other than one’s own, traded in foreign exchange markets.",
  "Federal Funds": "Federal Funds are excess reserve balances held by banks at the Federal Reserve, traded in the federal funds market.",
  "Demand": "Demand is the quantity of a good or service consumers are willing and able to purchase at various prices.",
  "Securities": "Securities are financial instruments that represent ownership (stocks), debt (bonds), or rights to ownership.",
  "War": "War refers to armed conflict that can disrupt economies, markets, trade, and fiscal stability.",
  "Finance": "Finance is the management of money, including investing, borrowing, lending, budgeting, and forecasting.",
  "Debt": "Debt is the obligation to repay borrowed money, typically with interest, used by governments and firms to fund operations.",
  "Mortgage": "A Mortgage is a loan secured by real estate, commonly used to purchase homes.",
  "Maturity": "Maturity is the date on which a financial instrument's principal is due to be repaid.",
  "Credit": "Credit is the ability to borrow money or access goods and services with the agreement to pay later.",
  "Labor Market": "The Labor Market represents the supply and demand for labor, influencing wages and employment levels.",
  "Auction": "An auction is a market process used to sell government securities to investors at competitive rates.",
  "Press Conference": "A Press Conference is a formal media event where Fed officials communicate policy decisions and outlook.",
  "Banking System": "The Banking System consists of institutions that accept deposits and make loans, forming the backbone of financial intermediation.",
  "Uncertain": "Uncertain refers to economic conditions that are unpredictable or volatile, affecting confidence and investment.",
  "Development": "Development refers to progress in economic or institutional capacity, particularly in emerging markets.",
  "Economic Outlook": "Economic outlook refers to the forecasted future condition of the economy based on key indicators.",
  "Countries": "Countries are sovereign entities with distinct economic, legal, and political systems, often compared in global economic analyses."
}

entities_embeddings = {
    entity: sent_model.encode(desc, convert_to_tensor=True)
    for entity, desc in CUSTOM_ENTITIES.items()
}

def match_entities(sentence):
    sent_emb = sent_model.encode(sentence, convert_to_tensor=True)
    similarities = {
        entity: util.cos_sim(sent_emb, emb).item()
        for entity, emb in entities_embeddings.items()
    }

    threshold = 0.30
    filtered_entities = {
        entity: sim for entity, sim in similarities.items() if sim >= threshold
    }

    top_entities = sorted(filtered_entities.items(), key=lambda x: x[1], reverse=True)[:5]

    return [
        {"entity": lbl, "similarity": round(sim, 3)} for lbl, sim in top_entities
    ]

def extract_nlp_entities(sentences):
    """
    Given a list of sentences, returns a list of dicts:
    [{"sentence": <sentence>, "entities": [<entity1>, <entity2>, ...]}, ...]
    """
    nlp = spacy.load("en_core_web_sm")
    results = []
    for sent in sentences:
        doc = nlp(sent)
        matched_entities = []
        for s in doc.sents:
            matched = match_entities(s.text)
            if matched:
                matched_entities.extend([item["entity"] for item in matched])
        results.append({
            "sentence": sent,
            "entities": matched_entities
        })
    return results
