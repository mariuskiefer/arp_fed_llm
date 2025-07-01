#%% Load Data and split train, test
import pandas as pd
import ast

# read data
df = pd.read_csv("ARP_PreLabel - Dataset Updated.csv")

#df = df.dropna(subset=["Sentence", "Entities"])

# change string to Python list, remove sentiment score
def extract_entity_texts(entity_str): 
    try:
        entity_list = ast.literal_eval(entity_str)
        return [e[0] for e in entity_list if isinstance(e, tuple) and len(e) > 0]
    except:
        return []

df["Entity_Texts"] = df["Entities"].apply(extract_entity_texts)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=38)

#%% --- Load Embedding Model ---
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
#sent_model = SentenceTransformer('all-mpnet-base-v2')

#%% --- Custom Entities & Definition ---
CUSTOM_ENTITIES = {
  "Federal Reserve": "The Federal Reserve is the central bank of the United States, responsible for implementing monetary policy, supervising banks, and maintaining financial system stability.",
  "Interest Rates": "Interest Rates are the cost of borrowing money or the return on savings, typically influenced by a central bank to control inflation and stimulate or cool the economy.",
  "Inflation": "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power.",
  "Employment": "Employment refers to the condition where individuals engage in paid work, serving as a key indicator of economic activity.",
  "Unemployment": "Unemployment occurs when individuals who are actively seeking work are unable to find jobs, reflecting slack in the labor market.",
  "GDP": "Gross Domestic Product (GDP) is the total market value of all final goods and services produced within a country over a specific period.",
  "Trade": "Trade is the exchange of goods and services across borders or within regions, influencing balance of payments and economic growth.",
  "Congress": "Congress is the legislative branch of the U.S. government responsible for enacting laws, approving budgets, and overseeing fiscal policy.",
  "Monetary Policy": "Monetary Policy is the process by which a central bank manages money supply and interest rates to achieve macroeconomic objectives like price stability and employment.",
  "Countries": "Countries are sovereign entities with distinct economic, legal, and political systems, often compared in global economic analyses.",
  "Economy/ Economic": "The economy refers to the system of production and consumption activities that determine the allocation of resources and income.",
  "Reinvestment": "Reinvestment is the allocation of profits or returns back into assets, typically used for growth or debt reduction.",
  "Federal Open Market Committee(FOMC)": "The FOMC is a committee within the Federal Reserve that oversees open market operations and sets the federal funds rate.",
  "Regulatory Implementation": "Regulatory Implementation refers to the enactment and enforcement of rules that govern financial institutions and markets.",
  "Financial Stability": "Financial Stability denotes a condition in which the financial system operates efficiently and is resilient to economic shocks.",
  "Price Stability": "Price Stability is a macroeconomic goal aimed at maintaining a low and stable rate of inflation.",
  "Pandemic": "A pandemic is a global disease outbreak that can cause economic disruptions, impacting labor markets, supply chains, and consumption.",
  "Asset Runoff": "Asset Runoff refers to the natural reduction of a central bank's balance sheet as maturing securities are not reinvested.",
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
  "US dollar Depreciation": "US dollar Depreciation means the decline in value of the U.S. dollar relative to other currencies.",
  "US dollar Appreciation": "US dollar Appreciation means the increase in value of the U.S. dollar compared to other currencies.",
  "Federal Funds Target": "The Federal Funds Target is the interest rate range set by the FOMC to guide overnight interbank lending.",
  "Federal Funds Rate": "The Federal Funds Rate is the interest rate at which depository institutions lend reserves to each other overnight.",
  "Term Premium": "Term Premium refers to the extra yield investors require for holding longer-term bonds relative to short-term ones.",
  "Demand": "Demand is the quantity of a good or service consumers are willing and able to purchase at various prices.",
  "Personal Consumption Expenditures (PCE)": "PCE is a measure of consumer spending and a key inflation gauge closely watched by the Federal Reserve.",
  "Energy Prices": "Energy Prices refer to the cost of energy commodities like oil, gas, and electricity, impacting inflation and production costs.",
  "Market Liquidity": "Market Liquidity is the ease with which assets can be bought or sold without affecting their price.",
  "Expectation": "Expectation refers to beliefs about future economic or financial outcomes, influencing investment and consumption decisions.",
  "Securities": "Securities are financial instruments that represent ownership (stocks), debt (bonds), or rights to ownership.",
  "Growth": "Growth usually refers to economic growth, typically measured by the increase in real GDP over time.",
  "Yields": "Yields are the earnings generated from investments, commonly expressed as a percentage of the investment’s cost or market value.",
  "Overnight Reverse Repurchase Agreement (ON RRP)": "ON RRP is a monetary policy tool allowing eligible institutions to invest funds overnight with the Federal Reserve.",
  "Overnight Rate": "The Overnight Rate is the interest rate charged on loans between financial institutions for one day.",
  "Interest on Excess Reserves (IOER)": "IOER is the interest paid by the Federal Reserve on reserve balances held by depository institutions.",
  "Repurchase Agreement (repo) Rates": "Repo Rates are the interest rates applied in repurchase agreements where securities are used as collateral.",
  "Equity Price": "Equity Price refers to the market value of a company’s shares traded on a stock exchange.",
  "Cyclically Sensitive Sectors": "These are industries whose performance is highly dependent on the business cycle, such as housing or manufacturing.",
  "War": "War refers to armed conflict that can disrupt economies, markets, trade, and fiscal stability.",
  "Risk": "Risk is the potential for financial loss or uncertainty in investment outcomes.",
  "Finance": "Finance is the management of money, including investing, borrowing, lending, budgeting, and forecasting.",
  "Debt": "Debt is the obligation to repay borrowed money, typically with interest, used by governments and firms to fund operations.",
  "Mortgage": "A Mortgage is a loan secured by real estate, commonly used to purchase homes.",
  "domestic policy directive": "A domestic policy directive guides the operations of the Federal Reserve System in implementing monetary policy.",
  "SOMA": "SOMA stands for System Open Market Account, the Fed’s portfolio used to implement monetary policy.",
  "Maturity": "Maturity is the date on which a financial instrument's principal is due to be repaid.",
  "Credit": "Credit is the ability to borrow money or access goods and services with the agreement to pay later.",
  "Dual Mandate": "The Federal Reserve’s dual mandate is to achieve maximum employment and stable prices.",
  "Labor Market": "The Labor Market represents the supply and demand for labor, influencing wages and employment levels.",
  "repurchase agreement": "A repurchase agreement (repo) is a short-term loan where securities are sold and later repurchased at a higher price.",
  "auction": "An auction is a market process used to sell government securities to investors at competitive rates.",
  "Asset": "An Asset is a resource with economic value owned by an individual, company, or government.",
  "Average Hourly Earnings": "Average Hourly Earnings track changes in the compensation received by workers, used as a wage growth indicator.",
  "Balance": "Balance refers to the net position in a financial account or the equilibrium between inflows and outflows.",
  "Board": "The Board refers to the Federal Reserve Board of Governors, which oversees the Federal Reserve System.",
  "Chair": "The Chair is the head of the Federal Reserve Board of Governors, currently Jerome Powell (as of 2025).",
  "Powell": "Jerome Powell is the Chair of the Federal Reserve, guiding U.S. monetary policy decisions.",
  "Press Conference": "A Press Conference is a formal media event where Fed officials communicate policy decisions and outlook.",
  "maximum employment": "Maximum employment is the highest level of employment the economy can sustain without causing inflation.",
  "Economic outlook": "Economic outlook refers to the forecasted future condition of the economy based on key indicators.",
  "Security holdings": "Security holdings are financial instruments held by institutions or investors as assets.",
  "Treasury Securities": "Treasury Securities are government debt instruments issued by the U.S. Department of the Treasury.",
  "Mortgage-Backed Securities": "Mortgage-Backed Securities are investment products backed by a pool of home loans.",
  "Banking System": "The Banking System consists of institutions that accept deposits and make loans, forming the backbone of financial intermediation.",
  "HIRE": "HIRE likely refers to hiring activity, a measure of labor market strength and economic momentum.",
  "Uncertain": "Uncertain refers to economic conditions that are unpredictable or volatile, affecting confidence and investment.",
  "Development": "Development refers to progress in economic or institutional capacity, particularly in emerging markets.",
  "Transitory": "Transitory is a term used to describe temporary economic phenomena, often in reference to inflation surges."
}

entities_embeddings = {
    entity: sent_model.encode(desc, convert_to_tensor=True)
    for entity, desc in CUSTOM_ENTITIES.items()
}
#%% --- Entity Matching ---
def match_entities(sentence):
    sent_emb = sent_model.encode(sentence, convert_to_tensor=True)
    similarities = {
        entity: util.cos_sim(sent_emb, emb).item()
        for entity, emb in entities_embeddings.items()
    }

    threshold = 0.35
    filtered_entities = {
        entity: sim for entity, sim in similarities.items() if sim >= threshold
    }

    top_entities = sorted(filtered_entities.items(), key=lambda x: x[1], reverse=True)[:5]

    return [
        {"entity": lbl, "similarity": round(sim, 3)} for lbl, sim in top_entities
    ]

#%% --- Main Pipeline ---
import spacy
from collections import defaultdict

def process_all_rows_grouped(df, column_name):
    nlp = spacy.load("en_core_web_sm")
    grouped_results = defaultdict(list)

    for idx, text in enumerate(df[column_name]):
        doc = nlp(text)
        for sent in doc.sents:
            matched = match_entities(sent.text)  # return list of dicts
            if matched:
                for item in matched:
                    grouped_results[idx].append(item["entity"])  # extract entity name
    '''
    # 
    result_list = []
    for i in range(len(df)):
        unique_entities = list(set(grouped_results[i]))
        result_list.append(unique_entities)
    return result_list
    '''
    # return list
    result_list = []
    for i in range(len(df)):
        result_list.append(grouped_results[i])
    return result_list

#%% --- Example Usage ---
test_df["matched_entities"] = process_all_rows_grouped(test_df, "Sentence")
print(test_df[["Sentence", "matched_entities"]].head())


#%% evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_ner(true_entities, pred_entities):
    true_set = set(true_entities)
    pred_set = set(pred_entities)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
    return precision, recall, f1

# calculate average score
scores = test_df.apply(lambda row: evaluate_ner(row["Entity_Texts"], row["matched_entities"]), axis=1)
precision_avg = scores.map(lambda x: x[0]).mean()
recall_avg = scores.map(lambda x: x[1]).mean()
f1_avg = scores.map(lambda x: x[2]).mean()

print(f"Precision: {precision_avg:.3f}, Recall: {recall_avg:.3f}, F1-score: {f1_avg:.3f}")

# %%
test_df.to_csv("Definition_outcome.csv",index=False)
# %%
