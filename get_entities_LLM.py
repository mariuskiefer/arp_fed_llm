#%%
import pandas as pd
import ast
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
#%%
'''
# --- Load entity vocabulary from dataset (for prompt consistency) ---
# This block is run at import time to build the entity list for the prompt.
_df = pd.read_csv("ARP_PreLabel - Dataset Updated.csv")
def _extract_entity_texts(entity_str):
    try:
        entity_list = ast.literal_eval(entity_str)
        return [e[0] for e in entity_list if isinstance(e, tuple) and len(e) > 0]
    except:
        return []
_df["Entity_Texts"] = _df["Entities"].apply(_extract_entity_texts)
all_entities = [entity for sublist in _df["Entity_Texts"] for entity in sublist]
entity_vocab = sorted(set(all_entities))
'''

entity_vocab = ["Federal Reserve", "Interest Rates", "Inflation", "Employment", "Unemployment", "GDP", 
                "Trade", "Congress", "Monetary Policy", "Financial Stability", 
                "Price Stability", "Regulatory Implementation", "Pandemic", "Asset Runoff", 
                "Reinvestment", "Money Market", "Bond Market", "Equity Markets", 
                "Financial Markets", "Repo Markets", "Fiscal Policy", "Balance Sheet", 
                "Reserves", "Digital Dollar", "Foreign Currencies", "Federal Funds", "Demand", 
                "Securities", "War", "Finance", "Debt", "Mortgage", "Maturity", "Credit", 
                "Labor Market", "Auction", "Press Conference", "Banking System", "Uncertain", 
                "Development", "Economic Outlook", "Countries"]

#%% --- Setup OpenAI client ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")
'''
# --- Few-shot examples for prompt ---
from sklearn.model_selection import train_test_split
train_df, _ = train_test_split(_df, test_size=0.2, random_state=42)
few_shot = train_df.sample(5, random_state=38)
'''
#%%

# Define few-shot examples as a string
few_shot = """\
Sentence: We didn’t take too much signal out of that.
Entities: []

Sentence: They noted, however, that economic activity and employment were currently well below levels consistent with maximum employment.
Entities: ['Employment', 'GDP']

Sentence: The longer-run projections represented each participant’s assessment of the rate to which each variable would be expected to converge, over time, under appropriate monetary policy and in the absence of further shocks to the economy.
Entities: ['Expectation', 'Monetary Policy', 'Risk']

Sentence: And will you be setting up the process for deciding that sooner, or will you wait until we’re close to the end?
Entities: []

Sentence: As part of its policy decision, the Federal Open Market Committee voted to direct the Open Market Desk at the Federal Reserve Bank of New York, until instructed otherwise, to execute transactions in the System Open Market Account in accordance with the following domestic policy directive: "Effective December 14, 2023, the Federal Open Market Committee directs the Desk to undertake open market operations as necessary to maintain the federal funds rate in a target range of 5-1/4 to 5-1/2 percent.
Entities: ['Federal Reserve', 'Interest Rates', 'Federal Open Market Committee(FOMC)', 'domestic policy directive']
"""

# Define prompt construction function using the few_shot string
def _build_prompt(few_shot, target_sentence):
    target = f"Sentence: {target_sentence}\nEntities:"
    return few_shot.strip() + "\n\n" + target


'''
def _build_prompt(few_shot_df, target_sentence):
    examples = ""
    for _, row in few_shot_df.iterrows():
        examples += f"Sentence: {row['Sentence']}\nEntities: {row['Entity_Texts']}\n\n"
    target = f"Sentence: {target_sentence}\nEntities:"
    return examples + target
'''


def _extract_entities_from_response(response_text):
    match = re.search(r"\[.*?\]", response_text)
    if match:
        return match.group(0)
    return "[]"

system_prompt = (
    "You are a financial named entity recognition (NER) assistant.\n"
    "Your task is to extract only relevant named entities from a given sentence.\n"
    "You MUST select entities exclusively from the provided list.\n"
    "Do not create new entities, synonyms, or paraphrased terms.\n"
    "If no matching entities are found in the sentence, return an empty list.\n"
    "Always return a valid Python list of exact entity strings from the list.\n\n"
    "Respond ONLY with the list. Do NOT explain your reasoning or output any additional text.\n\n"
    "Here is the list of all allowed entities:\n"
    f"{', '.join(entity_vocab)}"
)

def _predict_entities(sentence):
    prompt = _build_prompt(few_shot, sentence)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        content = response.choices[0].message.content.strip()
        return _extract_entities_from_response(content)
    except Exception as e:
        print("Error:", e)
        return "[]"

def _parse_pred(pred):
    try:
        return ast.literal_eval(pred)
    except:
        return []

def extract_llm_entities(sentences):
    """
    Given a list of sentences, returns a list of dicts:
    [{"sentence": <sentence>, "entities": [<entity1>, <entity2>, ...]}, ...]
    """
    results = []
    for sent in sentences:
        raw = _predict_entities(sent)
        entities = _parse_pred(raw)
        results.append({
            "sentence": sent,
            "entities": entities
        })
    return results
