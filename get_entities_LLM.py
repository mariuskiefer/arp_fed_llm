#%%
import pandas as pd
import ast
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
#%%
entity_vocab = ['Federal Reserve', 'Interest Rates', 'Inflation', 'Employment', 'Unemployment', 'GDP', 
                'Trade', 'Congress', 'Monetary Policy', 'Financial Stability', 
                'Price Stability', 'Regulatory Implementation', 'Pandemic', 'Asset Runoff', 
                'Reinvestment', 'Money Market', 'Bond Market', 'Equity Markets', 
                'Financial Markets', 'Repo Markets', 'Fiscal Policy', 'Balance Sheet', 
                'Reserves', 'Digital Dollar', 'Foreign Currencies', 'Federal Funds', 'Demand', 
                'Securities', 'War', 'Finance', 'Debt', 'Mortgage', 'Maturity', 'Credit', 
                'Labor Market', 'Auction', 'Press Conference', 'Banking System', 'Uncertain', 
                'Development', 'Economic Outlook', 'Countries']

#%% --- Setup OpenAI client ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

#%%
# Define few-shot examples as a string
few_shot = """\
Sentence: We didn't take too much signal out of that.
Entities: []

Sentence: The Federal Reserve raised interest rates to combat inflation, but the move sparked concerns about economic growth.
Entities: ['Federal Reserve', 'Interest Rates', 'Inflation', 'Monetary Policy', 'Economic Outlook', 'GDP', 'Federal Funds']

Sentence: The virus and the measures taken to protect public health are inducing sharp declines in economic activity and a surge in job losses.
Entities: ['Employment', 'Unemployment', 'Economic Outlook', 'Pandemic']

Sentence: They noted, however, that economic activity and employment were currently well below levels consistent with maximum employment.
Entities: ['Employment', 'GDP']

Sentence: The invasion and related events are creating additional upward pressure on inflation and are likely to weigh on economic activity.
Entities: ['Inflation', 'Economic Outlook', 'War']

Sentence: And will you be setting up the process for deciding that sooner, or will you wait until we're close to the end?
Entities: []

Sentence: The Committee will closely monitor market conditions and is prepared to adjust its plans as appropriate.
Entities: ['Federal Reserve', 'Monetary Policy', 'Economic Outlook']

Sentence: Markets reacted positively after the Federal Reserve indicated a pause in its tightening cycle.
Entities: ['Federal Reserve', 'Monetary Policy', 'Financial Markets', 'Economic Outlook']

Sentence: The central bank’s bond purchases were aimed at supporting market liquidity.
Entities: ['Federal Reserve', 'Monetary Policy', 'Financial Stability']

Sentence: Quantitative easing has helped keep borrowing costs low and stabilize the economy.
Entities: ['Monetary Policy', 'Financial Stability', 'GDP']

Sentence: After multiple rounds of tightening, the markets are bracing for the Fed’s next move.
Entities: ['Monetary Policy', 'Financial Markets', 'Federal Reserve']

"""

# Define prompt construction function using the few_shot string
def _build_prompt(few_shot, target_sentence):
    target = f"Sentence: {target_sentence}\nEntities:"
    return few_shot.strip() + "\n\n" + target


def _extract_entities_from_response(response_text):
    match = re.search(r"\[.*?\]", response_text)
    if match:
        return match.group(0)
    return "[]"
'''
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

system_prompt = (
    "You are a financial named entity recognition (NER) assistant.\n"
    "Your task is to extract only relevant named entities from a given sentence.\n"
    "You MUST select entities exclusively from the provided list.\n"
    "These entities may be explicitly mentioned or implicitly implied through financial context, technical terminology, or paraphrased references.\n"
    "Be sensitive to indirect mentions — if a sentence clearly refers to an entity's function, purpose, or impact, extract it.\n"
    "Do not invent entities beyond the list.\n"
    "If no matching entities are found, return an empty list.\n"
    "Always return a valid Python list of exact entity strings from the list.\n\n"
    "Respond ONLY with the list. Do NOT explain your reasoning or output any additional text.\n\n"
    f"Here is the list of all allowed entities:\n{', '.join(entity_vocab)}"
)
'''

system_prompt = (
    "You are a financial named entity recognition (NER) assistant.\n"
    "Your task is to extract all relevant financial entities from a given sentence.\n"
    "You MUST select entities exclusively from the provided list.\n"
    "These entities may be:\n"
    "- Explicitly mentioned (e.g., 'Federal Reserve', 'GDP')\n"
    "- Implied through financial terminology (e.g., 'tightening cycle' → 'Monetary Policy')\n"
    "- Referred to through indirect expressions related to their function, purpose, or economic impact\n\n"
    "Be attentive to subtle language. For example:\n"
    "- 'Quantitative easing' or 'bond purchases' should imply 'Monetary Policy'\n"
    "- 'Market reactions', 'liquidity conditions' may imply 'Financial Markets' or 'Financial Stability'\n"
    "- 'Minutes from the meeting' or 'FOMC decision' may suggest 'Press Conference' or 'Monetary Policy'\n\n"
    "IMPORTANT:\n"
    "- Do NOT invent new entities or paraphrase\n"
    "- Only include entities from the list below\n"
    "- If no matching entity is present, return an empty list\n"
    "- If an entity is mentioned multiple times, include it only once\n\n"
    "Respond ONLY with the list. Do NOT explain your reasoning or output any additional text.\n\n"
    f"Here is the list of all allowed entities:\n{', '.join(entity_vocab)}"
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

# %%
