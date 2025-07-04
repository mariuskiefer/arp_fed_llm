import pandas as pd
import ast
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

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

# --- Setup OpenAI client ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

# --- Few-shot examples for prompt ---
from sklearn.model_selection import train_test_split
train_df, _ = train_test_split(_df, test_size=0.2, random_state=42)
few_shot = train_df.sample(5, random_state=38)

def _build_prompt(few_shot_df, target_sentence):
    examples = ""
    for _, row in few_shot_df.iterrows():
        examples += f"Sentence: {row['Sentence']}\nEntities: {row['Entity_Texts']}\n\n"
    target = f"Sentence: {target_sentence}\nEntities:"
    return examples + target

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
