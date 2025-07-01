#%% Load Data and split train, test
import pandas as pd
import ast

# 讀取資料
df = pd.read_csv("ARP_PreLabel - Dataset Updated.csv")  # 或者用 df = pd.read_excel() 視你的格式而定

# 將字串轉成 Python list，並去除情感分數
def extract_entity_texts(entity_str):
    try:
        entity_list = ast.literal_eval(entity_str)
        return [e[0] for e in entity_list if isinstance(e, tuple) and len(e) > 0]
    except:
        return []

df["Entity_Texts"] = df["Entities"].apply(extract_entity_texts)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#%%
# Flatten all entity lists into one big list
all_entities = [entity for sublist in df["Entity_Texts"] for entity in sublist]
entity_vocab = sorted(set(all_entities))
# 檢查部分結果
print(entity_vocab[:20])  # 頭 20 個實體


#%%
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import ast
import re

#%%
# Load environment and setup client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

tqdm.pandas()

#%%
# Sample few-shot examples from train_df
few_shot = train_df.sample(20, random_state=38)

#%%
# Define prompt builder

def build_prompt(few_shot_df, target_sentence):
    examples = ""
    for _, row in few_shot_df.iterrows():
        examples += f"Sentence: {row['Sentence']}\nEntities: {row['Entity_Texts']}\n\n"

    target = f"Sentence: {target_sentence}\nEntities:"
    return examples + target


#%%
# Extract list-like string from response (e.g. ['Economy', 'Growth'])
def extract_entities_from_response(response_text):
    match = re.search(r"\[.*?\]", response_text)
    if match:
        return match.group(0)
    return "[]"

#%%
# Main prediction function
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


def predict_entities(sentence):
    prompt = build_prompt(few_shot, sentence)
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
        return extract_entities_from_response(content)
    except Exception as e:
        print("Error:", e)
        return "[]"


#%%
# Apply to test set
test_df = test_df.copy()
test_df["LLM_Entities_raw"] = test_df["Sentence"].progress_apply(predict_entities)

# Convert string to Python list
def parse_pred(pred):
    try:
        return ast.literal_eval(pred)
    except:
        return []

test_df["LLM_Entities"] = test_df["LLM_Entities_raw"].progress_apply(parse_pred)

#%%
# Preview result
print(test_df[["Sentence","Entity_Texts", "LLM_Entities_raw", "LLM_Entities"]].head())

#%% calculate the score
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

# 計算平均分數
scores = test_df.apply(lambda row: evaluate_ner(row["Entity_Texts"], row["LLM_Entities"]), axis=1)
precision_avg = scores.map(lambda x: x[0]).mean()
recall_avg = scores.map(lambda x: x[1]).mean()
f1_avg = scores.map(lambda x: x[2]).mean()

print(f"Precision: {precision_avg:.3f}, Recall: {recall_avg:.3f}, F1-score: {f1_avg:.3f}")


# %%
test_df.to_csv("test_predictions.csv", index=False)

# %%
