#%%
import os
from get_sentiment_llm import extract_llm_sentiment
from get_sentiment_nlp import extract_nlp_sentiment
from get_entities_LLM import extract_llm_entities  
from get_entities_nlp import extract_nlp_entities
from get_entities_ft_nlp import extract_ft_entities


# Main function that combines sentence retrieval, entity extraction and sentiment analysis

if __name__ == "__main__":
    # Example input: just a list of sentences
    sentences = [
        "The Federal Reserve raised interest rates to combat inflation, but the move sparked concerns about economic growth.",
        "Federal Reserve officials signaled that further rate hikes may be necessary if inflation remains elevated.",
        "The Federal Reserve announced a new round of quantitative easing to support the economy.",
        "Federal Reserve Chair Jerome Powell emphasized the importance of price stability in his latest speech.",
        "The Federal Reserve is closely monitoring labor market conditions as it considers future policy moves.",
        "Markets reacted positively after the Federal Reserve indicated a pause in its tightening cycle.",
        "The Federal Reserve's decision to cut rates was met with mixed reactions from economists.",
        "Federal Reserve policymakers expressed uncertainty about the outlook for economic growth.",
        "The Federal Reserve released minutes from its latest meeting, highlighting concerns about persistent inflation.",
        "Investors are watching the Federal Reserve for signals about the timing of the next interest rate change."
    ]

    method = input("Which entity extraction method do you want to use? (llm/nlp): ").strip().lower()
    if method == "nlp":
        entity_results = extract_nlp_entities(sentences)
    elif method == "ftnlp":
        entity_results = extract_ft_entities(sentences)
    else:
        entity_results = extract_llm_entities(sentences)
    # entity_results: [{"sentence": ..., "entities": [...]}, ...]

    # Step 2: Prepare input for sentiment assignment
    pipeline = []
    for item in entity_results:
        ents = [{"name": name} for name in item["entities"]]
        pipeline.append({
            "sentence": item["sentence"],
            "entities": ents
        })

    # Step 3: Assign sentiment to each entity
    sentiment_method = input("Which sentiment analysis method do you want to use? (llm/nlp): ").strip().lower()
    if sentiment_method == "nlp":
        results = extract_nlp_sentiment([(item["sentence"], [ent["name"] for ent in item["entities"]]) for item in pipeline])
    else:
        results = extract_llm_sentiment(pipeline)

    # Step 4: Print results
    for item in results:
        print(f"Sentence: {item['sentence']}")
        print(f"Entities: {[ent['name'] for ent in item['entities']]}")
        for ent in item["entities"]:
            print(f"  Entity: {ent['name']}, Sentiment: {ent['sentiment']}")
        print()
# %%
