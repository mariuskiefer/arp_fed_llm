from api import assign_sentiment_to_entities
from get_entities_LLM import extract_llm_entities  

# Main function that combines sentence retrieval, entity extraction and sentiment analysis

if __name__ == "__main__":
    # Example input: just a list of sentences
    sentences = [
        "The risks of higher unemployment and higher inflation appear to have risen, and we believe that the current stance of monetary policy leaves us well positioned to respond in a timely way to potential economic developments.",
        "The Federal Reserve raised interest rates to combat inflation, but the move sparked concerns about economic growth."
    ]

    # Step 1: Extract entities for each sentence
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
    results = assign_sentiment_to_entities(pipeline)

    # Step 4: Print results
    for item in results:
        print(f"Sentence: {item['sentence']}")
        print(f"Entities: {[ent['name'] for ent in item['entities']]}")
        for ent in item["entities"]:
            print(f"  Entity: {ent['name']}, Sentiment: {ent['sentiment']}")
        print()