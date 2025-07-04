from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def _clean_response(raw: str) -> str:
    """
    Strip Markdown fences and extract JSON array for parsing.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return text


def extract_llm_sentiment(items: list) -> list:
    """
    For each item with a sentence and entity names, calls the API to get a sentiment score.
    Quantizes sentiments to steps of approximately 0.33 between -1 and 1.
    Returns the same structure with numeric "sentiment" values.
    Input: list of dicts, each with keys:
        - "sentence": str
        - "entities": list of dicts, each with key "name"
    Output: list of dicts, each with keys:
        - "sentence": str
        - "entities": list of dicts, each with keys "name" and "sentiment"
    """
    results = []
    for item in items:
        sentence = item.get("sentence", "")
        entity_names = [e.get("name") for e in item.get("entities", [])]

        prompt = (
            f"Analyze the sentiment toward each entity in the sentence below. "
            f"Return a JSON array of objects with keys 'name' and 'sentiment', where 'sentiment' "
            f"is a number between -1 and 1 in steps of approximately 0.33 (e.g., -1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0).\n\n"
            f"Sentence: \"{sentence}\"\n"
            f"Entities: {json.dumps(entity_names)}\n"
            f"Example response format:\n"
            f"[{json.dumps({'name':entity_names[0] if entity_names else '', 'sentiment':0.33})}, ...]"
        )

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an assistant that provides numeric sentiment scores in defined steps."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        raw_output = resp.choices[0].message.content
        cleaned = _clean_response(raw_output)
        try:
            sentiments = json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON response: {raw_output}")

        # Merge, convert, and quantize
        merged_entities = []
        step = 1/3
        for ent in sentiments:
            try:
                raw_score = float(ent.get("sentiment"))
                quantized = round(raw_score / step) * step
                quantized = round(quantized, 2)
            except Exception:
                quantized = None
            merged_entities.append({
                "name": ent.get("name"),
                "sentiment": quantized
            })

        results.append({
            "sentence": sentence,
            "entities": merged_entities
        })

    return results

