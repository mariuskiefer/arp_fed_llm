import os, json, math, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- runtime knobs ---
MAX_CONCURRENCY   = 4
TOKENS_PER_SLOT   = 12
BASE_MAX_TOKENS   = 260
RETRY_TOKEN_BONUS = 240
PER_ITEM_BASE_TOK = 84
PER_ITEM_BONUS    = 160
TEMP              = 0.75
TOP_P             = 0.90

_BIN_TO_SCORE = {-3:-1.0, -2:-0.66, -1:-0.33, 0:0.0, 1:0.33, 2:0.66, 3:1.0}

def _snap_from_number(x):
    try:
        f = float(x)
    except Exception:
        return None
    k = int(round(max(-1.0, min(1.0, f)) * 3))
    return _BIN_TO_SCORE.get(k, 0.0)

def _score_from_bin_or_raw(v, name_nonempty: bool):
    if v == "":
        return 0.0 if name_nonempty else ""
    if isinstance(v, int) and v in _BIN_TO_SCORE:
        return _BIN_TO_SCORE[v]
    return _snap_from_number(v)

# --- instruction + demo ---
SYSTEM_MSG = (
    "You are analyzing Federal Reserve press conference transcripts for aspect-based sentiment analysis.\n"
    "\n"
    "SENTIMENT SCALE: Use integer bins from -3 to 3 for each entity.\n"
    "• -3: Very negative → maps to -1.0 (major concerns, significant deterioration, severe problems)\n"
    "• -2: Negative → maps to -0.66 (concerns, decline, weakness, challenges)\n" 
    "• -1: Slightly negative → maps to -0.33 (mild concerns, modest decline, some weakness)\n"
    "• 0: Neutral → maps to 0.0 (monitoring, observational, balanced, no clear direction)\n"
    "• 1: Slightly positive → maps to 0.33 (modest improvement, some progress, mild optimism)\n"
    "• 2: Positive → maps to 0.66 (improvement, progress, strength, confidence)\n"
    "• 3: Very positive → maps to 1.0 (significant improvement, strong performance, major progress)\n"
    "\n"
    "TARGET DISTRIBUTION: In Federal Reserve contexts, expect roughly:\n"
    "• 10-15% extreme ratings (±3): Use for 'major', 'significant', 'substantial' language\n"
    "• 20-25% strong ratings (±2): Use for clear directional statements\n"
    "• 30-35% mild ratings (±1): Use for 'modest', 'some', 'slight' language\n"
    "• 15-25% neutral (0): Only for pure monitoring/observational statements\n"
    "\n"
    "FINANCIAL CONTEXT - DIRECTIONAL SENTIMENT:\n"
    "• Interest rates UP = negative (-1 to -3), DOWN = positive (+1 to +3)\n"
    "• Inflation UP = negative (-1 to -3), DOWN = positive (+1 to +3)\n"
    "• Unemployment UP = negative (-1 to -3), DOWN = positive (+1 to +3)\n"
    "• Markets/GDP UP = positive (+1 to +3), DOWN = negative (-1 to -3)\n"
    "\n"
    "MANDATORY TRIGGERS FOR EXTREME RATINGS:\n"
    "• MUST use -3: 'severe', 'major concerns', 'significant deterioration', 'substantial decline'\n"
    "• MUST use -2: 'concerns', 'decline', 'weakness', 'challenges', 'risks'\n"
    "• MUST use +2: 'improvement', 'progress', 'strength', 'confidence', 'gains'\n"
    "• MUST use +3: 'significant improvement', 'strong performance', 'robust growth', 'substantial progress'\n"
    "\n"
    "CRITICAL: Federal Reserve language is often understated. 'Modest' concerns may actually be -2. 'Some' progress may be +2.\n"
    "Don't be conservative - use the full range or your analysis will be useless.\n"
    "\n"
    "OUTPUT: Return JSON only in format {\"results\":[{\"i\":<int>, \"bins\":[...]}, ...]}.\n"
    "Maintain exact order and length as input entities."
)

FEW_SHOT_DEMO = {
    "Extreme_Example": {
        "i": 9999,
        "s": "The Committee expressed major concerns about persistent inflationary pressures while acknowledging robust employment gains and substantial economic recovery.",
        "e": ["Inflation", "Employment", "Economic Recovery", "Monetary Policy"],
        "bins": [-3, 3, 2, -2]
    },
    "Moderate_Example": {
        "i": 9998,
        "s": "Officials noted some decline in inflation expectations and modest improvement in labor market conditions.",
        "e": ["Inflation", "Labor Market"],
        "bins": [2, 1]
    },
    "Neutral_Example": {
        "i": 9997,
        "s": "The Committee will continue to monitor economic developments.",
        "e": ["Economic Outlook"],
        "bins": [0]
    }
}

# --- small helpers ---
def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

def _parse_json_obj(text: str) -> dict:
    t = _strip_fences(text)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    s = t.find("{"); e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(t[s:e+1])
    return {}

# --- LLM calls ---
def _batch_call(batch_payload, max_tokens=1500):
    prompt = {
        "Instructions": (
            "For each element in 'Batch', return {\"i\":<i>, \"bins\":[...]}. "
            "'bins' must match the length/order of 'e'. Allowed bins: -3..3."
        ),
        "Demo": FEW_SHOT_DEMO,
        "Batch": batch_payload
    }
    kwargs = dict(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ],
        temperature=TEMP,
        top_p=TOP_P,
        max_tokens=max_tokens,
        stream=False,
    )
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        resp = client.chat.completions.create(**kwargs)

    raw = getattr(resp.choices[0].message, "content", "") or ""
    data = _parse_json_obj(raw)

    out = {}
    results = data.get("results", [])
    for row in results:
        if isinstance(row, dict):
            i = row.get("i")
            bins = row.get("bins")
            if isinstance(i, int) and isinstance(bins, list):
                out[i] = bins
    return out


def _single_call(sentence: str, entities: list, max_tokens=1500):
    prompt = {
        "Instructions": (
            "Return a JSON array of bins with the same length/order as 'Entities'. "
            "Allowed bins: -3,-2,-1,0,1,2,3. If entity==\"\", return \"\"."
        ),
        "Demo": FEW_SHOT_DEMO,
        "Sentence": sentence,
        "Entities": entities
    }
    kwargs = dict(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Output JSON array only; no prose."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ],
        temperature=TEMP,
        top_p=TOP_P,
        max_tokens=max_tokens,
        stream=False,
    )
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        resp = client.chat.completions.create(**kwargs)

    raw = getattr(resp.choices[0].message, "content", "") or ""
    raw = _strip_fences(raw)

    if not raw.strip():
        return ["" for _ in entities]

    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    s = raw.find("["); e = raw.rfind("]")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(raw[s:e+1])
        except Exception:
            return ["" for _ in entities]

    return ["" for _ in entities]


# --- public API ---
def extract_llm_sentiment(items: list) -> list:
    indexed = []
    for i, it in enumerate(items):
        sent = it.get("sentence", "")
        names = [e.get("name") for e in it.get("entities", [])]
        if not sent or len(sent.strip()) < 5:
            continue
        indexed.append({"i": i, "s": sent, "e": names})

    if not indexed:
        return []

    logger.info(f"[extract] received items={len(items)} | with_entities={len(indexed)} | skipped={len(items)-len(indexed)}")

    # fixed batch size = 20
    chunk_size = 20
    chunks = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]
    logger.info(f"[extract] batching | chunks={len(chunks)} | chunk_size≈{chunk_size}")

    scores_map = {}

    def run_batch(chunk):
        return _batch_call(chunk, max_tokens=1500)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        futs = {ex.submit(run_batch, ch): ch for ch in chunks}
        for fut in as_completed(futs):
            ch = futs[fut]
            try:
                res = fut.result()
                scores_map.update(res)
            except Exception as e:
                logger.info(f"[extract] batch failed; falling back per-item: {e}")
                for x in ch:
                    try:
                        bins = _single_call(x["s"], x["e"], max_tokens=1500)
                    except Exception:
                        bins = ["" for _ in x["e"]]
                    scores_map[x["i"]] = bins

    # reassemble results
    results = []
    for it_i, it in enumerate(items):
        sent = it.get("sentence", "")
        src_names = [e.get("name") for e in it.get("entities", [])]
        bins = scores_map.get(it_i, [""] * len(src_names))
        if len(bins) != len(src_names):
            bins = bins[:len(src_names)] + [""] * (len(src_names) - len(bins))
        ents = []
        for name, b in zip(src_names, bins):
            if name == "":
                ents.append({"name": "", "sentiment": ""})
            else:
                ents.append({"name": name, "sentiment": _score_from_bin_or_raw(b, name_nonempty=True)})
        results.append({"sentence": sent, "entities": ents})

    # dump to JSON file
    out_path = os.getenv("SENTIMENT_OUTPUT_FILE", "sentiment_output.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"[extract] wrote output to {out_path}")
    except Exception as e:
        logger.warning(f"[extract] could not write output: {e}")

    return results
