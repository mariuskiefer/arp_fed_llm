# get_sentiment_llm.py
# Zero-shot-ish (one micro few-shot), batched, JSON-only.
# Adds a tiny mid-bin demo to coax ±0.33 / ±0.66 usage without hurting speed.

from openai import OpenAI
import os, json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# --- client setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# --- runtime knobs (accuracy vs speed) ---
MAX_CONCURRENCY   = 4
TOKENS_PER_SLOT   = 12   # slightly more per slot -> better nuance
BASE_MAX_TOKENS   = 260  # small bump
RETRY_TOKEN_BONUS = 240
PER_ITEM_BASE_TOK = 84   # tiny bump to keep single fallback robust
PER_ITEM_BONUS    = 160
TEMP               = 0.42
TOP_P              = 0.95

# --- bin↔score mapping ---
_BIN_TO_SCORE = {-3:-1.0, -2:-0.66, -1:-0.33, 0:0.0, 1:0.33, 2:0.66, 3:1.0}

def _snap_from_number(x):
    try:
        f = float(x)
    except Exception:
        return None
    k = int(round(max(-1.0, min(1.0, f)) * 3))
    return _BIN_TO_SCORE.get(k, 0.0)

def _score_from_bin_or_raw(v, name_nonempty: bool):
    # accept bin int in [-3..3], "" for empty slot, or raw float; coerce ""→0.0 for non-empty names
    if v == "":
        return 0.0 if name_nonempty else ""
    if isinstance(v, int) and v in _BIN_TO_SCORE:
        return _BIN_TO_SCORE[v]
    return _snap_from_number(v)

# --- instruction + one tiny mid-bin demo ---
SYSTEM_MSG = (
    "Aspect-based sentiment per entity.\n"
    "Return JSON only: {\"results\":[{\"i\":<int>, \"bins\":[<bin or \"\">, ...]}, ...]}.\n"
    "Allowed bins: -3,-2,-1,0,1,2,3  (map internally to -1,-0.66,-0.33,0,0.33,0.66,1).\n"
    "Keep the same order/length as input. If entity==\"\", output \"\" at that position.\n"
    "Granularity: prefer -2/-1/1/2 for mild/moderate sentiment; reserve -3/+3 only for clearly strong statements.\n"
    "Economic changes: interest rates↑→negative, ↓→positive; inflation↑→negative, ↓→positive; unemployment↑→negative, ↓→positive; "
    "securities/assets/market value↑→positive, ↓→negative.\n"
    "Observational/monitoring statements → neutral unless explicit valence; "
    "phrases like 'concerns about X' or 'combat/reduce X' → negative for X."
)

# micro few-shot: showcases ±0.33/±0.66 usage in a compact JSON-only example
FEW_SHOT_DEMO = {
    "Example": {
        "i": 9999,
        "s": "Officials noted a slight uptick in inflation and modest progress in the labor market.",
        "e": ["Inflation", "Labor Market", "Monetary Policy", "Price Stability"],
        # bins: Inflation slight uptick -> negative but mild (-1); Labor modest progress -> positive mild (+1)
        # others neutral/mild
        "bins": [-1, 1, 0, 1]
    }
}

# --- small JSON helpers ---
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
    raise ValueError(f"Failed to parse JSON object. Preview:\n{t[:600]}")

# --- LLM calls ---
def _batch_call(batch_payload, max_tokens, try_json_mode_first=True):
    # batch_payload: [{"i": idx, "s": sentence, "e": [names...]}, ...]
    prompt = {
        "Instructions": (
            "For each element in 'Batch', return {\"i\":<i>, \"bins\":[...]}. "
            "'bins' must match the length/order of 'e'. Use only bins -3..3. "
            "Prefer -2/-1/1/2 for moderate sentiment; reserve -3/+3 for clearly strong statements. "
            "For entity==\"\", return \"\" at that slot."
        ),
        "Demo": FEW_SHOT_DEMO,   # single compact mid-bin example
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
    if try_json_mode_first:
        try:
            resp = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(**kwargs)
    else:
        resp = client.chat.completions.create(**kwargs)

    data = _parse_json_obj(resp.choices[0].message.content or "")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Missing 'results' in response. Preview: {json.dumps(data)[:400]}")
    out = {}
    for row in results:
        if not isinstance(row, dict): 
            continue
        i = row.get("i"); bins = row.get("bins")
        if isinstance(i, int) and isinstance(bins, list):
            out[i] = bins
    return out

def _single_call(sentence: str, entities: list, max_tokens: int, try_json_mode_first=True):
    # per-sentence fallback: returns list of bins or "" for empty slots
    prompt = {
        "Instructions": (
            "Return a JSON array of bins with the same length/order as 'Entities'. "
            "Allowed bins: -3,-2,-1,0,1,2,3. Prefer -2/-1/1/2 for moderate sentiment; use -3/+3 only for clearly strong statements. "
            "If entity==\"\", return \"\"."
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
    if try_json_mode_first:
        try:
            resp = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(**kwargs)
    else:
        resp = client.chat.completions.create(**kwargs)

    raw = _strip_fences(resp.choices[0].message.content or "")
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    s = raw.find("["); e = raw.rfind("]")
    if s != -1 and e != -1 and e > s:
        return json.loads(raw[s:e+1])
    raise ValueError(f"Single parse failed. Preview: {raw[:500]}")

# --- public API (unchanged for main.py) ---
def extract_llm_sentiment(items: list) -> list:
    """
    Input:  [{"sentence": str, "entities": [{"name": str}, ...]}, ...]
    Output: [{"sentence": str, "entities": [{"name": str, "sentiment": float|''}, ...]}, ...]
    """
    indexed = []
    for i, it in enumerate(items):
        sent = it.get("sentence", "")
        names = [e.get("name") for e in it.get("entities", [])]
        indexed.append({"i": i, "s": sent, "e": names})
    if not indexed:
        return []

    n = len(indexed)
    splits = max(1, min(MAX_CONCURRENCY, n))
    chunk_size = math.ceil(n / splits)
    chunks = [indexed[i:i + chunk_size] for i in range(0, n, chunk_size)]

    scores_map = {}

    def run_batch(chunk):
        total_slots = sum(len(x["e"]) for x in chunk)
        mt = max(BASE_MAX_TOKENS, TOKENS_PER_SLOT * total_slots)
        return _batch_call(chunk, max_tokens=mt, try_json_mode_first=True)

    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        futs = {ex.submit(run_batch, ch): ch for ch in chunks}
        for fut in as_completed(futs):
            ch = futs[fut]
            try:
                res = fut.result()
                scores_map.update(res)
            except Exception:
                try:
                    total_slots = sum(len(x["e"]) for x in ch)
                    mt_retry = max(BASE_MAX_TOKENS + RETRY_TOKEN_BONUS,
                                   TOKENS_PER_SLOT * total_slots + RETRY_TOKEN_BONUS)
                    res = _batch_call(ch, max_tokens=mt_retry, try_json_mode_first=True)
                    scores_map.update(res)
                except Exception:
                    for x in ch:
                        scores_map.setdefault(x["i"], None)

    # per-item fallback if any index missing
    for x in indexed:
        i = x["i"]
        if i in scores_map and isinstance(scores_map[i], list):
            continue
        names = x["e"]
        try:
            bins = _single_call(x["s"], names, max_tokens=max(PER_ITEM_BASE_TOK, 8 * len(names)))
        except Exception:
            bins = _single_call(x["s"], names, max_tokens=max(PER_ITEM_BASE_TOK + PER_ITEM_BONUS, 12 * len(names)))
        scores_map[i] = bins

    # rebuild final structure and map bins → exact scores
    results = []
    for it_i, it in enumerate(items):
        sent = it.get("sentence", "")
        src_names = [e.get("name") for e in it.get("entities", [])]
        bins = scores_map.get(it_i, [])
        if len(bins) != len(src_names):
            if len(bins) > len(src_names):
                bins = bins[:len(src_names)]
            else:
                bins = bins + [""] * (len(src_names) - len(bins))
        ents = []
        for name, b in zip(src_names, bins):
            if name == "":
                ents.append({"name": "", "sentiment": ""})
            else:
                ents.append({"name": name, "sentiment": _score_from_bin_or_raw(b, name_nonempty=True)})
        results.append({"sentence": sent, "entities": ents})
    return results
