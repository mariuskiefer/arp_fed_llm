# Output format: [{"sentence": str, "entities": [str, ...]}, ...]
# Note: Requires python-dotenv and openai packages.

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os, re, ast, time

# --- Configurable parameters for speed and reliability ---
MAX_CONCURRENCY = 12   # Number of threads for parallel API calls
TEMP = 0.0             # Set to 0 for deterministic results
TOP_P = 0.9
MAX_TOKENS = 96        # Should be enough for a short JSON list
RETRY_ONCE = True
RETRY_BACKOFF_SEC = 0.8

# --- Entity vocabulary (must match allowed labels) ---
entity_vocab = [
    'Federal Reserve', 'Interest Rates', 'Inflation', 'Employment', 'Unemployment', 'GDP',
    'Trade', 'Congress', 'Monetary Policy', 'Financial Stability', 'Price Stability',
    'Regulatory Implementation', 'Pandemic', 'Asset Runoff', 'Reinvestment', 'Money Market',
    'Bond Market', 'Equity Markets', 'Financial Markets', 'Repo Markets', 'Fiscal Policy',
    'Balance Sheet', 'Reserves', 'Digital Dollar', 'Foreign Currencies', 'Federal Funds',
    'Demand', 'Securities', 'War', 'Finance', 'Debt', 'Mortgage', 'Maturity', 'Credit',
    'Labor Market', 'Auction', 'Press Conference', 'Banking System', 'Uncertain',
    'Development', 'Economic Outlook', 'Countries'
]
_VOCAB_SET = set(entity_vocab)

# --- Set up OpenAI/DeepSeek client ---
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com")

# --- Prompt for the LLM (includes instructions and examples) ---
_SYSTEM_PROMPT = (
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
    f"Here is the list of all allowed entities:\n{', '.join(entity_vocab)}\n\n"
    "Here are examples to guide you:\n\n"
    "Sentence: We didn't take too much signal out of that.\n"
    "Entities: []\n\n"
    "Sentence: The Federal Reserve raised interest rates to combat inflation, but the move sparked concerns about economic growth.\n"
    "Entities: ['Federal Reserve', 'Interest Rates', 'Inflation', 'Monetary Policy', 'Economic Outlook', 'GDP', 'Federal Funds']\n\n"
    "Sentence: The virus and the measures taken to protect public health are inducing sharp declines in economic activity and a surge in job losses.\n"
    "Entities: ['Employment', 'Unemployment', 'Economic Outlook', 'Pandemic']\n\n"
    "Sentence: They noted, however, that economic activity and employment were currently well below levels consistent with maximum employment.\n"
    "Entities: ['Employment', 'GDP']\n\n"
    "Sentence: The invasion and related events are creating additional upward pressure on inflation and are likely to weigh on economic activity.\n"
    "Entities: ['Inflation', 'Economic Outlook', 'War']\n\n"
    "Sentence: And will you be setting up the process for deciding that sooner, or will you wait until we're close to the end?\n"
    "Entities: []\n\n"
    "Sentence: The Committee will closely monitor market conditions and is prepared to adjust its plans as appropriate.\n"
    "Entities: ['Federal Reserve', 'Monetary Policy', 'Economic Outlook']\n\n"
    "Sentence: Markets reacted positively after the Federal Reserve indicated a pause in its tightening cycle.\n"
    "Entities: ['Federal Reserve', 'Monetary Policy', 'Financial Markets', 'Economic Outlook']\n\n"
    "Sentence: The central bank’s bond purchases were aimed at supporting market liquidity.\n"
    "Entities: ['Federal Reserve', 'Monetary Policy', 'Financial Stability']\n\n"
    "Sentence: Quantitative easing has helped keep borrowing costs low and stabilize the economy.\n"
    "Entities: ['Monetary Policy', 'Financial Stability', 'GDP']\n\n"
    "Sentence: After multiple rounds of tightening, the markets are bracing for the Fed’s next move.\n"
    "Entities: ['Monetary Policy', 'Financial Markets', 'Federal Reserve']\n"
)

def _user_prompt(sentence: str) -> str:
    # User prompt is kept minimal to save tokens
    return f"Sentence: {sentence}\nEntities:"

# --- Helper for extracting the first bracketed list from LLM output ---
_BRACKETS_RE = re.compile(r"\[.*?\]", re.S)

def _extract_entities_from_response(text: str):
    # Try to find the first list in the response, fallback to empty list if not found
    if not text:
        return "[]"
    m = _BRACKETS_RE.search(text)
    return m.group(0) if m else "[]"

def _parse_list(pred: str):
    # Safely parse a string like "['A','B']" into a Python list
    try:
        arr = ast.literal_eval(pred)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

def _filter_to_vocab(unique_list):
    # Remove duplicates and keep only valid vocab entities, preserving order
    seen = set()
    out = []
    for x in unique_list:
        if x in _VOCAB_SET and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# --- Core LLM call (single sentence, no retry) ---
def _predict_once(sentence: str) -> list:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(sentence)}
        ],
        temperature=TEMP,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stream=False
    )
    content = (resp.choices[0].message.content or "").strip()
    raw_list = _extract_entities_from_response(content)
    return _filter_to_vocab(_parse_list(raw_list))

# --- Retry logic for robustness (one retry on failure) ---
def _predict_with_retry(sentence: str) -> list:
    try:
        return _predict_once(sentence)
    except Exception:
        if not RETRY_ONCE:
            return []
        time.sleep(RETRY_BACKOFF_SEC)
        try:
            return _predict_once(sentence)
        except Exception:
            return []

# --- Main API: batch sentences, deduplicate, parallelize, preserve order ---
def extract_llm_entities(sentences):
    """
    Args:
        sentences: List[str]
    Returns:
        List[{"sentence": <sentence>, "entities": [<entity>, ...]}]
    """
    if not sentences:
        return []

    # Deduplicate sentences for efficiency (avoid redundant API calls)
    unique = []
    idx_map = {}
    for i, s in enumerate(sentences):
        if s not in idx_map:
            idx_map[s] = len(unique)
            unique.append(s)

    cache = [None] * len(unique)

    def _worker(j, s):
        ents = _predict_with_retry(s)
        cache[j] = ents

    workers = min(MAX_CONCURRENCY, max(1, len(unique)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, idx_map[s], s) for s in idx_map.keys()]
        for _ in as_completed(futures):
            pass  # All results are written to cache

    # Restore original order and structure
    out = []
    for s in sentences:
        ents = cache[idx_map[s]] or []
        out.append({"sentence": s, "entities": ents})
    return out
