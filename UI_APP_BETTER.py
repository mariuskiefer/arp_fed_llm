#---------------------------------------------------------------------------------------------------------------
import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# PDF extraction (use PyMuPDF for robust spacing)
# pip install pymupdf
import fitz  # PyMuPDF

# Optional segmentation library
# pip install pysbd
try:
    import pysbd
    _PYSBD_SEGMENTER = pysbd.Segmenter(language="en", clean=True)  # you can try clean=True
except Exception:
    _PYSBD_SEGMENTER = None

# ------------------------------
# Load API key from .env
# ------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

# ------------------------------
# Import ALL backend functions
# ------------------------------
from get_sentiment_llm import extract_llm_sentiment
from get_sentiment_nlp import extract_nlp_sentiment
from get_entities_LLM import extract_llm_entities
from get_entities_nlp import extract_nlp_entities
from get_entities_ft_nlp import extract_ft_entities


# ------------------------------
# PDF reader (PyMuPDF)
# ------------------------------
def read_pdf(file):
    """Extract text from an uploaded PDF file (robust spacing) using PyMuPDF."""
    file.seek(0)
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)


# ------------------------------
# Text normalization
# ------------------------------
def normalize_pdf_text(text: str) -> str:
    """Normalize raw PDF text for sentence segmentation."""
    # Remove soft hyphen and weird spaces
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\xa0", " ")   # non-breaking space

    # Fix hyphenation at line breaks: "inflation-\nadjusted" -> "inflation-adjusted"
    text = re.sub(r"-\s*\n\s*", "-", text)

    # Collapse line breaks inside paragraphs: single \n becomes space; preserve double \n as paragraph break
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # single newline -> space
    text = re.sub(r"\n{2,}", "\n\n", text)         # 2+ newlines -> exactly two

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    # Normalize odd Unicode spaces to regular space; drop zero-width marks
    text = re.sub(r"[\u2009\u200a\u202f\u200b\u2060]", " ", text)  # thin/narrow/ZW spaces -> space
    text = re.sub(r"[\u200c\u200d\uFEFF]", "", text)               # ZWNJ/ZWJ/BOM -> remove

    # De-hyphenate layout-induced word breaks: 'tremen-dous' -> 'tremendous' (letters only)
    text = re.sub(r"(?i)(?<=[A-Za-z])-\s*(?=[A-Za-z])", "", text)

    return text.strip()


# ------------------------------
# Sentence segmentation helpers
# ------------------------------
# Abbreviations to protect (extend as needed)
_ABBREV_PAT = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Co|vs|No|Art|Fig|cf|e\.g|i\.e|U\.S|U\.K|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.',
    flags=re.IGNORECASE
)

# Split on sentence-ending punctuation followed by optional quotes/brackets and whitespace
# End of sentence:
# 1) [.!?] + optional closing quote/bracket, or closing quote + [.!?], then whitespace
# 2) OR a colon that is immediately followed by an opening quote (e.g., … p.m.: “The …)
_SENT_END_PAT = re.compile(
    r'(?:[.!?][\"\')\]\}\u2019\u201d\u201f\u00bb»]?|[\"\')\]\}\u2019\u201d\u201f\u00bb»][.!?])\s+'
    r'|:(?=\s*[\"\u201c])\s+'
)


# Lightweight probe for potential sentence ends, independent of quotes
_EOS_PROBE = re.compile(r'[.!?](?:\s+|$)')


# A rare placeholder char to protect dots in abbreviations
_PROTECT_DOT = "\u2E3A"  # placeholder for '.'

def _protect_abbrev(text: str) -> str:
    """Replace dots in known abbreviations so they won't trigger sentence split."""
    return _ABBREV_PAT.sub(lambda m: m.group(0).replace('.', _PROTECT_DOT), text)

def _restore_abbrev(text: str) -> str:
    """Restore protected dots in abbreviations."""
    return text.replace(_PROTECT_DOT, '.')

# --- Enforce boundaries after period if a directive verb follows (robust to weird spaces) ---
# verbs/phrases that start FOMC directives
_DIRECTIVE_VERBS = ["Undertake", "Conduct", "Roll over", "Reinvest", "Redeem", "Allow"]

def _loose_phrase(phrase: str) -> str:
    """Build a regex that tolerates stray spaces inside words and between tokens."""
    tokens = phrase.split()
    token_parts = []
    for t in tokens:
        # allow spaces between letters: R\s*e\s*d\s*e\s*e\s*m
        chars = "".join(re.escape(ch) + r"\s*" for ch in t)
        token_parts.append(chars)
    # allow spaces between tokens: Roll\s+over
    return r"\s+".join(token_parts)

_VERBS_LOOSE = "|".join(_loose_phrase(v) for v in _DIRECTIVE_VERBS)
_DIRECTIVE_SPLIT_RE = re.compile(rf"(?<=[.!?])\s+(?=(?:{_VERBS_LOOSE})\b)", flags=re.IGNORECASE)

def enforce_directive_boundaries(text: str) -> str:
    """Insert a newline after .?! if a directive verb/phrase follows (handles weird spaces)."""
    return _DIRECTIVE_SPLIT_RE.sub("\n", text)

def _regex_split_sentences(text: str) -> list[str]:
    """Pure-regex splitter with abbreviation protection."""
    t = _protect_abbrev(text)
    parts, start = [], 0
    for m in _SENT_END_PAT.finditer(t):
        end = m.end()
        parts.append(t[start:end].strip())
        start = end
    if start < len(t):
        parts.append(t[start:].strip())
    # restore abbreviations
    return [_restore_abbrev(s) for s in parts if s]

def split_into_sentences(text: str, min_len: int = 10) -> list[str]:
    """
    Robust sentence splitter:
    1) normalize PDF text
    2) enforce boundaries after .?! when a directive verb follows (e.g., 'Redeem', 'Roll over')
    3) run both pysbd (if available) and regex; choose the one that yields MORE sentences
    4) clean very short fragments and page numbers
    """
    # 1) Normalize PDF artifacts
    text = normalize_pdf_text(text)
    # 2) Ensure '. Redeem/Undertake/...' etc. becomes a hard boundary
    text = enforce_directive_boundaries(text)

    if not text:
        return []

    # --- run both engines ---
    s_pysbd = []
    if _PYSBD_SEGMENTER is not None:
        try:
            s_pysbd = _PYSBD_SEGMENTER.segment(text)
        except Exception:
            s_pysbd = []

    s_regex = _regex_split_sentences(text)

    # choose the more fine-grained result
    sents = s_pysbd if len(s_pysbd) >= len(s_regex) else s_regex

    # 4) Cleanup
    clean = []
    for s in sents:
        s = s.strip()
        if not s or len(s) < min_len:
            continue
        # filter lone page numbers like "12" or "(12)"
        if re.fullmatch(r"\(?\d{1,3}\)?", s):
            continue
        clean.append(s)
    return clean


# ------------------------------
# Bullet/list-aware second-stage splitter
# ------------------------------
_BULLET_CHARS = r"•◦○●\-–—o·"  # extend if needed
# markers like '(1) ', '(a) ', '1) ', 'a) ' at line/boundary/after colon/semicolon
_ENUM_MARKER = r"(?:\(\d+\)|\(\w\)|\d+\)|[a-zA-Z]\))"

def _split_on_bullets(text: str) -> list[str]:
    """
    Split on common bullet markers that typically indicate new instruction lines.
    Boundary can be start, newline, semicolon, colon, or period.
    """
    boundary = r"(?:^|[\n\r;:\.])"  # include period .
    pattern = rf"{boundary}\s*(?:[{_BULLET_CHARS}]|{_ENUM_MARKER})\s+"
    parts = re.split(pattern, text)
    return [p for p in parts if p is not None]

def _inject_newlines_for_bullets(text: str) -> str:
    """
    Force a newline before list markers when they appear right after a period (or closing quote).
    Turns '. o ' / '." o ' / '.” o ' into '.\n o ' so the bullet-splitter can see boundaries.
    """
    enum = _ENUM_MARKER
    bullets = _BULLET_CHARS
    pattern = rf'[\.\"\u201d]\s+(?=(?:[{bullets}]|{enum})\s+)'
    return re.sub(pattern, lambda m: m.group(0).rstrip() + "\n", text)

def further_split(sentences: list[str], min_len: int = 10) -> list[str]:
    """Apply bullet/list splitting to each sentence and clean up."""
    new_sents = []
    for s in sentences:
        # Make '. o ' turn into a visible boundary for splitting
        s = _inject_newlines_for_bullets(s)

        # Split on bullets/enumerations
        chunks = _split_on_bullets(s)
        for c in chunks:
            c = c.strip().strip('"\'')

            # --- SAFE normalisations (do NOT remove normal spaces) ---
            # Join hyphenated tokens and numeric ranges: 'per -counterparty', '4 -1/4' -> 'per-counterparty', '4-1/4'
            c = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", c)

            # Remove space before punctuation: 'percent .' -> 'percent.'
            c = re.sub(r"\s+([,.;:!?])", r"\1", c)

            # Fix times: '2: 00 p.m.' -> '2:00 p.m.'
            c = re.sub(r"(\d)\s*:\s*(\d{2})", r"\1:\2", c)

            # Fix 'a. m.' / 'p. m.' -> 'a.m.' / 'p.m.'
            c = re.sub(r"\b([ap])\.\s*m\.", r"\1.m.", c, flags=re.IGNORECASE)

            # Filters
            if len(c) < min_len:
                continue
            if re.fullmatch(r"\(?\d{1,3}\)?", c):
                continue
            new_sents.append(c)
    return new_sents


# ------------------------------
# Sentiment label mapper
# ------------------------------
def sentiment_label(score: float) -> str:
    """Map sentiment score (-1 to 1) into human-readable label."""
    if score <= -0.66: return "Very Negative"
    elif score <= -0.33: return "Medium Negative"
    elif score < 0: return "Slightly Negative"
    elif score == 0: return "Neutral"
    elif score < 0.33: return "Slightly Positive"
    elif score < 0.66: return "Medium Positive"
    else: return "Very Positive"


# ------------------------------
# Method selectors
# ------------------------------
def get_entity_extraction_function(method_name):
    """Return the appropriate entity extraction function based on user selection."""
    if method_name == "NLP":
        return extract_nlp_entities, "nlp"
    elif method_name == "Fine-tuned NLP":
        return extract_ft_entities, "ftnlp"
    else:  # LLM (default)
        return extract_llm_entities, "llm"

def get_sentiment_analysis_function(method_name):
    """Return the appropriate sentiment analysis function based on user selection."""
    if method_name == "NLP":
        return extract_nlp_sentiment, "nlp"
    else:  # LLM (default)
        return extract_llm_sentiment, "llm"


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Fed Transcript Analyzer", page_icon="📊", layout="wide")

st.title("📊 Federal Reserve Transcript Analyzer")
st.write("Upload a Federal Reserve transcript PDF and get instant insights powered by LLMs and NLP.")

# Method Selection Section
st.subheader("🎛️ Choose Your Analysis Methods")

col1, col2 = st.columns(2)

with col1:
    entity_method = st.selectbox(
        "**Entity Extraction Method**",
        options=["LLM", "NLP", "Fine-tuned NLP"],
        index=0,  # Default to LLM
        help="Choose your preferred entity extraction approach"
    )

with col2:
    sentiment_method = st.selectbox(
        "**Sentiment Analysis Method**",
        options=["LLM", "NLP"],
        index=0,  # Default to LLM
        help="Choose your preferred sentiment analysis approach"
    )

# Display selected methods
st.info(f"🔧 **Selected Configuration:** {entity_method} Entity Extraction + {sentiment_method} Sentiment Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = read_pdf(uploaded_file)

    # Split into sentences (primary with pysbd, fallback regex) + bullet-aware further split
    sentences = split_into_sentences(pdf_text, min_len=10)
    sentences = further_split(sentences, min_len=10)

    # --- Debug: show sentence splitting result ---
    with st.expander("🛠 Sentence Split Debug (click to expand)"):
        st.write(f"Total sentences: {len(sentences)}")
        debug_df = pd.DataFrame({
            "idx": range(len(sentences)),
            "char_len": [len(s) for s in sentences],
            "preview": [s[:120] + ("..." if len(s) > 120 else "") for s in sentences],
        })
        st.dataframe(debug_df, use_container_width=True)

        sel = st.number_input("Inspect sentence index", min_value=0, max_value=max(0, len(sentences)-1), value=0, step=1)
        if len(sentences) > 0:
            # Use plain text to avoid syntax highlighting
            st.text(sentences[sel])

    # Get selected functions
    entity_func, entity_method_used = get_entity_extraction_function(entity_method)
    sentiment_func, sentiment_method_used = get_sentiment_analysis_function(sentiment_method)

    # Run entity extraction with timing
    with st.spinner(f"Running entity extraction with {entity_method}..."):
        start_time = time.time()
        entity_results = entity_func(sentences)
        entity_time = time.time() - start_time

    # Prepare pipeline for sentiment analysis
    pipeline = []
    for item in entity_results:
        ents = [{"name": name} for name in item["entities"]]
        pipeline.append({"sentence": item["sentence"], "entities": ents})

    # Run sentiment analysis with timing
    with st.spinner(f"Running sentiment analysis with {sentiment_method}..."):
        start_time = time.time()
        if sentiment_method_used == "nlp":
            # NLP sentiment expects different input format
            nlp_input = [(item["sentence"], [ent["name"] for ent in item["entities"]]) for item in pipeline]
            results = sentiment_func(nlp_input)
        else:
            # LLM sentiment uses the pipeline format
            results = sentiment_func(pipeline)
        sentiment_time = time.time() - start_time

    # Performance Metrics
    st.subheader("⚡ Performance Metrics")
    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:
        st.metric(
            label="Entity Extraction",
            value=f"{entity_time:.2f}s",
            help=f"Time taken using {entity_method} method"
        )

    with perf_col2:
        st.metric(
            label="Sentiment Analysis",
            value=f"{sentiment_time:.2f}s",
            help=f"Time taken using {sentiment_method} method"
        )

    with perf_col3:
        st.metric(
            label="Total Processing",
            value=f"{entity_time + sentiment_time:.2f}s"
        )

    # --- Insights (summary, entities, overall sentiment) ---
    overall_text = " ".join(sentences[:50])
    try:
        summary_prompt = f"""
        You are a senior financial analyst specializing in monetary policy and Federal Reserve communications.
        Analyze the following transcript carefully and provide a concise 2–3 sentence summary.

        Requirements:
        - Capture the deeper meaning and policy implications, not just surface details.
        - Highlight the Fed's tone, stance, and any signals about inflation, rates, or growth.
        - Write for traders, economists, and finance creators who want time-saving insights.
        - Ensure the summary preserves the economic meaning of the full transcript.

        Transcript:
        {overall_text}
        """
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=150
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = "⚠️ Summary fallback: This transcript mainly discusses inflation and monetary policy with cautious tone."
        st.warning(f"Summary fallback used: {e}")

    st.subheader("📌 Summary")
    st.write(summary)

    # Top 5 entities with sentiment
    all_entities = {}
    for item in results:
        for ent in item["entities"]:
            name = ent["name"]
            score = ent["sentiment"]
            if name not in all_entities:
                all_entities[name] = []
            all_entities[name].append(score)

    if all_entities:
        entity_avg = {k: np.mean(v) for k, v in all_entities.items()}
        top_entities = sorted(entity_avg.items(), key=lambda x: len(all_entities[x[0]]), reverse=True)[:5]

        st.subheader("🏆 Top 5 Entities")
        for name, avg_score in top_entities:
            st.write(f"**{name}** → {sentiment_label(avg_score)} ({avg_score:.2f})")
    else:
        st.info("No entities detected in this PDF.")

    # Overall sentiment
    all_scores = [ent["sentiment"] for item in results for ent in item["entities"]]
    if all_scores:
        overall_score = np.mean(all_scores)
        st.subheader("📊 Overall Sentiment")
        st.write(f"{sentiment_label(overall_score)} ({overall_score:.2f})")
    else:
        st.subheader("📊 Overall Sentiment")
        st.write("Neutral (0.00) — no sentiment detected.")
