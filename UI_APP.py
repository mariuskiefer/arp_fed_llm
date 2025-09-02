# To run: use "streamlit run UI_APP.py" in your terminal.

# ===============================================================================================================
import os
import re
import time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI

# PDF text extraction using PyMuPDF (fitz)
import fitz

# Try to import sentence segmentation (pysbd), fallback if not available
try:
    import pysbd
    _PYSBD_SEGMENTER = pysbd.Segmenter(language="en", clean=True)
except Exception:
    _PYSBD_SEGMENTER = None

# Try to import rapidfuzz for fuzzy canonicalization, fallback if not available
try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Load OpenAI/DeepSeek API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

# Import my backend functions for entity and sentiment extraction
from get_sentiment_llm import extract_llm_sentiment
from get_sentiment_nlp import extract_nlp_sentiment
from get_entities_LLM import extract_llm_entities
from get_entities_ft_nlp import extract_ft_entities   # "NLP" option uses this

# =========================
#         I/O
# =========================
def read_pdf(file):
    file.seek(0)
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join([p.get_text("text") for p in doc])

# =========================
#   Text normalization and sentence splitting
# =========================
def normalize_pdf_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = re.sub(r"-\s*\n\s*", "-", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[\u2009\u200a\u202f\u200b\u2060]", " ", text)
    text = re.sub(r"[\u200c\u200d\uFEFF]", "", text)
    text = re.sub(r"(?i)(?<=[A-Za-z])-\s*(?=[A-Za-z])", "", text)
    return text.strip()

_ABBREV_PAT = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|Co|vs|No|Art|Fig|cf|e\.g|i\.e|U\.S|U\.K|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.',
    flags=re.IGNORECASE
)
_SENT_END_PAT = re.compile(
    r'(?:[.!?][\"\')\]\}\u2019\u201d\u201f\u00bb»]?|[\"\')\]\}\u2019\u201d\u201f\u00bb»][.!?])\s+'
    r'|:(?=\s*[\"\u201c])\s+'
)
_PROTECT_DOT = "\u2E3A"

def _protect_abbrev(text: str) -> str:
    return _ABBREV_PAT.sub(lambda m: m.group(0).replace('.', _PROTECT_DOT), text)

def _restore_abbrev(text: str) -> str:
    return text.replace(_PROTECT_DOT, '.')

_DIRECTIVE_VERBS = ["Undertake", "Conduct", "Roll over", "Reinvest", "Redeem", "Allow"]

def _loose_phrase(phrase: str) -> str:
    tokens = phrase.split()
    token_parts = []
    for t in tokens:
        chars = "".join(re.escape(ch) + r"\s*" for ch in t)
        token_parts.append(chars)
    return r"\s+".join(token_parts)

_VERBS_LOOSE = "|".join(_loose_phrase(v) for v in _DIRECTIVE_VERBS)
_DIRECTIVE_SPLIT_RE = re.compile(rf"(?<=[.!?])\s+(?=(?:{_VERBS_LOOSE})\b)", flags=re.IGNORECASE)

def enforce_directive_boundaries(text: str) -> str:
    return _DIRECTIVE_SPLIT_RE.sub("\n", text)

def _regex_split_sentences(text: str) -> list[str]:
    t = _protect_abbrev(text)
    parts, start = [], 0
    for m in _SENT_END_PAT.finditer(t):
        end = m.end()
        parts.append(t[start:end].strip())
        start = end
    if start < len(t):
        parts.append(t[start:].strip())
    return [_restore_abbrev(s) for s in parts if s]

def split_into_sentences(text: str, min_len: int = 10) -> list[str]:
    text = normalize_pdf_text(text)
    text = enforce_directive_boundaries(text)
    if not text:
        return []
    s_pysbd = []
    if _PYSBD_SEGMENTER is not None:
        try:
            s_pysbd = _PYSBD_SEGMENTER.segment(text)
        except Exception:
            s_pysbd = []
    s_regex = _regex_split_sentences(text)
    sents = s_pysbd if len(s_pysbd) >= len(s_regex) else s_regex
    clean = []
    for s in sents:
        s = s.strip()
        if not s or len(s) < min_len:
            continue
        if re.fullmatch(r"\(?\d{1,3}\)?", s):
            continue
        clean.append(s)
    return clean

_BULLET_CHARS = r"•◦○●\-–—o·"
_ENUM_MARKER = r"(?:\(\d+\)|\(\w\)|\d+\)|[a-zA-Z]\))"

def _split_on_bullets(text: str) -> list[str]:
    boundary = r"(?:^|[\n\r;:\.])"
    pattern = rf"{boundary}\s*(?:[{_BULLET_CHARS}]|{_ENUM_MARKER})\s+"
    parts = re.split(pattern, text)
    return [p for p in parts if p is not None]

def _inject_newlines_for_bullets(text: str) -> str:
    enum = _ENUM_MARKER
    bullets = _BULLET_CHARS
    pattern = rf'[\.\"\u201d]\s+(?=(?:[{bullets}]|{enum})\s+)'
    return re.sub(pattern, lambda m: m.group(0).rstrip() + "\n", text)

def further_split(sentences: list[str], min_len: int = 10) -> list[str]:
    new_sents = []
    for s in sentences:
        s = _inject_newlines_for_bullets(s)
        chunks = _split_on_bullets(s)
        for c in chunks:
            c = c.strip().strip('"\'')

            c = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", c)
            c = re.sub(r"\s+([,.;:!?])", r"\1", c)
            c = re.sub(r"(\d)\s*:\s*(\d{2})", r"\1:\2", c)
            c = re.sub(r"\b([ap])\.\s*m\.", r"\1.m.", c, flags=re.IGNORECASE)

            if len(c) < min_len:
                continue
            if re.fullmatch(r"\(?\d{1,3}\)?", c):
                continue
            new_sents.append(c)
    return new_sents

# =========================
#   Sentiment bins and method selection
# =========================
def sentiment_label(score: float) -> str:
    if score <= -0.9: return "Extremely Negative"
    elif score <= -0.6: return "Strongly Negative"
    elif score <= -0.3: return "Slightly Negative"
    elif score == 0:   return "Neutral"
    elif score <= 0.4: return "Slightly Positive"
    elif score <= 0.7: return "Strongly Positive"
    else:               return "Extremely Positive"

SENTIMENT_BINS = [
    "Extremely Negative",
    "Strongly Negative",
    "Slightly Negative",
    "Neutral",
    "Slightly Positive",
    "Strongly Positive",
    "Extremely Positive",
]

def get_entity_extraction_function(method_name):
    # "NLP" option uses fine-tuned entity extraction
    if method_name == "NLP":
        return extract_ft_entities, "ftnlp"
    else:
        return extract_llm_entities, "llm"

def get_sentiment_analysis_function(method_name):
    if method_name == "NLP":
        return extract_nlp_sentiment, "nlp"
    else:
        return extract_llm_sentiment, "llm"

# =========================
#   Canonicalization (optional, for entity names)
# =========================
CANON = {"Federal Reserve": ["Fed", "FOMC", "Federal Reserve Board"],
         "Treasury": ["UST", "US Treasury", "Treasuries"]}
ALIAS2CANON = {a: k for k, aliases in CANON.items() for a in aliases + [k]}

def canonicalize(name: str, score_cut: int = 90) -> str:
    if not _HAS_RAPIDFUZZ:
        return ALIAS2CANON.get(name, name)
    if name in ALIAS2CANON:
        return ALIAS2CANON[name]
    cand, sc, _ = process.extractOne(name, list(ALIAS2CANON.keys()), scorer=fuzz.WRatio)
    return ALIAS2CANON[cand] if sc >= score_cut else name

# =========================
#   Helper functions for building tables and stats
# =========================
def all_entities_tables_csv(entity_counts, overall_counts, entity_avg=None):
    frames = []
    for name, counts in entity_counts.items():
        df = pd.DataFrame({
            "Entities": [name] + [""] * (len(SENTIMENT_BINS) - 1),
            "Sentiment": SENTIMENT_BINS,
            "Frequency": [counts[b] for b in SENTIMENT_BINS],
        })
        if entity_avg and name in entity_avg:
            avg = round(float(entity_avg[name]), 6)
            df["Average Sentiment Score"] = [""] * len(df)
            df.loc[df.index[0], "Average Sentiment Score"] = avg
        frames.append(df)

    odf = pd.DataFrame({
        "Entities": ["Overall"] + [""] * (len(SENTIMENT_BINS) - 1),
        "Sentiment": SENTIMENT_BINS,
        "Frequency": [overall_counts[b] for b in SENTIMENT_BINS],
        "Average Sentiment Score": [""] * len(SENTIMENT_BINS)
    })
    frames.append(odf)
    return pd.concat(frames, ignore_index=True)

def build_pipeline(entity_results):
    return [{"sentence": item.get("sentence", ""),
             "entities": [{"name": name} for name in item.get("entities", [])]}
            for item in entity_results]

def build_entity_counts(results):
    entity_counts, entity_scores = {}, {}
    for item in results:
        for ent in item["entities"]:
            ent["name"] = canonicalize(ent["name"])
            name = ent["name"]
            sc = float(ent["sentiment"])
            bin_name = sentiment_label(sc)
            if name not in entity_counts:
                entity_counts[name] = {b: 0 for b in SENTIMENT_BINS}
                entity_scores[name] = []
            entity_counts[name][bin_name] += 1
            entity_scores[name].append(sc)
    overall_counts = {b: 0 for b in SENTIMENT_BINS}
    for cnts in entity_counts.values():
        for b, v in cnts.items():
            overall_counts[b] += v
    entity_totals = {k: sum(v.values()) for k, v in entity_counts.items()}
    entity_avg = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in entity_scores.items()}
    return entity_counts, overall_counts, entity_totals, entity_avg

def dfs_for_top_entities(entity_counts, entity_totals, top_n=5, min_count=1):
    top_entities = [(n, c) for n, c in entity_totals.items() if c >= min_count]
    top_entities = sorted(top_entities, key=lambda x: x[1], reverse=True)[:top_n]
    tables = []
    for name, _ in top_entities:
        counts = entity_counts[name]
        df = pd.DataFrame({
            "Sentiment": SENTIMENT_BINS,
            "Frequency": [counts[b] for b in SENTIMENT_BINS],
        })
        tables.append((name, df))
    return tables

def overall_distribution_df(overall_counts):
    return pd.DataFrame({
        "Sentiment": SENTIMENT_BINS,
        "Frequency": [overall_counts[b] for b in SENTIMENT_BINS],
    })

# =========================
#   Table styling for Streamlit (hide index, center Frequency)
# =========================
def style_sent_freq_center(df, sent_width="18ch", freq_width="8ch"):
    df = df.reset_index(drop=True)
    return (df.style
        .hide(axis="index")
        .set_properties(subset=["Sentiment"], **{
            "white-space": "normal",
            "overflow-wrap": "anywhere",
            "vertical-align": "middle",
        })
        .set_properties(subset=["Frequency"], **{
            "text-align": "center",
            "vertical-align": "middle",
        })
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "th.row_heading, td.row_heading, th.blank.level0",
             "props": [("display", "none")]},
            {"selector": ".col0", "props": [("width", sent_width)]},
            {"selector": ".col1", "props": [("width", freq_width)]},
        ])
    )

# =========================
#   Altair chart for overall sentiment distribution
# =========================
def chart_overall_distribution(overall_counts):
    dfc = pd.DataFrame({"Sentiment": list(overall_counts.keys()),
                        "Frequency": list(overall_counts.values())})
    return alt.Chart(dfc).mark_bar().encode(
        x=alt.X('Frequency:Q', title='Count'),
        y=alt.Y('Sentiment:N', sort=list(reversed(SENTIMENT_BINS)), title=''),
        tooltip=['Sentiment','Frequency']
    ).properties(height=260)

# =========================
#   Get sample sentences for a given entity and sentiment bucket
# =========================
def sample_sentences(results, entity, bucket, topn=3):
    out=[]
    for item in results:
        for e in item["entities"]:
            if e["name"]==entity and sentiment_label(float(e["sentiment"]))==bucket:
                out.append(item["sentence"])
    return out[:topn]

# =========================
#   Caching for heavy backend calls (pure functions)
# =========================
@st.cache_data(show_spinner=False)
def cached_entities(sentences, method_name):
    func, _ = get_entity_extraction_function(method_name)
    return func(sentences)

@st.cache_data(show_spinner=False)
def cached_sentiment(pipeline, method_name):
    func, tag = get_sentiment_analysis_function(method_name)
    if tag == "nlp":
        nlp_input = [(x["sentence"], [e["name"] for e in x["entities"]]) for x in pipeline]
        return func(nlp_input)
    return func(pipeline)

# =========================
#   Streamlit UI
# =========================
st.set_page_config(page_title="Fed Transcript Analyzer", page_icon="📊", layout="wide")
st.title("📊 Federal Reserve Transcript Analyzer")
st.write("Upload a Fed-related transcript PDF and get analyst-grade insights (entities, sentiment distributions, and more).")

# Session state for per-file analysis and file key
if "analysis" not in st.session_state:
    st.session_state.analysis = {}
if "file_key" not in st.session_state:
    st.session_state.file_key = None

with st.sidebar:
    st.subheader("⚙️ Settings")
    entity_method = st.selectbox("Entity Extraction Method", options=["LLM", "NLP"], index=0)
    sentiment_method = st.selectbox("Sentiment Analysis Method", options=["LLM", "NLP"], index=0)
    top_k = st.slider("Top-K entities to show", 3, 12, 5)
    reset = st.button("🔄 Reset analysis for this file",
                      help="Clear cached results and rerun heavy steps for the current file using the selected methods.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Generate a stable key for the uploaded file (hash of content)
    with st.spinner("Extracting text from PDF..."):
        pdf_text = read_pdf(uploaded_file)
    sentences = further_split(split_into_sentences(pdf_text, min_len=10), min_len=10)
    doc_hash = hashlib.md5((" ".join(sentences[:200])).encode("utf-8")).hexdigest()
    file_key = f"{uploaded_file.name}|{getattr(uploaded_file, 'size', 'NA')}|{doc_hash}"
    
    with st.expander("🛠 Sentence Split Debug (click to expand)"):
        st.write(f"Total sentences: {len(sentences)}")
        debug_df = pd.DataFrame({
            "idx": range(len(sentences)),
            "char_len": [len(s) for s in sentences],
            "preview": [s[:120] + ("..." if len(s) > 120 else "") for s in sentences],
        })
        st.dataframe(debug_df, use_container_width=True)
        if len(sentences) > 0:
            sel = st.number_input("Inspect sentence index", min_value=0, max_value=len(sentences)-1, value=0, step=1)
            st.text(sentences[sel])    

    # Reset analysis state if file changes or user clicks reset
    if st.session_state.file_key != file_key or reset:
        st.session_state.file_key = file_key
        st.session_state.analysis = {
            "entity_results": None,
            "results": None,
            "summary_text": None,
            "t_entity": None,
            "t_sent": None,
            "t_summ": None,
            "entity_method_used": entity_method,
            "sentiment_method_used": sentiment_method,
        }

    # Short alias for session state
    A = st.session_state.analysis

    # Entity extraction (only runs once per file)
    if A["entity_results"] is None:
        with st.spinner(f"Running entity extraction with {entity_method}..."):
            t0 = time.time()
            entity_results = cached_entities(sentences, entity_method)
            A["t_entity"] = time.time() - t0
            A["entity_results"] = entity_results
            A["entity_method_used"] = entity_method

    # Sentiment analysis (only runs once per file)
    if A["results"] is None:
        with st.spinner(f"Running sentiment analysis with {sentiment_method}..."):
            pipeline = build_pipeline(A["entity_results"])
            t0 = time.time()
            results = cached_sentiment(pipeline, sentiment_method)
            A["t_sent"] = time.time() - t0
            A["results"] = results
            A["sentiment_method_used"] = sentiment_method

    # Executive summary (LLM, only runs once per file)
    if A["summary_text"] is None:
        with st.spinner("Running Executive Summary..."):
            summary_text = ""
            t0 = time.time()
            try:
                overall_text = " ".join(sentences[:200])
                summary_prompt = f"""
                You are a senior financial analyst specializing in monetary policy and Federal Reserve communications.
                Analyze the following transcript carefully and provide a concise 2–3 sentence summary.

                Requirements:
                - Capture the deeper meaning and policy implications, not just surface details.
                - Highlight the Fed's tone, stance, and any signals about inflation, rates, growth or securities.
                - Write for traders, economists, and finance creators who want time-saving insights.
                - Ensure the summary preserves the economic meaning of the full transcript.

                Transcript:
                {overall_text}
                """
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=180,
                    temperature=0
                )
                summary_text = resp.choices[0].message.content.strip()
            except Exception as e:
                summary_text = "⚠️ Summary fallback: This transcript mainly discusses inflation and monetary policy with cautious tone."
                st.warning(f"Summary fallback used: {e}")
            A["t_summ"] = time.time() - t0
            A["summary_text"] = summary_text

    # Show performance metrics (timings for each step)
    st.subheader("⚡ Performance Metrics (Total includes Summary)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entity Extraction", f"{A['t_entity']:.2f}s", help=f"Method used: {A['entity_method_used']}")
    c2.metric("Sentiment Analysis", f"{A['t_sent']:.2f}s", help=f"Method used: {A['sentiment_method_used']}")
    c3.metric("Summary (LLM)", f"{A['t_summ']:.2f}s", help="Time to generate the executive summary")
    total_time = A['t_entity'] + A['t_sent'] + A['t_summ']
    c4.metric("Total Processing", f"{total_time:.2f}s")

    # Build stats from cached results
    entity_counts, overall_counts, entity_totals, entity_avg = build_entity_counts(A["results"])
    all_scores = [float(ent["sentiment"]) for item in A["results"] for ent in item["entities"]]
    overall_avg = float(np.mean(all_scores)) if all_scores else 0.0

    # Tabs for summary, entities, overall, and sentence samples
    tab_summary, tab_entities, tab_overall, tab_samples = st.tabs(
        ["Summary", "Entities", "Overall", "Sentences"]
    )

    with tab_summary:
        st.subheader("📌 Executive Summary")
        st.write(A["summary_text"])
        st.caption("Summary and timings are cached per file. Change methods and press **Reset analysis for this file** to recompute.")

    with tab_entities:
        st.subheader("🏆 Top Entities — Sentiment Distribution")
        top_tables = dfs_for_top_entities(entity_counts, entity_totals, top_n=top_k, min_count=1)
        if not top_tables:
            st.info("No entities found.")
        for name, df2 in top_tables:
            avg_val = float(entity_avg.get(name, 0.0))
            st.markdown(f"#### - {name} (avg. sentiment score = {avg_val:.3f})")
            st.table(style_sent_freq_center(df2))   # Hide index, center Frequency column

    with tab_overall:
        st.subheader(f"📊 Overall Sentiment Distribution (avg. sentiment score = {overall_avg:.3f})")
        st.altair_chart(chart_overall_distribution(overall_counts), use_container_width=True)
        st.table(style_sent_freq_center(overall_distribution_df(overall_counts)))

        export_df = all_entities_tables_csv(entity_counts, overall_counts, entity_avg=entity_avg)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download All Entities & Overall Sentiment Tables (CSV)",
            data=csv_bytes,
            file_name="all_entity_sentiment_tables.csv",
            mime="text/csv"
        )

    with tab_samples:
        st.subheader("🧩 Sample Sentences by Entity & Sentiment")
        if top_tables:
            # These controls only filter the display, don't trigger recomputation (session_state blocks recompute)
            pick_entity = st.selectbox("Choose entity", [n for n, _ in top_tables])
            pick_bucket = st.selectbox("Choose sentiment bucket", SENTIMENT_BINS, index=3)
            topn = st.slider("How many samples", 1, 10, 3)
            sents = sample_sentences(A["results"], pick_entity, pick_bucket, topn=topn)
            if not sents:
                st.info("No samples found for this selection.")
            else:
                for i, s in enumerate(sents, 1):
                    st.markdown(f"**{i}.** {s}")
# ===============================================================================================================
