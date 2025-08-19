# import os
# import streamlit as st
# import pandas as pd
# import numpy as np

# # Set OpenAI API key (temporary solution - use .env file in production)
# if not os.getenv('OPENAI_API_KEY'):
#     # Add your OpenAI API key here for development
#     os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'  # Replace with your actual key
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import re
# from datetime import datetime
# import io
# import base64
# from collections import Counter
# import time
# import nltk
# from textblob import TextBlob

# # Import your processing functions
# from get_sentiment_llm import extract_llm_sentiment
# from get_sentiment_nlp import extract_nlp_sentiment
# from get_entities_LLM import extract_llm_entities
# from get_entities_nlp import extract_nlp_entities
# from get_entities_ft_nlp import extract_ft_entities

# # Page configuration
# st.set_page_config(
#     page_title="Fed Sentiment AI - Decode the Fed in Seconds",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         padding: 2rem 0;
#         background: linear-gradient(90deg, #1f4e79, #2d5aa0);
#         color: white;
#         border-radius: 10px;
#         margin-bottom: 2rem;
#     }
#     .feature-card {
#         background: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 4px solid #2d5aa0;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         text-align: center;
#     }
#     .sentiment-positive {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .sentiment-negative {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .sentiment-neutral {
#         color: #6c757d;
#         font-weight: bold;
#     }
#     .entity-card {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 0.5rem 0;
#         border-left: 3px solid #2d5aa0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Main header
# st.markdown("""
# <div class="main-header">
#     <h1>📊 Fed Sentiment AI</h1>
#     <h3>Decode the Fed in Seconds</h3>
#     <p>AI-powered analysis of Federal Reserve transcripts for instant financial insights</p>
# </div>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'analysis_complete' not in st.session_state:
#     st.session_state.analysis_complete = False
# if 'transcript_text' not in st.session_state:
#     st.session_state.transcript_text = ""
# if 'sentiment_data' not in st.session_state:
#     st.session_state.sentiment_data = {}
# if 'detailed_results' not in st.session_state:
#     st.session_state.detailed_results = {}

# # Sidebar navigation
# st.sidebar.title("📊 Navigation")
# page = st.sidebar.selectbox(
#     "Choose your analysis type:",
#     ["🏠 Home", "📄 Upload & Analyze", "📈 Dashboard", "⚙️ Settings"]
# )

# # PDF processing function
# try:
#     import PyPDF2
#     PDF_AVAILABLE = True
# except ImportError:
#     PDF_AVAILABLE = False

# def extract_text_from_pdf(uploaded_file):
#     """Extract text from uploaded PDF file"""
#     if not PDF_AVAILABLE:
#         return "PDF processing unavailable. Please install PyPDF2."
    
#     try:
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e:
#         return f"Error reading PDF: {str(e)}"

# def split_into_sentences(text):
#     """Split text into sentences for processing"""
#     # Simple sentence splitting - you might want to use nltk.sent_tokenize for better results
#     sentences = re.split(r'[.!?]+', text)
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
#     return sentences

# def analyze_with_your_pipeline(text, entity_method="nlp", sentiment_method="llm"):
#     """
#     Main analysis function using your existing pipeline
#     Default: NLP for entities, LLM for sentiment
#     """
#     try:
#         # Step 1: Split text into sentences
#         sentences = split_into_sentences(text)
        
#         if not sentences:
#             return None, "No valid sentences found in the text."
        
#         # Step 2: Entity extraction
#         if entity_method == "nlp":
#             entity_results = extract_nlp_entities(sentences)
#         elif entity_method == "ftnlp":
#             entity_results = extract_ft_entities(sentences)
#         else:
#             entity_results = extract_llm_entities(sentences)
        
#         # Step 3: Prepare input for sentiment analysis
#         pipeline = []
#         for item in entity_results:
#             ents = [{"name": name} for name in item["entities"]]
#             pipeline.append({
#                 "sentence": item["sentence"],
#                 "entities": ents
#             })
        
#         # Step 4: Sentiment analysis
#         if sentiment_method == "nlp":
#             results = extract_nlp_sentiment([(item["sentence"], [ent["name"] for ent in item["entities"]]) for item in pipeline])
#         else:
#             results = extract_llm_sentiment(pipeline)
        
#         return results, None
        
#     except Exception as e:
#         return None, f"Error in analysis pipeline: {str(e)}"

# def process_results_for_dashboard(results):
#     """Process detailed results for dashboard visualization"""
#     if not results:
#         return {}
    
#     # Aggregate sentiment by entity type
#     entity_sentiments = {}
#     all_entities = []
#     sentence_sentiments = []
    
#     for item in results:
#         sentence = item['sentence']
        
#         # Calculate sentence-level sentiment (average of entity sentiments)
#         if item['entities']:
#             sentence_sentiment = np.mean([
#                 1 if ent['sentiment'].lower() == 'positive' else 
#                 -1 if ent['sentiment'].lower() == 'negative' else 0 
#                 for ent in item['entities']
#             ])
#             sentence_sentiments.append({
#                 'sentence': sentence[:100] + "..." if len(sentence) > 100 else sentence,
#                 'sentiment_score': sentence_sentiment,
#                 'entity_count': len(item['entities'])
#             })
        
#         # Process entities
#         for ent in item['entities']:
#             entity_name = ent['name']
#             sentiment = ent['sentiment']
            
#             all_entities.append({
#                 'entity': entity_name,
#                 'sentiment': sentiment,
#                 'sentence': sentence
#             })
            
#             if entity_name not in entity_sentiments:
#                 entity_sentiments[entity_name] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
#             entity_sentiments[entity_name][sentiment.lower()] += 1
    
#     # Calculate overall metrics
#     total_entities = len(all_entities)
#     sentiment_counts = Counter([ent['sentiment'].lower() for ent in all_entities])
    
#     # Most mentioned entities
#     entity_mentions = Counter([ent['entity'] for ent in all_entities])
#     top_entities = entity_mentions.most_common(10)
    
#     return {
#         'total_entities': total_entities,
#         'sentiment_counts': sentiment_counts,
#         'entity_sentiments': entity_sentiments,
#         'top_entities': top_entities,
#         'sentence_sentiments': sentence_sentiments,
#         'all_entities': all_entities
#     }

# # HOME PAGE
# if page == "🏠 Home":
#     # Key features section
#     st.markdown("## 🚀 Key Features")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         <div class="feature-card">
#             <h4>⚡ Fast Analysis</h4>
#             <p>Upload Fed transcripts and get instant AI-powered sentiment analysis using advanced NLP and LLM techniques.</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("""
#         <div class="feature-card">
#             <h4>📊 Visual Dashboard</h4>
#             <p>Interactive charts, sentiment heatmaps, and keyword extraction visualizations.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div class="feature-card">
#             <h4>🔍 Advanced Entity Recognition</h4>
#             <p>Specialized Fed entity extraction with NLP and fine-tuned models for financial terminology.</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("""
#         <div class="feature-card">
#             <h4>🤖 LLM-Powered Sentiment</h4>
#             <p>Context-aware sentiment analysis using Large Language Models for nuanced financial insights.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Demo section
#     st.markdown("---")
#     st.markdown("## 📋 Analysis Methods")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Entity Extraction Methods:**
#         - 🧠 **NLP**: Fast rule-based extraction (Default)
#         - 🤖 **LLM**: Context-aware entity recognition
#         - ⚡ **Fine-tuned NLP**: Specialized Fed terminology
#         """)
    
#     with col2:
#         st.markdown("""
#         **Sentiment Analysis Methods:**
#         - 🤖 **LLM**: Advanced context understanding (Default)
#         - 📊 **NLP**: Traditional sentiment scoring
#         """)
    
#     st.info("💡 Default configuration: NLP entities + LLM sentiment for optimal speed and accuracy balance!")

# # UPLOAD & ANALYZE PAGE
# elif page == "📄 Upload & Analyze":
#     st.markdown("## 📤 Upload Federal Reserve Transcript")
#     st.markdown("Upload your Fed meeting transcript (PDF or TXT) for instant AI analysis.")
    
#     # Analysis method selection
#     with st.expander("⚙️ Analysis Configuration"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             entity_method = st.selectbox(
#                 "Entity Extraction Method:",
#                 ["nlp", "llm", "ftnlp"],
#                 index=0,  # Default to nlp
#                 help="NLP: Fast rule-based | LLM: Context-aware | FT-NLP: Fine-tuned"
#             )
        
#         with col2:
#             sentiment_method = st.selectbox(
#                 "Sentiment Analysis Method:",
#                 ["llm", "nlp"],
#                 index=0,  # Default to llm
#                 help="LLM: Advanced context | NLP: Traditional scoring"
#             )
        
#         st.info(f"Selected: {entity_method.upper()} entities + {sentiment_method.upper()} sentiment")
    
#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Choose your Fed transcript file",
#         type=['pdf', 'txt'],
#         help="Supported formats: PDF, TXT (Max size: 10MB)"
#     )
    
#     if uploaded_file is not None:
#         st.success(f"✅ File uploaded: {uploaded_file.name}")
        
#         # File details
#         file_details = {
#             "Filename": uploaded_file.name,
#             "File size": f"{uploaded_file.size / 1024:.2f} KB",
#             "File type": uploaded_file.type
#         }
        
#         with st.expander("📋 File Details"):
#             for key, value in file_details.items():
#                 st.write(f"**{key}:** {value}")
        
#         # Process file content
#         try:
#             if uploaded_file.type == "application/pdf":
#                 with st.spinner("📄 Extracting text from PDF..."):
#                     transcript_text = extract_text_from_pdf(uploaded_file)
#             else:
#                 transcript_text = uploaded_file.read().decode('utf-8')
            
#             if "Error reading PDF" in transcript_text:
#                 st.error(transcript_text)
#                 st.stop()
            
#             st.session_state.transcript_text = transcript_text
            
#             # Show preview of content
#             with st.expander("👀 Content Preview (First 500 characters)"):
#                 st.text(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
            
#             # Analysis button
#             col1, col2 = st.columns([1, 3])
            
#             with col1:
#                 analyze_button = st.button("🤖 Analyze with AI", type="primary")
            
#             with col2:
#                 if st.session_state.analysis_complete:
#                     st.success("✅ Analysis complete! View results in Dashboard →")
            
#             if analyze_button:
#                 with st.spinner("🔍 AI is analyzing the transcript..."):
#                     # Analysis progress
#                     progress_bar = st.progress(0)
#                     status_text = st.empty()
                    
#                     status_text.text("📖 Processing document...")
#                     progress_bar.progress(20)
#                     time.sleep(0.5)
                    
#                     status_text.text(f"🔍 Extracting entities using {entity_method.upper()}...")
#                     progress_bar.progress(40)
#                     time.sleep(0.3)
                    
#                     status_text.text(f"🤖 Analyzing sentiment using {sentiment_method.upper()}...")
#                     progress_bar.progress(70)
                    
#                     # Perform actual analysis using your pipeline
#                     results, error = analyze_with_your_pipeline(
#                         transcript_text, 
#                         entity_method=entity_method, 
#                         sentiment_method=sentiment_method
#                     )
                    
#                     if error:
#                         st.error(f"❌ Analysis failed: {error}")
#                         st.stop()
                    
#                     progress_bar.progress(90)
#                     status_text.text("📊 Processing results...")
                    
#                     # Process results for dashboard
#                     processed_data = process_results_for_dashboard(results)
                    
#                     # Store results in session state
#                     st.session_state.detailed_results = results
#                     st.session_state.sentiment_data = processed_data
#                     st.session_state.analysis_complete = True
                    
#                     progress_bar.progress(100)
#                     status_text.text("✅ Analysis complete!")
                    
#                     # Show quick results preview
#                     st.markdown("### 🎯 Quick Results Preview")
                    
#                     if processed_data:
#                         col1, col2, col3 = st.columns(3)
                        
#                         with col1:
#                             total_entities = processed_data.get('total_entities', 0)
#                             positive_pct = processed_data.get('sentiment_counts', {}).get('positive', 0) / max(total_entities, 1) * 100
#                             st.metric(
#                                 "Entities Analyzed",
#                                 f"{total_entities:,}",
#                                 f"{positive_pct:.1f}% positive"
#                             )
                        
#                         with col2:
#                             top_entity = processed_data.get('top_entities', [('None', 0)])[0]
#                             st.metric(
#                                 "Top Entity",
#                                 top_entity[0][:20] + "..." if len(top_entity[0]) > 20 else top_entity[0],
#                                 f"{top_entity[1]} mentions"
#                             )
                        
#                         with col3:
#                             sentences_analyzed = len(processed_data.get('sentence_sentiments', []))
#                             st.metric(
#                                 "Sentences Analyzed",
#                                 f"{sentences_analyzed:,}",
#                                 f"Using {entity_method.upper()}+{sentiment_method.upper()}"
#                             )
                    
#                     st.success("🎉 Full analysis available in the Dashboard section!")
#                     st.balloons()
        
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
#     else:
#         # Sample transcript option
#         st.markdown("---")
#         st.markdown("### 📋 Try with Sample Data")
#         if st.button("📄 Load Sample Fed Transcript"):
#             sample_text = """
#             CHAIR POWELL: Good afternoon. At today's meeting, the Federal Open Market Committee decided to raise the target range for the federal funds rate by 25 basis points, bringing it to 5.25 to 5.50 percent. We continue to believe that the U.S. economy is in a good place, and our decision today reflects our commitment to returning inflation to our 2 percent target.

#             Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have been robust, and the unemployment rate has remained low. Nonetheless, inflation remains elevated. Over the 12 months ending in June, total PCE prices rose 3 percent; excluding the volatile food and energy categories, core PCE prices rose 4.1 percent.

#             The Federal Reserve is committed to bringing inflation back down to our 2 percent target. We will continue to assess additional information and its implications for monetary policy. As always, we remain prepared to adjust our approach as appropriate.

#             Looking ahead, we will continue to take a data-dependent approach to our policy decisions. We recognize that monetary policy affects the economy and inflation with uncertain lags, and we are mindful of the risks of both over- and under-tightening.

#             The labor market remains tight, with strong job creation and wage growth. However, we are monitoring signs that supply and demand in the labor market are coming into better balance. The housing market has shown signs of cooling, which may help reduce inflationary pressures over time.

#             We are committed to achieving maximum employment and price stability, and we will continue to use our tools to support these objectives. Thank you, and I look forward to your questions.
#             """
            
#             st.session_state.transcript_text = sample_text
#             st.success("✅ Sample transcript loaded! Click 'Analyze with AI' to see results.")

# # DASHBOARD PAGE
# elif page == "📈 Dashboard":
#     if not st.session_state.analysis_complete:
#         st.warning("⚠️ No analysis found. Please upload and analyze a transcript first.")
#         st.markdown("👈 Go to 'Upload & Analyze' to get started.")
#     else:
#         st.markdown("## 📊 Analysis Dashboard")
        
#         data = st.session_state.sentiment_data
#         detailed_results = st.session_state.detailed_results
        
#         if not data:
#             st.error("No data available for visualization.")
#             st.stop()
        
#         # Overview metrics
#         st.markdown("### 📈 Overview Metrics")
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric(
#                 "Total Entities",
#                 data.get('total_entities', 0),
#                 help="Total number of Fed-related entities found"
#             )
        
#         with col2:
#             sentiment_counts = data.get('sentiment_counts', {})
#             positive_count = sentiment_counts.get('positive', 0)
#             st.metric(
#                 "Positive Sentiment",
#                 positive_count,
#                 f"{positive_count/max(data.get('total_entities', 1), 1)*100:.1f}%"
#             )
        
#         with col3:
#             negative_count = sentiment_counts.get('negative', 0)
#             st.metric(
#                 "Negative Sentiment", 
#                 negative_count,
#                 f"{negative_count/max(data.get('total_entities', 1), 1)*100:.1f}%"
#             )
        
#         with col4:
#             neutral_count = sentiment_counts.get('neutral', 0)
#             st.metric(
#                 "Neutral Sentiment",
#                 neutral_count,
#                 f"{neutral_count/max(data.get('total_entities', 1), 1)*100:.1f}%"
#             )
        
#         # Sentiment distribution chart
#         st.markdown("### 📊 Sentiment Distribution")
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             sentiment_counts = data.get('sentiment_counts', {})
#             if sentiment_counts:
#                 fig = px.pie(
#                     values=list(sentiment_counts.values()),
#                     names=list(sentiment_counts.keys()),
#                     title="Overall Sentiment Distribution",
#                     color_discrete_map={
#                         'positive': '#28a745',
#                         'negative': '#dc3545', 
#                         'neutral': '#6c757d'
#                     }
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             st.markdown("**Top Entities:**")
#             top_entities = data.get('top_entities', [])
#             for i, (entity, count) in enumerate(top_entities[:5]):
#                 st.markdown(f"{i+1}. **{entity}** ({count} mentions)")
        
#         # Entity sentiment analysis
#         st.markdown("### 🏷️ Entity Sentiment Analysis")
        
#         entity_sentiments = data.get('entity_sentiments', {})
#         if entity_sentiments:
#             # Create entity sentiment dataframe
#             entity_data = []
#             for entity, sentiments in entity_sentiments.items():
#                 total = sum(sentiments.values())
#                 if total > 0:  # Only include entities with mentions
#                     entity_data.append({
#                         'Entity': entity,
#                         'Positive': sentiments.get('positive', 0),
#                         'Negative': sentiments.get('negative', 0),
#                         'Neutral': sentiments.get('neutral', 0),
#                         'Total': total
#                     })
            
#             if entity_data:
#                 entity_df = pd.DataFrame(entity_data).sort_values('Total', ascending=False).head(10)
                
#                 fig = px.bar(
#                     entity_df.melt(id_vars=['Entity', 'Total'], 
#                                   value_vars=['Positive', 'Negative', 'Neutral'],
#                                   var_name='Sentiment', value_name='Count'),
#                     x='Entity',
#                     y='Count',
#                     color='Sentiment',
#                     title="Sentiment by Entity (Top 10)",
#                     color_discrete_map={
#                         'Positive': '#28a745',
#                         'Negative': '#dc3545',
#                         'Neutral': '#6c757d'
#                     }
#                 )
#                 fig.update_xaxes(tickangle=45)
#                 st.plotly_chart(fig, use_container_width=True)
        
#         # Detailed results table
#         st.markdown("### 📝 Detailed Analysis Results")
        
#         with st.expander("View All Entity-Sentiment Pairs"):
#             if detailed_results:
#                 all_entities_data = []
#                 for item in detailed_results:
#                     sentence = item['sentence']
#                     for ent in item['entities']:
#                         all_entities_data.append({
#                             'Entity': ent['name'],
#                             'Sentiment': ent['sentiment'],
#                             'Context': sentence[:100] + "..." if len(sentence) > 100 else sentence
#                         })
                
#                 if all_entities_data:
#                     results_df = pd.DataFrame(all_entities_data)
#                     st.dataframe(
#                         results_df,
#                         use_container_width=True,
#                         height=400
#                     )
                    
#                     # Download results
#                     csv = results_df.to_csv(index=False)
#                     st.download_button(
#                         label="📥 Download Results as CSV",
#                         data=csv,
#                         file_name="fed_sentiment_analysis.csv",
#                         mime="text/csv"
#                     )

# # SETTINGS PAGE
# elif page == "⚙️ Settings":
#     st.markdown("## ⚙️ Analysis Settings")
    
#     st.markdown("### 🔧 Default Configuration")
#     st.info("""
#     **Current Defaults:**
#     - Entity Extraction: NLP (Fast rule-based)
#     - Sentiment Analysis: LLM (Context-aware)
    
#     This combination provides the best balance of speed and accuracy for Fed transcript analysis.
#     """)
    
#     st.markdown("### 📚 Model Information")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Entity Extraction Methods:**
#         - **NLP**: Rule-based pattern matching with Fed-specific entities
#         - **LLM**: Large Language Model entity recognition
#         - **FT-NLP**: Fine-tuned NLP model for financial terminology
#         """)
    
#     with col2:
#         st.markdown("""
#         **Sentiment Analysis Methods:**
#         - **LLM**: Advanced context-aware sentiment analysis
#         - **NLP**: Traditional sentiment scoring algorithms
#         """)
    
#     st.markdown("### 📊 Entity Categories")
#     st.markdown("""
#     Our system recognizes Fed-specific entities including:
#     - Federal Reserve terminology
#     - Interest rates and monetary policy
#     - Inflation and employment metrics
#     - Economic outlook indicators
#     - Financial markets and instruments
#     """)

# # Download required NLTK data if needed
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('vader_lexicon', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except:
#     pass









#---------------------------------------------------------------------------------------------------------------
import os
import streamlit as st
import PyPDF2
import numpy as np
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com")

# Import ALL backend functions (updated imports)
from get_sentiment_llm import extract_llm_sentiment
from get_sentiment_nlp import extract_nlp_sentiment
from get_entities_LLM import extract_llm_entities
from get_entities_nlp import extract_nlp_entities
from get_entities_ft_nlp import extract_ft_entities

# --- Helper Functions ---
def read_pdf(file):
    """Extract text from an uploaded PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def sentiment_label(score: float) -> str:
    """Map sentiment score (-1 to 1) into human-readable label."""
    if score <= -0.66: return "Very Negative"
    elif score <= -0.33: return "Medium Negative"
    elif score < 0: return "Slightly Negative"
    elif score == 0: return "Neutral"
    elif score < 0.33: return "Slightly Positive"
    elif score < 0.66: return "Medium Positive"
    else: return "Very Positive"

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

# --- Streamlit UI ---
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

    # Split into sentences (improved method)
    sentences = [s.strip() for s in pdf_text.split(".") if s.strip() and len(s.strip()) > 10]

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

    # --- Your existing insights section (summary, entities, etc.) ---
    # Generate summary
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
