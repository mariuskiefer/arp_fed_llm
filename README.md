# Fed Sentiment Analysis App

Analyze **FOMC** documents for **entities** and their **sentiment** with a simple Streamlit UI.  
Upload any PDF — **Statements**, **Minutes**, or **Press conference transcripts** — choose an extraction mode (**NLP** or **LLM**), and get sentence-level outputs, summaries, and frequency tables.

Official source for documents: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

---

## Features

- PDF upload for **Statements**, **Minutes**, and **Press conference transcripts**
- Two extraction modes:
  - **NLP** — fine-tuned models for entities and sentiment
  - **LLM** — API-driven extraction with strict JSON schema
- Sentence-level **entities + sentiment bins**
- Document summary and **entity frequency** tables

---

## Requirements

- **Conda** (Anaconda or Miniconda)
- **Git** and **Git LFS** (for large model files)
- **DeepSeek API key**

---

## Installation
> Run the following commands in your terminal **(bash)**. Each block starts with a comment as a reminder.

1) **Clone the repository**

    ```bash
    git clone <YOUR_REPO_URL>
    cd <YOUR_REPO_NAME>
    ```

2) **Create the Conda environment**

    ```bash
    conda env create -f environment.yml
    ```

3) **Activate the environment**

    ```bash
    conda activate arp_feds
    ```

4) **Fetch model artifacts with Git LFS**

    ```bash
    git lfs install
    git lfs pull
    ```
    Required local models (downloaded via LFS):
    - `finbert-finetuned/` — sentiment model  
    - `best_model_entities/` — multi-label entity extractor 

5) **Configure the LLM API and .env**

    Create a `.env` file in the project root.
    Create variables:
    `OPENAI_API_KEY="key"`
    `TOKENIZERS_PARALLELISM=true`
    `SENTIMENT_DEBUG=1`
    `SENTIMENT_DEBUG_FILE=sentiment_debug.log`
    `SENTIMENT_OUTPUT_FILE=sentiment_output.json`

6) **Run the Streamlit app**

    ```bash
    streamlit run UI_APP.py
    ```