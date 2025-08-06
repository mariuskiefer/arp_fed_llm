# File: evaluation.py
# Description:  This file is used to evaluate the performance of the entity extraction and sentiment analysis methods. 
#               We are using data from ARP_dataset_fixed_Sentiment.csv to finetune nlp methods and then evaluate llm and nlp methods with the data that was not used for finetuning.

import os
from get_sentiment_llm import extract_llm_sentiment
from get_sentiment_nlp import extract_nlp_sentiment
from get_entities_LLM import extract_llm_entities  
from get_entities_nlp import extract_nlp_entities