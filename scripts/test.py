
# library imports
import os
import numpy as np
import pandas as pd
import string
import time
import urllib.request
import zipfile
import torch
import pickle

from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
#import unidecode
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#local imports
import clean_dataframe_text
from clean_dataframe_text import join_and_clean_text, clean_text, clean_dataframe
from config import datapath, sent_transformer
from create_text_embeddings import create_embeddings, prepare_df

example_df = pd.read_pickle('../data/podcast_df_040423.pkl')
example_df = prepare_df(example_df)
cleaned_df = clean_dataframe(example_df)
data_and_embeds, embeds = create_embeddings(cleaned_df)

display(embeds)