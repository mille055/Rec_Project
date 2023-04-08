import os
import numpy as np
import pandas as pd
import string
import time
import urllib.request
import zipfile
import torch

from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import unidecode
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## configuration variables
datapath = '../data/podcast_df_0040423.pkl'
sent_transformer = 'all-MiniLM-L6-v2'