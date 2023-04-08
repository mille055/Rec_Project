
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
import unidecode
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#local imports
import clean_dataframe_text
from clean_dataframe_text import join_and_clean_text, clean_text
from config import datapath, sent_transformer

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## loads in the full dataset
def get_full_dataset(datapath=datapath):
    # Unpickle the dataset
    loaded_df = pd.read_pickle(datpath)
    loaded_df = loaded_df.reset_index(drop=True)

    return loaded_df


### reduces the size of the dataframe prior to processing as do not need user ratings and userid for this
def prepare_df(df):
  '''
  Takes the full dataframe which has multiple user ratings for each podcast, and
  gets rid of duplicate and eliminates user, rating columns. This is for processing
  the embeddings and cosimilarity matrices. alternatively, can loaded in the pickled
  dataframe podcast_base_with_embeds.pkl which contains the podcast info as well as the
  text embeddings, or podcast_embeddings_only.pkl which has only the embeddings for 
  each itunes_id

  Input:
    df(dataframe): the dataframe to prepare
  Output:
    df_no_dups(dataframe): smaller dataframe with no user/rating  
  '''
  df1= podcast_df.copy()
  # get rid of duplicates based on itunes_id
  cols_drop_dup = ['itunes_id']
  df_no_dups = df1.drop_duplicates(subset=cols_drop_dup)
  print('shape of new df without duplicates is ', df_no_dups.shape)
  # remove columns containing the user and user rating
  print('removing user  and rating columns')
  df_no_dups.drop(columns=['user', 'rating'], inplace=True)

  return df_no_dups


# Create embeddings for given cols (such as description, episode descriptions, genre)
def create_embeddings(df, cols, verbose=True, picklish = False):
  '''
  Uses a sentence transformer model (from the config) to get embeddings for the
  text in the provided columns. The dataframe should have been previously cleaned with
  clean_dataframe_text
  Input: 
    df(dataframe): the dataframe containing the cleaned trext data
    cols(list[str]): column names on which to perform embeddings
    verbose(bool): prints updates if true
    picklish(bool): pickles the modified and new dataframes for storage
    Output: df1(dataframe): modified dataframe which has base data plus new columns for the embeddings
        embeddings_df(dataframe): new dataframe with itunes_id and embeddings only
  '''

  df1 = df.copy()
  senttrans_model = SentenceTransformer(sent_transformer,device=device)

  new_col_names = []
  for col in cols:
    if verbose:
        print('Now embedding column', col)
    col_data = df1[col].values.tolist()
    col_embeds = [senttrans_model.encode(doc) for doc in col_data]
    new_col_name = col + '_embedding'
    df1[new_col_name] = col_embeds
    new_col_names.append(new_col_name)

  embeddings_df = df1[new_col_names]
  embeddings_df['itunes_id'] = df1['itunes_id']

  if picklish:
     df1.to_pickle('podcast_base_with_embeds.pkl')
     embeddings_df.to_pickle('podcast_embeddings_only.pkl')

  return df1, embeddings_df



