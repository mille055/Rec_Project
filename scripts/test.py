
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
from content_based_filtering import create_cosine_similarity, predict_rating, predict_new_pair_rating, create_train_val_datasets
from content_based_filtering import calculate_rmse, generate_recommendations, append_new_user

example_df = pd.read_pickle('../data/podcast_df_040423.pkl')
reduced_df = prepare_df(example_df)
cleaned_df = clean_dataframe(reduced_df)
#data_and_embeds, embeds = create_embeddings(cleaned_df)
embeds = pd.read_pickle('../data/podcast_embeddings_only.pkl')
print(embeds)

## Calculate cosine similarity matrices for different combinations of feaatures
cs_all = create_cosine_similarity(embeds) # all three (genre, description, episode_descriptions)
cs_genre = create_cosine_similarity(embeds, feats=['genre_embedding'])
cs_desc = create_cosine_similarity(embeds, feats=['description_embedding'])
cs_episo = create_cosine_similarity(embeds, feats=['episode_descriptions_embedding'])
cs_gen_desc = create_cosine_similarity(embeds, feats=['genre_embedding', 'description_embedding'])

# get train val sets
X_train, X_val, y_train, y_val = create_train_val_datasets(example_df)
print('X_train has shape: ', X_train.shape)
print('X_train columns are: ', X_train.columns)
print('X_val has shape: ', X_val.shape)
print('X_val columns are: ', X_val.columns)


#make similarity matrix to use from the choice of the cosine similarity numpy arrays above
sim_matrix = pd.DataFrame(cs_gen_desc, columns=embeds.itunes_id,index=embeds.itunes_id)

# calculate rmse for the validation set
val_rmse = calculate_rmse(X_train, X_val, y_train, y_val, sim_matrix)

# Get recommendations for a random user
user = example_df.iloc[100].user
topratedpodcast, recs = generate_recommendations(user,sim_matrix,example_df)
print("User's highest rated podcast was {}".format(topratedpodcast))
for i,rec in enumerate(recs):
  print('Recommendation {} (Title, Genre): {}, {}'.format(i,rec[0], rec[1]))

## making a new user to generate predictions and appending to the original dataframe

wait_id = '121493804'
superdatascience_id = '1163599059'
thisamerican_id = '1223767856'
collegebball_id = '268800565'
verge_id = '430333725'
userA = pd.DataFrame({'user': 'A', 'itunes_id': [wait_id, superdatascience_id, thisamerican_id, collegebball_id, verge_id], 'rating': [4, 5, 4, 4, 5] })
updated_df = append_new_user('A', userA['itunes_id'], userA['rating'], example_df)
print(updated_df.loc[updated_df.user=='A'])
#print(updated_df.tail(10))

## getting recommendations for the new user
user = 'A'
topratedpodcast, recs = generate_recommendations(user,sim_matrix,updated_df)
print("User's highest rated podcast was {}".format(topratedpodcast))
for i,rec in enumerate(recs):
  print('Recommendations {} (Title, Genre): {}, {}'.format(i,rec[0], rec[1]))