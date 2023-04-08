import os
import numpy as np
import pandas as pd
import string
import time
import urllib.request
import zipfile
import torch
from sentence_transformers import SentenceTransformer
import unidecode
import sys
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# to creater cosine similarity matrix from the text embeddings in genre, description, and episode_descriptions
def create_cosine_similarity(df, feats = ['genre_embedding', 'description_embedding', 'episode_descriptions_embedding']):
  array_list = []
  for feat in feats:
    array_list.append(np.stack(df[feat].values))
  concat_array = np.concatenate((array_list), axis=1)
  print('after concatenate, data size is ', concat_array.shape)
  matrix = cosine_similarity(concat_array)
  
  return matrix

# predicts rating based on a user-item pair, the cosine similarity matrix, and the training data
def predict_rating(user_item_pair,simtable,X_train, y_train):
    podcast_to_rate = user_item_pair['itunes_id']
    user_to_assess = user_item_pair['user']
    #print(user_to_assess, podcast_to_rate)
    
    # Filter similarity matrix to only podcasts already reviewed by user
    prior_podcasts = X_train.loc[X_train['user']==user_to_assess, 'itunes_id'].tolist()
    #print(prior_podcasts)
    if not prior_podcasts:
      return None
    
    simtable_filtered = simtable.loc[podcast_to_rate, prior_podcasts]
    #print(simtable_filtered)
    
    # Get the most similar podcast to current podcast to rate
    most_similar = simtable_filtered.index[np.argmax(simtable_filtered)]
    #print(most_similar)
    
    # Get user's rating for most similar podcast
    idx = X_train.loc[(X_train['user']==user_to_assess) & (X_train['itunes_id']==most_similar)].index.values[0]
    #print('idx is ',idx)
    most_similar_rating = y_train.loc[idx]
    
    return most_similar_rating

# predict rating for a sample user
def predict_new_pair_rating(user,podcast,simtable,X_train, y_train):
    # Filter similarity matrix to only podcasts already consumed by user
    prior_podcast = X_train.loc[X_train['user']==user, 'itunes_id'].tolist()
    simtable_filtered = simtable.loc[podcast,prior_podcast]
    # Get the most similar movie already watched to current podcast to rate
    most_similar = simtable_filtered.index[np.argmax(simtable_filtered)]
    # Get user's rating for most similar podcast
    idx = X_train.loc[(X_train['user']==user) & (X_train['itunes_id']==most_similar)].index.values[0]
    most_similar_rating = y_train.loc[idx]
    return most_similar_rating

def create_train_val_datasets(df):
    X = df[['user', 'itunes_id']]
    y = df['rating']
    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.2)

    return X_train, X_val, y_train, y_val


def calculate_rmse(X_train, y_train, X_val, y_val, simtable):
    # Get the predicted ratings for each podcast in the validation set and calculate the RMSE
    ratings_set = X_val.apply(lambda x: predict_rating(X_val,simtable,X_train, y_train ))

    # Have many Nan so getting rid of those by creating dataframe and dropna
    df = pd.DataFrame({'ratings_set': ratings_set, 'y_val':y_val})
    df.dropna(inplace=True)

    rmse = np.sqrt(mean_squared_error(df.y_val.values,df.ratings_set.values))
    print('RMSE of predicted ratings is {:.3f}'.format(rmse))

    return rmse


def generate_recommendations(user,simtable,df):
    # Get top rated podcast by user
    user_ratings = df.loc[df['user']==user]
    user_ratings = user_ratings.sort_values(by='rating',axis=0,ascending=False)
    topratedpodcast = user_ratings.iloc[0,:]['itunes_id']
    topratedpodcast_title = df.loc[df['itunes_id']==topratedpodcast,'title'].values[0]
    # Find most similar podcasts to the user's top rated movie
    sims = simtable.loc[topratedpodcast,:]
    mostsimilar = sims.sort_values(ascending=False).index.values
    # Get 10 most similar podcasts excluding the podcast itself
    mostsimilar = mostsimilar[1:11]
    # Get titles of movies from ids
    mostsim_podcasts = []
    for m in mostsimilar:
        mostsim_podcasts.append(df.loc[df['itunes_id']==m,['title', 'genre']].values[0])
        #mostsim_podcast_genres.append(df.loc[df['itunes_id']==m, 'genre'].values[0])
    return topratedpodcast_title, mostsim_podcasts

def predict_new_pair_rating(user,podcast,simtable,X_train, y_train):
    # Filter similarity matrix to only podcasts already consumed by user
    prior_podcast = X_train.loc[X_train['user']==user, 'itunes_id'].tolist()
    simtable_filtered = simtable.loc[podcast,prior_podcast]
    # Get the most similar movie already watched to current podcast to rate
    most_similar = simtable_filtered.index[np.argmax(simtable_filtered)]
    # Get user's rating for most similar podcast
    idx = X_train.loc[(X_train['user']==user) & (X_train['itunes_id']==most_similar)].index.values[0]
    most_similar_rating = y_train.loc[idx]
    return most_similar_rating


## generates a set of new rows for the podcast df based on the input user dictionary
def append_new_user(name, podcast_list, ratings, df):
    new_user_data = []
    df1 = df.copy()

    for itunes_id, rating in zip(podcast_list, ratings):
        #print('matching podcast', itunes_id)
        matching_row = df1.loc[df1.itunes_id == itunes_id].iloc[0]
        #print(matching_row)

        user_row = {'user': name, 'itunes_id': itunes_id, 'rating': rating}
        for column in df1.columns:
            if column not in user_row and column != 'episode_descriptions':
                #print('adding column', column)
                user_row[column] = matching_row[column]

        new_user_data.append(user_row)

    new_user_df = pd.DataFrame(new_user_data)
    df1.append(new_user_df)

    return df1

