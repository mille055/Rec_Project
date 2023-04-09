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
    '''
    Creates the cosine similarity matrix using sklearn's cosine_similarity function. It takes in a dataframe and columns
    comprise the feature set to perform the similarity upon. It concatenates the information from the various columns
    in the axis=1 direction; for N entries in the dataframe, it creates an NxN cosine similarity matrix
    Input: 
        df(dataframe): the dataframe for the data
        feats(list[string]): the column names that are to be included in the cosine similarity calculation
    Output:
        matrix(np.array): the cosine similarity matrix
    '''
    array_list = []
    for feat in feats:
        array_list.append(np.stack(df[feat].values))
    concat_array = np.concatenate((array_list), axis=1)
    print('after concatenate, data size is ', concat_array.shape)
    matrix = cosine_similarity(concat_array)
  
    return matrix

# predicts rating based on a user-item pair, the cosine similarity matrix, and the training data
def predict_rating(user_item_pair,simtable,X_train, y_train):
     '''
    Predicts the rating of the podcast for a selected user-item pair using the similarity matrix. It requires the 
    training dataset as well in the calculation. 
    Input: 
        user_item_pair(tuple): the 'user' and 'itunes_id' columns for a particular row 
        simtable(dataframe): the cosine similarity matrix, as a dataframe that can be index, with labels 'itunes_id' in both directions
        X_train, y_train: training dataset, where X_train is Nx2 and contains 'user' and 'itunes_id' and y_train is 'rating' column
    Output:
        most_similar_rating(int): the predicted numerical rating for the user-item pair on a scale of 1-5
    '''
    
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
    '''
    Predicts the rating of the podcast for a selected user-item pair using the similarity matrix. It requires the 
    training dataset as well in the calculation. 
    Input: 
        user_item_pair(tuple): the 'user' and 'itunes_id' columns for a particular row 
        simtable(dataframe): the cosine similarity matrix, as a dataframe that can be index, with labels 'itunes_id' in both directions
        X_train, y_train: training dataset, where X_train is Nx2 and contains 'user' and 'itunes_id' and y_train is 'rating' column
    Output:
        most_similar_rating(int): the predicted numerical rating for the user-item pair on a scale of 1-5
    '''
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
     '''
    Splits the dataframe into train and val datasets using sklearn train_test_split
    Input: 
        df(dataframe): data to be split
    Output:
        X_train, X_val, y_train, y_val(pd.Series): Training and validation data and labels
    '''
    X = df[['user', 'itunes_id']]
    y = df['rating']
    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0, test_size=0.2)

    return X_train, X_val, y_train, y_val


def calculate_rmse(X_train, X_val, y_train, y_val, simtable):
    '''
    Calculates RMSE for the validation data.
    Input: 
        X_train, X_val, y_train, y_val(pd.Series): Training and validation datasets
        simtable(np.array): cosine similarity matrix
    Output:
        X_train, X_val, y_train, y_val(pd.Series): Training and validation data and labels
        simtable(dataframe): the cosine similarity matrix, as a dataframe that can be index, 
            with labels 'itunes_id' in both directions
    '''
    
    # Get the predicted ratings for each podcast in the validation set and calculate the RMSE
    ratings_set = X_val.apply(lambda x: predict_rating(x,simtable,X_train, y_train ), axis=1)

    # Have many Nan so getting rid of those by creating dataframe and dropna
    df = pd.DataFrame({'ratings_set': ratings_set, 'y_val':y_val})
    df.dropna(inplace=True)

    rmse = np.sqrt(mean_squared_error(df.y_val.values,df.ratings_set.values))
    print('RMSE of predicted ratings is {:.3f}'.format(rmse))

    return rmse


def generate_recommendations(user,simtable,df):
    '''
    Gets podcast recommendations for a selected user using the similarity matrix. 
    Input: 
        user(string): a selection from the 'user' column from the dataframe for a particul
        simtable(dataframe): the cosine similarity matrix, as a dataframe that can be index, with labels 'itunes_id' in both directions
        df(dataframe): the dataframe containing the data, with the version that has all the user/ratings
    Output: 
        topratedpodcast_title(str): name of the top rated podast by the user
        mostsim_podcasts(list[(str,str)]): list of the names and genres of the most similar podcasts 
    '''
    
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
     '''
    Predicts the rating of the podcast for a selected user-item pair using the similarity matrix. It requires the 
    training dataset as well in the calculation. 
    Input: 
        user_item_pair(tuple): the 'user' and 'itunes_id' columns for a particular row 
        simtable(dataframe): the cosine similarity matrix, as a dataframe that can be index, with labels 'itunes_id' in both directions
        X_train, y_train: training dataset, where X_train is Nx2 and contains 'user' and 'itunes_id' and y_train is 'rating' column
    Output:
        most_similar_rating(int): the predicted numerical rating for the user-item pair on a scale of 1-5
    '''

    # # Filter similarity matrix to only podcasts already consumed by user
    # prior_podcast = X_train.loc[X_train['user']==user, 'itunes_id'].tolist()
    # simtable_filtered = simtable.loc[podcast,prior_podcast]
    # # Get the most similar movie already watched to current podcast to rate
    # most_similar = simtable_filtered.index[np.argmax(simtable_filtered)]
    # # Get user's rating for most similar podcast
    # idx = X_train.loc[(X_train['user']==user) & (X_train['itunes_id']==most_similar)].index.values[0]
    # most_similar_rating = y_train.loc[idx]
    # return most_similar_rating


## generates a set of new rows for the podcast df based on the input user dictionary
def append_new_user(name, podcast_list, ratings, df):
     '''
    Adds new user info to the dataframe with a user name, list of podcasts and ratings
    Input: 
        name(str): the name of the user
        podcast_list(list[str]): list of itunes_id's that the new user has rated
        ratings(list[int]): list of rating values for the podcasts for the new user
        df(dataframe): the main dataframe, to which the new user info is added
    Output:
        addended_df(dataframe): the new dataframe including the new user info. all columns present except doesn't fill
            the episode_descriptions text column
    '''


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
    addended_df = df1.append(new_user_df)

    return addended_df

