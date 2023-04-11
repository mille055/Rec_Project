import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_PODCAST_DATA_CLEAN = '../data/cleaned_df.pkl'
_PODCAST_DATA_ALL = '../data/podcast_df_040423.pkl'
_PODCAST_EMB = '../data/podcast_embeddings_only.pkl'


def create_cosine_similarity(df, feats = ['genre_embedding', 'description_embedding', 'episode_descriptions_embedding']):
    """
    Creates the cosine similarity matrix using sklearn's cosine_similarity function. It takes in a dataframe and columns
    comprise the feature set to perform the similarity upon. It concatenates the information from the various columns
    in the axis=1 direction; for N entries in the dataframe, it creates an NxN cosine similarity matrix
    
    Args: 
        df(pd.DataFrame): the dataframe for the data
        feats(List[str]): the column names that are to be included in the cosine similarity calculation
    
    Returns:
        matrix(np.ndarray): the cosine similarity matrix
    """
    array_list = []
    for feat in feats:
        array_list.append(np.stack(df[feat].values))
    concat_array = np.concatenate((array_list), axis=1)
    matrix = cosine_similarity(concat_array)
    return matrix


def append_new_user(name, podcast_list, ratings, df):
    '''
    Adds new user info to the dataframe with a user name, list of podcasts and ratings
    
    Args: 
        name(str): the name of the user
        podcast_list(List[str]): list of itunes_id's that the new user has rated
        ratings(List[int]): list of rating values for the podcasts for the new user
        df(pd.DataFrame): the main dataframe, to which the new user info is added
    
    Returns:
        addended_df(pd.DataFrame): the new dataframe including the new user info. all columns present except doesn't fill the episode_descriptions text column
    '''
    new_user_data = []
    df1 = df.copy()

    for itunes_id, rating in zip(podcast_list, ratings):
        matching_row = df1.loc[df1.itunes_id == itunes_id].iloc[0]
        user_row = {'user': name, 'itunes_id': itunes_id, 'rating': rating}
        for column in df1.columns:
            if column not in user_row and column != 'episode_descriptions':
                user_row[column] = matching_row[column]
        new_user_data.append(user_row)

    new_user_df = pd.DataFrame(new_user_data)
    addended_df = df1.append(new_user_df)
    return addended_df


def generate_recommendations(user,simtable,df, k):
    """
    Gets podcast recommendations for a selected user using the similarity matrix. 
    
    Args: 
        user(str): a selection from the 'user' column from the dataframe for a particul
        simtable(pd.DataFrame): the cosine similarity matrix, as a dataframe that can be index, with labels 'itunes_id' in both directions
        df(pd.Dataframe): the dataframe containing the data, with the version that has all the user/ratings
    
    Returns: 
        topratedpodcast_title(str): name of the top rated podast by the user
        mostsim_podcasts(List(str)): list of the itunes_id of the most similar podcasts 
    """
    
    # Get top rated podcast by user
    user_ratings = df.loc[df['user']==user]
    user_ratings = user_ratings.sort_values(by='rating',axis=0,ascending=False)
    topratedpodcast = user_ratings.iloc[0,:]['itunes_id']
    topratedpodcast_title = df.loc[df['itunes_id']==topratedpodcast,'title'].values[0]
    # Find most similar podcasts to the user's top rated movie
    sims = simtable.loc[topratedpodcast,:]
    mostsimilar = sims.sort_values(ascending=False).index.values
    # Get top k most similar podcasts excluding the podcast itself
    mostsimilar = mostsimilar[1:k+1]
    # Get titles of movies from ids
    mostsim_podcasts = pd.DataFrame()
    for m in mostsimilar:
        mostsim_podcasts = pd.concat([mostsim_podcasts, df.loc[df['itunes_id']==m,['itunes_id']].drop_duplicates(subset=["itunes_id"], keep="first")])
    return topratedpodcast_title, mostsim_podcasts


def main(args):
    # Load pickled podcast dataframe and embeddings
    podcast_df = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_DATA_CLEAN)).reset_index(drop=True)
    user_df = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_DATA_ALL)).reset_index(drop=True)
    embeddings = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_EMB)).reset_index(drop=True)
    # Get cosine simlarity matrix between all podcasts based on embeddings
    cs_gen_desc = create_cosine_similarity(embeddings, feats=['genre_embedding', 'description_embedding'])
    sim_matrix = pd.DataFrame(cs_gen_desc, columns=embeddings.itunes_id, index=embeddings.itunes_id)
    # Get number of unique podcasts
    len_podcasts = podcast_df['itunes_id'].nunique()
    # Get unique list of users
    users = user_df['user'].unique().tolist()
    # Create progress bar
    pbar = tqdm(total=len(users))
    # Initialize empty list of unique recommendations
    recs = set()
    for user in users:
        # Obtain recommended podcasts for each user
        _, podcastrecs = generate_recommendations(user, sim_matrix, user_df, args.k)
        # Add recs to set
        recs.update(podcastrecs['itunes_id'].to_list())
        pbar.update()
    pbar.close()
    # Calculate coverage of recommendations across all users
    cov = len(recs)/len_podcasts * 100.0
    print(f"Coverage {cov:.2f}%")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="Script to check coverage of podcast recommendations based on content filtering approach",
        epilog="Example usage: content_coverage.py --k 5"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations to generate for each user, must be greater than 0. Defaults to 5.")
    args = parser.parse_args()
    print("Command Line Args: ", args)

    # Pass command line arguments to main function
    main(args)