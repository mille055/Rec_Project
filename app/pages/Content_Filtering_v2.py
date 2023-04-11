import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_star_rating import st_star_rating
from sklearn.metrics.pairwise import cosine_similarity


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_PODCAST_DATA = '../../data/cleaned_df.pkl'
_PODCAST_EMB = '../../data/podcast_embeddings_only.pkl'


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


def generate_recommendations(user,simtable,df):
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
    # Get 10 most similar podcasts excluding the podcast itself
    mostsimilar = mostsimilar[1:11]
    # Get titles of movies from ids
    mostsim_podcasts = []
    for m in mostsimilar:
        mostsim_podcasts.append(df.loc[df['itunes_id']==m,['itunes_id']].values[0])
    return topratedpodcast_title, mostsim_podcasts


def make_grid(rows, cols):
    """
    Creates a grid of size rows * cols in Streamlit. 
    Reference: https://towardsdatascience.com/how-to-create-a-grid-layout-in-streamlit-7aff16b94508.
    
    Args:
        rows(int): Number of rows
        cols(int): Number of columns
    
    Returns:
        grid(List[List]): A two-dimensional grid of size rows * cols
    """
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid


def main():
    # Instantiate a random number generator to randomize coldstart list of podcasts
    rng = np.random.default_rng()
    # Load pickled podcast dataframe and embeddings
    podcast_df = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_DATA)).reset_index(drop=True)
    embeddings = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_EMB)).reset_index(drop=True)
    # Get cosine simlarity matrix between all podcasts based on embeddings
    cs_gen_desc = create_cosine_similarity(embeddings, feats=['genre_embedding', 'description_embedding'])
    sim_matrix = pd.DataFrame(cs_gen_desc, columns=embeddings.itunes_id, index=embeddings.itunes_id)
    
    # Setup the Streamlit page
    st.set_page_config(page_title='Podcast Recommender 🎙️', layout='wide')
    st.title('Podcast Recommender 🎙️')
    st.markdown('Tell us some podcasts that you like and we will recommend some others to listen to!')

    # Setup display style of podcast title text
    st.markdown("""
        <style>
        .auto-resize-text {
            display: inline-block;
            font-size: 18px;
            max-width: 100%;
            line-height: 1.2;
            height: 48px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
        <script>
        function resizeText(element) {
            let fontSize = parseFloat(window.getComputedStyle(element).fontSize);
            while (element.scrollHeight > element.offsetHeight) {
                fontSize -= 0.5;
                element.style.fontSize = fontSize + 'px';
            }
        }

        document.addEventListener("DOMContentLoaded", function() {
            const elements = document.querySelectorAll(".auto-resize-text");
            elements.forEach(function(element) {
                resizeText(element);
            });
        });
        </script>
        """, unsafe_allow_html=True)
    
    # Initialize Streamlit session variables to manage control flow
    st.session_state.coldstart = False
    st.session_state.ratings = False
    st.session_state.recs = False

    # Placeholder container that can be emptied later
    placeholder = st.empty()
    
    with placeholder.form(key='begin'):
        name = st.text_input('Welcome! What\'s your name?') # Input text box for user's name
        selected_genres = st.multiselect('Select genres you want to include:', options=podcast_df['genre'].unique()) # Allow user to select genres
        submit = st.form_submit_button('Next') # Submit button to proceed to next section
    
    # Sample an initial list of podcasts based on popularity and genre for user selection to solve coldstart problem
    if selected_genres:
        coldstart_df = podcast_df[podcast_df['genre'].isin(selected_genres)].sort_values(by=['genre', 'avg_rating', 'num_reviews'], ascending=False).reset_index(drop=True)
    else:
        coldstart_df = podcast_df.sort_values(by=['genre', 'avg_rating', 'num_reviews'], ascending=False).reset_index(drop=True)
    coldstart_df = pd.DataFrame(coldstart_df.groupby('genre').apply(lambda x: x.head(20))).reset_index(drop=True)
    coldstart_df_select = coldstart_df.sample(n=20, random_state=rng, ignore_index=True)
    
    if submit and name is not None and not st.session_state.coldstart:
        st.session_state.name = name
        st.session_state.coldstart_df = coldstart_df_select.copy()
        # Clear out previous container
        placeholder.empty()
        # Create another container that can be emptied later
        coldstart = st.empty()
        
        # Function that stores selected podcasts upon clicking Next button
        def tally_selection():
            st.session_state.coldstart = True
            coldstart_selected = [podcast for podcast in st.session_state.coldstart_df['itunes_id'].to_list() if st.session_state[podcast]]
            if 0 < len(coldstart_selected) <= 5:
                st.session_state['selections'] = coldstart_selected
            elif len(coldstart_selected) == 0:
                st.warning("Please select at least one podcast.")
            else:
                st.warning("Please select no more than 5 podcasts.")

        # Display in a grid a sample of popular podcasts that user can select from (re: coldstart problem)
        with coldstart.form(key='selections'):
            st.subheader(f'Hi {st.session_state.name}! Select up to 5 podcasts that you like:')
            grid = make_grid(4,5)
            for i in range(4):
                for j in range(5):
                    with grid[i][j]:
                        itunes_id = st.session_state.coldstart_df['itunes_id'][i*5+j]
                        st.markdown(f"<a href='{st.session_state.coldstart_df['link'][i*5+j]}' style='color:#ffffff;text-decoration:none'><img src='{st.session_state.coldstart_df['image'][i*5+j]}' style='width:auto;height:auto;max-width:100%;' /></a>", unsafe_allow_html=True)
                        st.markdown(f"<div class='auto-resize-text'>{st.session_state.coldstart_df['title'][i*5+j]}</div>", unsafe_allow_html=True, help=st.session_state.coldstart_df['description'][i*5+j])
                        st.checkbox("Select", key=itunes_id, label_visibility="hidden")
            
            submit_selections = st.form_submit_button('Next', on_click=tally_selection, use_container_width=True) # Button to proceed

    if 'selections' in st.session_state and not st.session_state.ratings:
        # Clear out previous container
        placeholder.empty()
        # Create another container that can be emptied later
        ratings = st.empty()
        
        # Function that stores ratings of selected podcasts upon clicking Next button
        def tally_ratings():
            scores = []
            for podcast in st.session_state['selections']:
                scores.append(st.session_state[f"rating_{podcast}"])
            st.session_state.scores = scores
            st.session_state.ratings = True
        
        # Display in a grid the selected podcasts so that user can rate them
        num_selected_podcasts = len(st.session_state['selections'])

        with ratings.form(key='rating'):
            st.subheader("Now, rate the podcasts that you've selected:")
            rows = 1 #if num_selected_podcasts <= 5 else 2
            cols = min(5, num_selected_podcasts)
            grid = make_grid(rows, cols)
            idx=0
            for i in range(rows):
                for j in range(cols):
                    with grid[i][j]:
                        if idx < num_selected_podcasts:
                            itunes_id = st.session_state['selections'][i*5+j]
                            selection = st.session_state.coldstart_df[st.session_state.coldstart_df['itunes_id']==itunes_id]
                            st.markdown(f"<a href='{selection['link'].values[0]}' style='color:#ffffff;text-decoration:none'><img src='{selection['image'].values[0]}' style='width:auto;height:auto;max-width:100%;' /></a>", unsafe_allow_html=True)
                            st.markdown(f"<div class='auto-resize-text'>{selection['title'].values[0]}</div>", unsafe_allow_html=True, help=selection['description'].values[0])
                            # Create star rating widget for each selected podcast
                            st_star_rating("", maxValue=5, defaultValue=0, key=f"rating_{itunes_id}")
                            idx += 1
                        else: 
                            break
            submit_ratings = st.form_submit_button('Next', on_click=tally_ratings, use_container_width=True) # Button to proceed
    
    if 'scores' in st.session_state and not st.session_state.recs:
        # Clear out previous container
        placeholder.empty()
        ratings.empty()
        # Create another container that can be emptied later        
        recs = st.empty()
        
        # Function that resets Streamlit session variables so that user can try again
        def reset_session():
            recs.empty()
            st.session_state.coldstart = False
            st.session_state.ratings = False
            del st.session_state.selections
            del st.session_state.scores
            del st.session_state.coldstart_df
            del st.session_state.name
        
        with recs.form(key='recommendation'):
            # Get new dataframe with user and user preferences added
            podcast_dfnew = append_new_user(st.session_state.name, st.session_state['selections'], st.session_state['scores'], podcast_df)
            # Obtain recommended podcasts
            _, podcastrecs = generate_recommendations(st.session_state.name, sim_matrix, podcast_dfnew)

            st.subheader("Based on your selections, we recommend the following podcasts:")
            
            # Display recommended podcasts in a grid
            grid = make_grid(2,5)
            for i in range(2):
                for j in range(5):
                    with grid[i][j]:
                        itunes_id = podcastrecs[i*5+j][0]
                        podcast_rec = podcast_df[podcast_df['itunes_id']==itunes_id]
                        st.markdown(f"<a href='{podcast_rec['link'].values[0]}' style='color:#ffffff;text-decoration:none'><img src='{podcast_rec['image'].values[0]}' style='width:auto;height:auto;max-width:100%;' /></a>", unsafe_allow_html=True)
                        st.markdown(f"<div class='auto-resize-text'>{podcast_rec['title'].values[0]}</div>", unsafe_allow_html=True, help=podcast_rec['description'].values[0])
            
            reset_button = st.form_submit_button('Start Over', on_click=reset_session, use_container_width=True) # Button to start over


if __name__ == '__main__':
    main()