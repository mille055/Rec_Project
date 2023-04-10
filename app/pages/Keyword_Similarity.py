import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_PODCAST_DATA = '../../data/podcast_df_tokens_040723.pkl'


def generate_vector(vocabulary, tfidf, tokens):
    """
    Creates a vector representing the tokens in a string by using the TF-IDF vectorizer.
    
    Args:
        vocabulary(List(str)): List of words representing vocabulary of podcast description
        tfidf(TfidfVectorizer): TF-IDF vectorizer fitted with description tokens of podcast dataset
        tokens(): 
    
    Returns:
        query_vec(np.ndarray): Numpy array representing the vector of a string of tokens
    """
    query_vec = np.zeros((len(vocabulary)))    
    x = tfidf.transform(tokens)
    for token in tokens[0].split():
        try:
            idx = vocabulary.index(token)
            query_vec[idx]  = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return query_vec


def calculate_cos_sim(v1, v2):
    """
    Calculates the cosine similarity between two given vectors - v1 and v2.

    Args:
        v1(np.ndarray): Numpy array of a vector
        v2(np.ndarray): Numpy array of a vector
    
    Returns:
        cos_sim(float): A floating point representing the cosine similarity between v1 and v2
    """
    cos_sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos_sim


def generate_tfidf(df):
    """
    Generates a TF-IDF weighted document-term matrix from the 'description' field of the podcast dataset.
    
    Args:
        df(pd.DataFrame): Pandas dataframe containing the podcast dataset

    Returns:
        vocabulary(List(str)): List of words representing vocabulary of podcast description
        tfidf(TfidfVectorizer): TF-IDF vectorizer fitted with description tokens of podcast dataset
        tfidf_matrix(np.ndarray): TF-IDF weighted document-term matrix
    """
    # Create vocabulary set
    vocabulary = set()
    for tokens in df['description_tokens']:
        vocabulary.update(tokens.split())
    vocabulary = list(vocabulary)
    # Instanstiate TF-IDF model and fit tokens to it
    tfidf = TfidfVectorizer(vocabulary=vocabulary)
    tfidf.fit(df['description_tokens'])
    # Generate TF-IDF matrix
    tfidf_matrix = tfidf.transform(df['description_tokens'])
    return vocabulary, tfidf, tfidf_matrix


def get_recommendations(vocabulary, tfidf, tfidf_matrix, df_podcast, query, k=10):
    """
    Pre-processes user query to get a vector, obtains cosine similarity between query vector and existing entries in TF-IDF matrix, and recommends a list of podcasts based on highest cosine similarity scores.
    Reference: https://medium.com/analytics-vidhya/build-your-semantic-document-search-engine-with-tf-idf-and-google-use-c836bf5f27fb.
    
    Args:
        vocabulary(List(str)): List of words representing vocabulary of podcast description
        tfidf(TfidfVectorizer): TF-IDF vectorizer fitted with description tokens of podcast dataset
        tfidf_matrix(np.ndarray): TF-IDF weighted document-term matrix
        df_podcast(pd.DataFrame): Pandas dataframe containing the podcast dataset
        query(str): Query string from user
        k(int): Number of recommendations to return, defaults to 10

    Returns:
        recs(pd.DataFrame): Pandas Dataframe containing the recommended podcasts
    """
    # Remove non-alpanumeric characters, convert to lowercase, and remove white space at both ends
    preprocessed_query = re.sub("\W+", " ", query).lower().strip()
    # Tokenize query
    tokens = word_tokenize(str(preprocessed_query))
    # Create a new dataframe to store query tokens
    query_df = pd.DataFrame(columns=['query_tokens'])
    query_df.loc[0, 'query_tokens'] = tokens
    # Instantiate stopwords and lemmatizer 
    english_stopwords = stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()
    # Remove stop words, lemmatize words, and join them in a string
    query_df['query_tokens'] = query_df['query_tokens'].apply(lambda x: [word for word in x if word not in english_stopwords])
    query_df['query_tokens'] = query_df['query_tokens'].apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])
    query_df['query_tokens'] = query_df['query_tokens'].apply(lambda x: " ".join(x))
    # Initialize array to store cosine similarities
    cos_sim_arr = []
    # Get vector for prepocessed query tokens
    query_vector = generate_vector(vocabulary, tfidf, query_df['query_tokens'])
    # Calculate cosine similarities between query vector and all other vectors in TFIDF matrix
    for v in tfidf_matrix.A:
        cos_sim_arr.append(calculate_cos_sim(query_vector, v))
    # Get indices of highest to lowest cosine similarity scores and keep aside first
    output = np.array(cos_sim_arr).argsort()[-k:][::-1]
    # Sort cosine similarity scores
    cos_sim_arr.sort()
    # Create new dataframe to store recommendations
    recs = pd.DataFrame()
    # Enumerate through recommendations and get all the required fields
    for i, index in enumerate(output):
        recs.loc[i,'title'] = df_podcast['title'][index]
        recs.loc[i,'description'] = df_podcast['description'][index]
        recs.loc[i,'image'] = df_podcast['image'][index]
        recs.loc[i,'link'] = df_podcast['link'][index]
    # Store the similarity scores as well
    for i, sim_score in enumerate(cos_sim_arr[-k:][::-1]):
        recs.loc[i,'score'] = sim_score
    return recs


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
    # Load pickled podcast dataframe
    df_podcast = pd.read_pickle(os.path.join(_CURRENT_DIR, _PODCAST_DATA))
    # Generate vocabulary, TF-IDF model, and TF-IDF matrix
    vocabulary, tfidf, tfidf_matrix = generate_tfidf(df_podcast)
    
    st.set_page_config(page_title='Podcast Recommender 🎙️', layout='wide')
    st.title('Podcast Recommender 🎙️')
    st.markdown('Enter some keywords related to what you want to learn about and we will recommend some podcasts to listen to!')

    with st.form(key='my_form'):
        query = st.text_input('What interests you?') # Input text box for keywords
        submit = st.form_submit_button('Surprise me!') # Submit button
    
    if submit and query is not None:
        with st.spinner('Generating recommendations...'):
            # Retrieve recommendations
            results = get_recommendations(vocabulary, tfidf, tfidf_matrix, df_podcast, query)
            st.subheader('Try these!')
            # Display recommendations in grid form
            grid = make_grid(2,5)
            for i in range(2):
                for j in range(5):
                    with grid[i][j]:
                        st.markdown(f"<a href='{results['link'][i*5+j]}' style='color:#ffffff;text-decoration:none'><img src='{results['image'][i*5+j]}' style='width:auto;height:auto;max-width:100%;' /></a>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:18px;text-align:center;'>{results['title'][i*5+j]}</p>", unsafe_allow_html=True, help=results['description'][i*5+j])


if __name__ == '__main__':
    main()