import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import os
import nltk


# Download NLTK packages as Streamlit cloud VM does not have them
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


# Define constants
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_BANNER = 'assets/podcast_neon_banner.jpeg'


if __name__ == '__main__':
    st.set_page_config(page_title='Podcast Recommender 🎙️')
    st.title('Podcast Recommender 🎙️')
    st.image(os.path.join(_CURRENT_DIR, _BANNER))
    st.markdown('This web app demonstrates different recommendation systems for podcasts based on keyword similarity and content filtering.')
    
    st.markdown("### Data")
    st.write("")
    
    st.markdown("#### Content-based Filtering")
    st.write("")
    
    st.markdown("#### Keyword Similarity")
    st.write("")
    
    st.markdown("##### Click on the pages in the menu to try out the recommendation systems.")