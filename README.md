# Recommender System for Podcasts
## Project for AIPI540
## Team members: Shen, Chad, Zenan

>![img.jpg](assets/bestdspods.jpg)

## Background
Podcasts are a great way to stay informed, become educated on a variety of topics, or enjoy entertaining content. With millions of podcasts in the Apple podcast store, there are an overwheling number of podcasts to choose from, and finding the right content can be challenging. Our project employs three types of models to recommend content: a hybrid system, a content-based filtering system, and a non-ML approach in TF-IDF term seach. 

## Installation instructions

To prepare for the code in the repo, first install the required packages using:

```
pip install -r requirements.txt
```

The streamlit demo can be run by

## Dataset
The dataset is comprised of data that we scraped from the itunes podcast website using a method similar to that in https://github.com/siddgood/podcast-recommendation-engine except also scraping user ratings, image url, and and episode description data. In a separate method (in notebooks/getting_more_podcast_reviews2.ipynb) we obtained additional user-rating pairs. The data collected was from the group of URLs listing popular podcasts under each genre, with a limited number of user-ratings-reviews. The final dataset is comprised of 46,711 user-rating pairs corresponding to 3,936 unique podcasts, which have a full set of the following data as columns in the dataframe:

- Title: Title of the podcast
- Producer: Producer of the podcast
- Genre: 19 unique genres, including
    a. 'Leisure'
    b. 'True Crime'
    c.  'Business'
    d.  'Education'
    e.  'Society & Culture'
    f.  'Government'
    g.  'Health & Fitness'
    h.  'Sports'
    i.  'Kids & Family'
    j.  'Science'
    k.  'TV & Film'
    l.  'Comedy'
    m.  'Technology'
    n.  'Fiction'
    o.  'History'
    p.  'Religion & Spirituality'
    q.  'News'
    r.  'Arts'
    s.  'Music'
- Description: A text description of the overall podcast information
- Num_episodes: Number of podcast episodes
- Avg_rating: Average rating a on a scale of 1-5
- Num_reviews: Number of ratings contributing to the average rating
- Link: URL for the podcast
- Itunes_id: A unique identifier from the Apple podcast site, used to merge the review information
- Episode Descriptions: Summaries of individual podcast episodes, usually the last 5
- Rating: Rating by an individual user on a scale from 1-5
- User: Username for the reviewer

For the ratings, the data was heavily skewed towards high ratings:
    

>![img.png](assets/user_ratings_podcasts.png)

Of note, we initially planned to use a large Kaggle dataset consisting of about 75K podcasts and over 5.6 million reviews in two separate json files, but the podcast_id identifiers did not match between the two (only about 30 common ids between the two sets of data), so we could not use this for the hybrid model we initially planned. Instead, we scraped the much smaller dataset ourselves as described above. 

## Non-ML approach: TF-IDF




## Content-based filtering
We performed content-based filtering on podcast features in the podcast dataset. In particular, we used text embeddings of the 'genre', 'description', and 'episode_descriptions' columns to create similarity matrices using cosine similarity. The text from the description and the 'episode_descriptions' columns was first processed to exclude url links and special characters and other standard text preprocessing steps such striping white spaces and changing to all lower case. The embeddings were performed using pre-trained sentence transformers model using Siamese-BERT at HuggingFace (model 'all-MiniLM-L6-v2'). Different combinations of the features (embedded text from the 3 columns) were investigated and the best results were obtained with the combination of the genre and overall podcast description. Performance is not as high as we would like, with a RMSE of 1.48 for predicting the rating (scale 1-5), possibly due to the small number of podcasts and ratings and the skewed distribution of user ratings.  


## Hybrid model



