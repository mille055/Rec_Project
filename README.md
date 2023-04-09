# Recommender System for Podcasts
## Project for AIPI540
## Team members: Shen, Chad, Zenan

>![img.jpg](assets/bestdspods.jpg)

## Background
Podcasts are a great way to stay informed, become educated on a variety of topics, or enjoy entertaining content. With millions of podcasts in the Apple podcast store, there are an overwheling number of podcasts to choose from, and finding the right content can be challenging. Our project uses a hybrid recommender system to help find the right content for you....

## Installation instructions

To prepare for the code in the repo, first install the required modules using:

```
pip install -r requirements.txt
```

The streamlit demo can be run by

## Dataset
The dataset is comprised of data that I scraped from the itunes podcast website (using a method similar to that in https://github.com/siddgood/podcast-recommendation-engine except also scraping user ratings and episode description data. In a separate method (in notebooks/getting_more_podcast_reviews2.ipynb) I was able to obtain additional user-rating pairs. The data collected was from the group of URLs listing popular podcasts under each genre, with a limited number of user-ratings-reviews. The final dataset is comprised of 46,711 user-rating pairs corresponding to 3,936 unique podcasts, which have a full set of the following data as columns in the dataframe:

Title: Title of the podcast
Producer: Producer of the podcast
Genre: 19 unique genres, including 
    Leisure','True Crime', 'Business', 'Education', 'Society & Culture', 'Government', 'Health & Fitness','Sports', 'Kids & Family', 'Science', 'TV & Film', 'Comedy', 'Technology', 'Fiction','History', 'Religion & Spirituality', 'News', 'Arts', 'Music'

For the ratings, the data was heavily skewed towards high ratings:
    5    35835
    4     2237
    3     2351
    2     1955
    1     4333

>![img.png](assets/user_ratings_podcasts.png)
