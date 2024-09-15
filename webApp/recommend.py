import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
    credits = pd.read_csv('dataset/tmdb_5000_credits.csv')
    return movies, credits

# Data preprocessing
def preprocess_data(movies, credits):
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # Extract director from crew
    def get_director(x):
        crew = json.loads(x)
        for i in crew:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    movies['director'] = movies['crew'].apply(get_director)
    
    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        if feature == 'director':
            movies[feature] = movies[feature].apply(lambda x: [x] if isinstance(x, str) else [])
        else:
            movies[feature] = movies[feature].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
        movies[feature] = movies[feature].apply(lambda x: [i['name'] for i in x] if feature != 'director' else x)
        movies[feature] = movies[feature].apply(clean_data)
    
    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        movies[feature] = movies[feature].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    movies['soup'] = movies['overview'] + ' ' + movies['cast'] + ' ' + movies['director'] + ' ' + movies['keywords'] + ' ' + movies['genres']
    
    return movies

# Create count matrix and cosine similarity matrix
@st.cache_resource
def create_similarity_matrix(movies):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim2

# Function to get movie recommendations
def get_recommendations(title, cosine_sim, movies):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Function to fetch movie poster
@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=bd6327eeb3471344006290c45690438f&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path

def Recommend():
    st.title('Movie Recommender System')
    
    # Load and preprocess data
    movies, credits = load_data()
    movies = preprocess_data(movies, credits)
    cosine_sim = create_similarity_matrix(movies)
    
    # User input
    movie_list = movies['title'].tolist()
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
    
    if st.button('Show Recommendation'):
        recommended_movies = get_recommendations(selected_movie, cosine_sim, movies)
        st.write("Recommended Movies:")
        
        cols = st.columns(5)
        for i, (_, movie) in enumerate(recommended_movies.iterrows()):
            with cols[i % 5]:
                st.text(movie['title'])
                poster_url = fetch_poster(movie['movie_id'])
                st.image(poster_url, width=150)
