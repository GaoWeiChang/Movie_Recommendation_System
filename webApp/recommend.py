import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
    credits = pd.read_csv('dataset/tmdb_5000_credits.csv')
    return movies, credits