# app.py
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
import json
import os
import sys

# Initialize Spark
@st.cache_resource
def init_spark():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    return SparkSession.builder.appName("Movie Recommender").getOrCreate()

# Load and preprocess data
@st.cache_data
def load_data():
    movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
    credits = pd.read_csv('dataset/tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    return preprocess_data(movies)

def preprocess_data(movies):
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    
    def extract_names(genre_string):
        genre_list = json.loads(genre_string)
        return [item['name'] for item in genre_list]
    
    def extract_director(crew_string):
        crew_list = json.loads(crew_string)
        return [item['name'] for item in crew_list if item['job']=='Director']
    
    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['cast'] = movies['cast'].apply(extract_names)
    movies['crew'] = movies['crew'].apply(extract_director)
    
    return movies

# Create recommender system
def create_recommender(spark, movie_df):
    for feature in ['cast', 'keywords', 'crew', 'genres']:
        movie_df = movie_df.withColumn(
            feature,
            when(size(col(feature)) > 3, slice(col(feature), 1, 3))
            .otherwise(col(feature))
        )
    
    movie_df = movie_df.withColumn(
        "tags",
        lower(
            concat_ws(
                " ",
                array_join(col("keywords"), " "),
                array_join(col("cast"), " "),
                array_join(col("crew"), " "),
                array_join(col("genres"), " ")
            )
        )
    )
    
    tokenizer = Tokenizer(inputCol="tags", outputCol="words")
    countVectorizer = CountVectorizer(inputCol="words", outputCol="features",
                                    vocabSize=20000, minDF=1.0)
    
    pipeline = Pipeline(stages=[tokenizer, countVectorizer])
    model = pipeline.fit(movie_df)
    
    return model.transform(movie_df)

def get_recommendations(title, feature_df, n=10):
    @udf(returnType=FloatType())
    def cosine_similarity_vectors(v1, v2):
        v1_array = v1.toArray()
        v2_array = v2.toArray()
        dot_product = float(v1_array.dot(v2_array))
        norm1 = float(np.sqrt(v1_array.dot(v1_array)))
        norm2 = float(np.sqrt(v2_array.dot(v2_array)))
        return float(dot_product / (norm1 * norm2)) if norm1 * norm2 != 0 else 0.
    
    input_vector = feature_df.filter(col("title") == title).first()
    if not input_vector:
        return None
        
    recommendations = feature_df.crossJoin(
        broadcast(feature_df.filter(col("title") == title)
        .select("features")
        .withColumnRenamed("features", "input_features"))
    )
    
    recommendations = recommendations.withColumn(
        "similarity",
        cosine_similarity_vectors(col("features"), col("input_features"))
    )
    
    return recommendations.filter(col("title") != title) \
                         .orderBy(col("similarity").desc()) \
                         .select("title", "similarity") \
                         .limit(n)

# Streamlit UI
def Recommend():
    st.title("Movie Recommendation System")
    
    # Initialize Spark
    spark = init_spark()
    
    # Load data
    movies = load_data()
    movie_df = spark.createDataFrame(movies)
    
    # Create feature DataFrame
    feature_df = create_recommender(spark, movie_df)
    
    # Movie selection
    movie_titles = movies['title'].tolist()
    selected_movie = st.selectbox("Select a movie:", movie_titles)
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_movie, feature_df)
        if recommendations:
            st.subheader("Recommended Movies:")
            
            # Convert to pandas for easier display
            rec_pd = recommendations.toPandas()
            for _, row in rec_pd.iterrows():
                st.write(f"{row['title']} (Similarity: {row['similarity']:.2f})")
        else:
            st.error("Movie not found in database.")
