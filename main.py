from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.linalg import Vectors, VectorUDT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import json

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize Spark
spark = SparkSession.builder.appName("CBR System").getOrCreate()


''' Load Dataset '''
movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')

''' Data Pre-Processing '''
# merge into single dataset
movies = movies.merge(credits, on='title')

# Keeping important columns for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)

# parse JSON
def extract_names(genre_string):
    # Parse the JSON string to a list of dictionaries
    genre_list = json.loads(genre_string)
    
    # Extract the names from the list of dictionaries
    return [item['name'] for item in genre_list]

# handle genre
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(extract_names)

def extract_director(crew_string):
    # Parse the JSON string to a list of dictionaries
    crew_list = json.loads(crew_string)
    
    # Extract the names from the list of dictionaries
    return [item['name'] for item in crew_list if item['job']=='Director']

movies['crew'] = movies['crew'].apply(extract_director)

# Convert pandas DataFrame to Spark DataFrame
movie_df = spark.createDataFrame(movies)

''' Build Recommendation System '''

'''
Plot description based recommender
    - We will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score. The plot description is given in the overview feature of our dataset
    - This approach is indeed based on computing the cosine similarity between the vector representations of movie overviews.
'''
def create_pb_recommender(movie_df):
    # Tokenize overview
    tokenizer = Tokenizer(inputCol="overview", outputCol="words")
    
    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # Convert words to term frequency vectors
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=20000)
    
    # Calculate IDF
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    
    # Create and fit the pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    model = pipeline.fit(movie_df)
    tfidf_df = model.transform(movie_df)
    
    return tfidf_df

# cosine similarity
def cosine_similarity_vectors(v1, v2):
    v1_array = v1.toArray()
    v2_array = v2.toArray()
    dot_product = float(v1_array.dot(v2_array))
    norm1 = float(np.sqrt(v1_array.dot(v1_array)))
    norm2 = float(np.sqrt(v2_array.dot(v2_array)))
    
    return float(dot_product / (norm1 * norm2)) if norm1 * norm2 != 0 else 0.0

def get_pb_recommendations(movie_title, feature_df, feature_col, top_n=10):
    # Get the feature vector for the input movie
    movie_features = feature_df.filter(col("title") == movie_title) \
                              .select(feature_col) \
                              .first()
    
    if not movie_features:
        return None
    
    # Create broadcast variable for movie features
    movie_vector_broadcast = spark.sparkContext.broadcast(movie_features[0])
    
    # Create UDF with broadcasted vector
    @udf(returnType=FloatType())
    def cosine_sim_with_broadcast(v1):
        return cosine_similarity_vectors(v1, movie_vector_broadcast.value)
    
    # Calculate similarities
    similarities = feature_df.withColumn(
        "similarity",
        cosine_sim_with_broadcast(col(feature_col))
    )
    
    # Get top N similar movies
    result = similarities.filter(col("title") != movie_title) \
                        .orderBy(col("similarity").desc()) \
                        .select("title", "similarity") \
                        .limit(top_n)
    
    # Clean up broadcast variable
    movie_vector_broadcast.unpersist()
    
    return result

# create recommender
tfidf_df = create_pb_recommender(movie_df)

# Plot-based recommendations
plot_recommendations = get_pb_recommendations("The Dark Knight Rises", tfidf_df, "tfidf_features")

plot_recommendations.show()

'''
Metadata-based Recommendations
    A recommender based on the following metadata: the 3 top actors, the director, related genres and the movie plot keywords.
    Pre-process data for Metadata-based Recommendations
        - Since there are many genres, actor/actress and keyword in each row, we will choose top 3 most related related items for each column.
        - This will improve recommendation quality, reduces noise and enhances computation efficiency
'''


'''
convert the names and keyword instances into lowercase and strip all the spaces between them. 
This is done so that our vectorizer doesn't count the Johnny of "Johnny Depp" and "Johnny Galecki" as the same.
'''

# Define UDF for cleaning array/list data
@udf(returnType=ArrayType(StringType()))
def clean_array_data(x):
    if x is None:
        return []
    return [str(i).lower().replace(" ", "") for i in x]

# Define UDF for cleaning string data
@udf(returnType=StringType())
def clean_string_data(x):
    if x is None:
        return ''
    return str(x).lower().replace(" ", "")

# Clean the features
features = ['cast', 'keywords', 'crew', 'genres']

# Apply cleaning to each feature
for feature in features:
    # Check if the column is an array type
    if isinstance(movie_df.schema[feature].dataType, ArrayType):
        movie_df = movie_df.withColumn(feature, clean_array_data(col(feature)))
    else:
        movie_df = movie_df.withColumn(feature, clean_string_data(col(feature)))

# select the top 3
features = ['cast', 'keywords', 'genres']

for feature in features:
    # Using slice function is more efficient than UDF
    movie_df = movie_df.withColumn(
        feature,
        # When array size > 3, take first 3 elements, otherwise keep all
        when(size(col(feature)) > 3,
             slice(col(feature), 1, 3))
        .otherwise(col(feature))
    )

# Create tags that that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).
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

# Perform Metadata-based Recommendations
@udf(returnType=FloatType())
def cosine_similarity_vectors(v1, v2):
    v1_array = v1.toArray()
    v2_array = v2.toArray()
    dot_product = float(v1_array.dot(v2_array))
    norm1 = float(np.sqrt(v1_array.dot(v1_array)))
    norm2 = float(np.sqrt(v2_array.dot(v2_array)))
    
    return float(dot_product / (norm1 * norm2)) if norm1 * norm2 != 0 else 0.

def create_metadata_based_recommender(movie_df):
    # Create pipeline for count vectorization
    tokenizer = Tokenizer(inputCol="tags", outputCol="words")
    countVectorizer = CountVectorizer(inputCol="words", 
                                    outputCol="features",
                                    vocabSize=20000,
                                    minDF=1.0)
    
    # Create and fit pipeline
    pipeline = Pipeline(stages=[tokenizer, countVectorizer])
    model = pipeline.fit(movie_df)
    features_df = model.transform(movie_df)
    
    return features_df, model

def get_recommendations(title, feature_df, n=10):
    # Get the feature vector for the input movie
    input_vector = feature_df.filter(col("title") == title).first()
    
    if not input_vector:
        return None
    
    # Create a cross join with the input vector's features
    recommendations = feature_df.crossJoin(
        broadcast(feature_df.filter(col("title") == title)
        .select("features")
        .withColumnRenamed("features", "input_features"))
    )
    
    # Calculate similarities
    recommendations = recommendations.withColumn(
        "similarity",
        cosine_similarity_vectors(col("features"), col("input_features"))
    )
    
    # Get top N recommendations
    return recommendations.filter(col("title") != title) \
                         .orderBy(col("similarity").desc()) \
                         .select("title", "similarity") \
                         .limit(n)

feature_df, model = create_metadata_based_recommender(movie_df)
meta_recommendations = get_recommendations("The Dark Knight Rises", feature_df)
meta_recommendations.show()

spark.stop()