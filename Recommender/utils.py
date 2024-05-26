import numpy as np
import pandas as pd
from django_pandas.io import read_frame
import random
from enum import Enum
from typing import Set, Literal, Union
import ast


from .core import find_k_nearest, get_top_movies, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS
from .parse import preprocess_pipeline, clean_pipeline #, ft
from .models import Movie
from RecSystems5.settings import DEBUG

POSITIVE_RATING_THRESHOLD = 3

if DEBUG:
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# Age Ratings defined by mpaa + TV parental guidelines
# https://en.wikipedia.org/wiki/Motion_Picture_Association_film_rating_system
# https://en.wikipedia.org/wiki/TV_Parental_Guidelines
AgeRatingsDict = {"G": 1,
                  "TVG": 1, "TV-G": 1,
                  "TVY": 1, "TV-Y": 1,
                  "PG": 2,
                  "TVPG": 2, "TV-PG": 2,
                  "TVY7": 2, "TV-Y7": 2,
                  "PG13": 3, "PG-13": 3,
                  "TV14": 4, "TV-14": 4,
                  "R": 5,
                  "TVMA": 5, "TV-MA": 5,
                  "NR": 5, "UR": 5, "NOT RATED": 5, "UNRATED": 5,
                  "NC17": 6, "NC-17": 6}
AgeRating = Enum("AgeRating", AgeRatingsDict)
# This is used to determine whether it's safe to recommend a movie that doesn't have a record for the age rating.
# If the requested age rating is higher than this value, then the movie will be recommended regardless.
SAFE_AGE_RATING = AgeRating["PG-13"].value


def get_movies_recommendations(user_id: int, users: Set[int], ratings: pd.DataFrame, movies: pd.DataFrame,
                               minimum_ratings: int = 5,
                               metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average"):
    neighbours, similarities = find_k_nearest(user_id, users, ratings, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS)
    return get_top_movies(user_id, neighbours, ratings, movies, similarities, minimum_ratings, metric)


def compute_synopsis_vec(text: str) -> Union[np.array, None]:
    preprocessed_text = preprocess_pipeline(clean_pipeline(text))
    vec = [ft[word] for word in preprocessed_text.split()]
    vec = np.mean(vec, axis=0)
    if np.any(np.isnan(vec)):
        vec = None
    return vec


def format_movie_recommendations(recommendations: pd.DataFrame, top_n: int = 0, round_to: int = 1) -> pd.DataFrame:
    recommendations["actors"] = recommendations["actors"].apply(
        lambda x: x.replace("[", "").replace("(", "").replace("'", "").replace("]", "").replace(")", "").split(", ") if x else ""
    )
    recommendations["directors"] = recommendations["directors"].apply(
        lambda x: x.replace("[", "").replace("(", "").replace("'", "").replace("]", "").replace(")", "").split(", ") if x else ""
    )
    recommendations["genres"] = recommendations["genres"].apply(
        lambda x: x.replace("[", "").replace("]", "").replace("'", "").split(", ") if x else ""
    )
    recommendations["poster"] = recommendations["poster"].apply(lambda x: Movie.get_base_url() + x if x else "")
    if "rating" in recommendations.columns:
        recommendations["rating"] = recommendations["rating"].apply(lambda x: round(x, round_to))
    return recommendations.head(top_n) if top_n > 0 else recommendations


def compare_age_rating(age_rating: str, target_rating: str):
    target_rating = AgeRating[target_rating.upper()].value
    if (not age_rating or age_rating not in AgeRatingsDict) and target_rating <= SAFE_AGE_RATING:
        return False
    elif not age_rating or age_rating not in AgeRatingsDict:
        return True
    try:
        age_rating = AgeRating[age_rating.upper()].value
    except KeyError as e:
        print("Error in compare_age_rating:")
        print(e)
        print(f"Age Rating: {age_rating}")
        print(f"Target Rating: {target_rating}")
        return False
    return age_rating <= target_rating

def format_gpt_response(content:str) -> dict:
    #get the last 20 lines which is hopefuly the recommendations
    movie_lines = content.split("\n")[-10:]

    #extract movie titles from each line, get rid of the number.
    #movie_titles = [line.split('. ', 1)[1].strip() for line in movie_lines]
    movie_titles = [line.split('. ', 1)[1].strip() for line in movie_lines if '. ' in line]

    # Filter out movies not present in the dataset and get their queryset
    movie_queryset = Movie.objects.filter(title__in=movie_titles)

    # Convert the queryset to a DataFrame
    df_movies = read_frame(movie_queryset)

    return df_movies

def compute_similarity(x,reference,r_len):
    if pd.isna(x):
        return None

    try:
        x = ast.literal_eval(x)
    except ValueError:
        return None

    val = (2 * len(set(x).intersection(reference))) / (r_len + len(x))
    #print(set(x).intersection(reference))
    return val


def compute_similarity_actors(x,reference):
    result = 0
    if pd.isna(x):
        return None

    try:
        x = ast.literal_eval(x)
    except ValueError:
        return None

    actors_overlap = len(set(x).intersection(reference))
    #print(set(x).intersection(reference))
    #print(actors_overlap)
    if actors_overlap == 0:
        result = 0.05
    else:
        result = actors_overlap / 10
    return result

