import ast
import functools

import numpy as np
import pandas as pd
import random
import re
import requests
import time
from bs4 import BeautifulSoup
from django_pandas.io import read_frame
from enum import Enum
from typing import Set, Literal, Union, List

from .core import find_k_nearest, get_top_movies, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS
from .parse import preprocess_pipeline, clean_pipeline, vectorize
from .models import Movie
from django.template.defaulttags import register
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
SAFE_AGE_RATING = AgeRating["PG-13"]


def get_ratings_up_to(target_rating: str) -> List[str]:
    target_rating = AgeRating[target_rating.upper()].value
    tolerance = 0
    if target_rating > SAFE_AGE_RATING.value:
        tolerance = 1
    valid_ratings = []
    for rating, value in AgeRatingsDict.items():
        if value > target_rating + tolerance:
            break
        valid_ratings.append(rating)
    return valid_ratings


def get_movies_recommendations(user_id: int, users: Set[int], ratings: pd.DataFrame, movies: pd.DataFrame,
                               minimum_ratings: int = 5,
                               metric: Literal["neighbours_average", "bias_correction"] = "neighbours_average"):
    neighbours, similarities = find_k_nearest(user_id, users, ratings, NEIGHBOURHOOD_SIZE, MINIMUM_COMMON_RATINGS)
    return get_top_movies(user_id, neighbours, ratings, movies, similarities, minimum_ratings, metric)


def compute_synopsis_vec(text: str) -> Union[np.array, None]:
    preprocessed_text = preprocess_pipeline(clean_pipeline(text))
    vec = vectorize(preprocessed_text)
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


def compare_age_rating(age_rating: str, target_rating: str) -> bool:
    target_rating = AgeRating[target_rating.upper()].value
    if (not age_rating or age_rating not in AgeRatingsDict) and target_rating <= SAFE_AGE_RATING.value:
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


def format_gpt_response(content: str) -> dict:
    # Get the last 20 lines which is hopefuly the recommendations
    movie_lines = content.split("\n")[-10:]

    # Extract movie titles from each line, get rid of the number.
    # movie_titles = [line.split('. ', 1)[1].strip() for line in movie_lines]
    movie_titles = [line.split('. ', 1)[1].strip() for line in movie_lines if '. ' in line]

    # Filter out movies not present in the dataset and get their queryset
    movie_queryset = Movie.objects.filter(title__in=movie_titles)

    # Convert the queryset to a DataFrame
    df_movies = read_frame(movie_queryset)

    return df_movies


def compute_similarity(x, reference, r_len):
    if pd.isna(x):
        return None

    try:
        x = ast.literal_eval(x)
    except ValueError:
        return None

    val = (2 * len(set(x).intersection(reference))) / (r_len + len(x))
    return val


def compute_similarity_actors(x, reference):
    if pd.isna(x):
        return None

    try:
        x = ast.literal_eval(x)
    except ValueError:
        return None

    actors_overlap = len(set(x).intersection(reference))
    if actors_overlap == 0:
        result = 0.05
    else:
        result = actors_overlap / 10
    return result


def evaluate_recommendations(recommendations, ground_truth):
    recs = {}
    ground_truth = ast.literal_eval(ground_truth)
    if len(ground_truth) == 0:
        print("No ground truth available")

    print("Evaluation:")
    # Get tmdb id lists of recommendations
    for key, reclist in recommendations.items():

        if key != "TQDM Recommendations":
            recs[key] = []
            for rec in reclist:
                recs[key].append(rec["tmdb_id"])

    precisions = {}
    # Compute precision by comparing how many movie ids appear in the ground truth
    for key, reclist in recs.items():
        overlap_len = len(set(reclist).intersection(ground_truth))
        precisions[key] = overlap_len / len(reclist) if len(reclist) > 0 else 0

    return precisions


def scrape_imdb_poster(imdb_link: str) -> str:
    imdb_base_url = "https://www.imdb.com"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
               "Accept-Language": "en-US,en;q=0.5"}
    bs = BeautifulSoup(requests.get(imdb_link, headers=headers).content, "lxml")
    poster_page = bs.find("a",
                          attrs={"class": "ipc-lockup-overlay", "aria-label": re.compile("View .* Poster")})["href"]
    bs = BeautifulSoup(requests.get(imdb_base_url + poster_page, headers=headers).content, "lxml")
    poster_link = bs.find("img", attrs={"data-image-id": re.compile(".*-curr")})["src"]
    return poster_link


def posters_decorator(func):

    @functools.wraps(func)
    def check_posters(*args, **kwargs):
        recommendations = func(*args, **kwargs)
        if not recommendations:
            return recommendations
        start = time.time()
        for recommendation in recommendations:
            if not recommendation.get("imdb_poster") and recommendation.get("tmdb_id"):
                # TODO: Change functions to also return movie_id in the recommendations
                # Due to how pandas' DataFrame.to_dict("records") works, we don't have access
                # to the movie_id field because it's used as the index of the dataframe and thus
                # not included in the records dictionary. Only a handful of items has a missing
                # tmdb_id but there are some duplicates. Will change in the future.
                # movie = Movie.objects.get(movie_id=recommendation["movie_id"])
                movie = Movie.objects.filter(tmdb_id=recommendation["tmdb_id"]).first()
                if movie.imdb_link:
                    poster_link = scrape_imdb_poster(movie.imdb_link)
                    movie.imdb_poster = poster_link
                    movie.save()
                    recommendation["imdb_poster"] = poster_link
        print(f"Checking for posters took {round(time.time() - start, 3)}")
        return recommendations

    return check_posters


def timing_decorator(func: callable):

    @functools.wraps(func)
    def timeit(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} took {round(time.time() - start, 3)}")
        return result

    return timeit


@register.filter
def getitem(dictionary, key):
    return dictionary.get(key)


@register.filter
def percentage(value):
    return round(value * 100)
