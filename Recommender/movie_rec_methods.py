from .utils import *

import ast
import openai
import pandas as pd
import pickle
from .core import BEST_STAR_RATINGS, MINIMUM_RATINGS_PERCENT, POPULARITY_PENALTY, POPULARITY_PENALTY_CENTRE
from .models import Movie, Rating
from django.db.models import Avg, Count, Max
from django_pandas.io import read_frame
from sortedcontainers import SortedList
from scipy import spatial
from typing import Literal, List


@posters_decorator
@timing_decorator
def tqdm_recommendations(movie_id: int):
    target_movie = Movie.objects.get(movie_id=movie_id)
    rec_ids = target_movie.tmdb_recommendations.replace("]", "").replace("[", "").split(", ")
    rec_ids = [int(i) for i in rec_ids]
    rec_movies = Movie.objects.filter(tmdb_id__in=rec_ids)
    df_movies = read_frame(rec_movies)
    recommendations = format_movie_recommendations(df_movies, top_n=5)
    return recommendations.to_dict("records")


@posters_decorator
@timing_decorator
def gpt_recommendations(movie_id: int, top_n: int = 5):
    """
    Given a movie, recommends a list of movies based on recommendations from ChatGPT(4)
    :param movie_id: The movie ID to get recommendations for.
    :param top_n: The number of recommendations to return.
    """
    # TODO: Investigate whether a faster library for http requests can improve performance.
    target_movie = Movie.objects.get(movie_id=movie_id)

    # INFO: OpenAI() client default api-key is given fetched from a local os environmental variable.
    # If you want to use it, enter your own api key. OpenAI(api_key=OPENAI_API_KEY)
    client = openai.OpenAI()
    prompt = (
    f"Based on this movie: '{target_movie.title}', "
    "please provide 10 other movie recommendations."
    "Please answer with correct movie titles, and make sure to not include TV-series."
)
    # Call the OpenAI API
    response = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
        ],
        model="gpt-4",
    )

    # Extract the recommendations from the response
    recommendations = response.choices[0].message.content
    # Get the last 20 lines which is hopefully the recommendations
    movie_lines = recommendations.split("\n")[-10:]

    # Extract movie titles from each line, get rid of the number, quotations etc
    movie_titles = [line.split('. ', 1)[1].strip().strip("'").strip('"') for line in movie_lines if '. ' in line]
    # Filter out movies not present in the dataset, to be able to show additonal data.
    movie_queryset = Movie.objects.filter(title__in=movie_titles)
    df_movies = read_frame(movie_queryset)

    # Make sure that the order is the same as given by the response from chatgpt.
    df_movies['title'] = pd.Categorical(df_movies['title'], categories=movie_titles, ordered=True)
    df_movies = df_movies.sort_values('title')
    recommendations = format_movie_recommendations(df_movies, top_n=top_n)
    
    return recommendations.to_dict("records")


@posters_decorator
@timing_decorator
# noinspection PyPackageRequirements
def year_genre_recommend(movie_id: int, metric: str = "keyword",
                         parental_control: bool = True, year_proximity: int = 5, top_n: int = 5):
    """
    Year-Genre recommendations
    Given a movie, recommends a list of movies based on similar release_year, popularity, genre count and the final
    criterion which is specified in the parameter "type".
    :param movie_id: The movie ID to get recommendations for.
    :param metric: Final parameter to consider in the recommendation. Can be one either "keyword" or "actors".
    :param parental_control: Toggles a filter to not recommend movies for adult audiences
    if the target movie is for children.
    :param year_proximity: The range of years from the reference movie to get recommendations for.
    :param top_n: The number of recommendations to return.
    """
    # get required target movie characteristics
    target_movie = Movie.objects.get(movie_id=movie_id)
    movie_year = target_movie.release_year
    # convert string to python list
    genres = ast.literal_eval(target_movie.genres)
    n_movie_genres = len(genres)

    # exclude the target movie from recommendations
    all_movies = Movie.objects.exclude(movie_id=movie_id)

    if parental_control:
        # Do not include movies of higher age_ratings
        all_movies = all_movies.filter(age_rating__in=get_ratings_up_to(target_movie.age_rating
                                                                        if target_movie.age_rating
                                                                        else SAFE_AGE_RATING.name))
    # filter based on release date close to target movie
    close_years_movies = all_movies.filter(release_year__gte=movie_year - year_proximity,
                                           release_year__lte=movie_year + year_proximity)
    # convert to pandas dataframe
    close_years_movies_df = read_frame(close_years_movies)
    # compute the popularity of the movies
    close_years_movies_df['rating'] = close_years_movies_df['avg_ratings'] * (
                close_years_movies_df.num_ratings / close_years_movies_df.num_ratings.max())
    # include only first 600 most popular movies if there are more suitable
    if close_years_movies_df.shape[0] > 600:
        chosen_movies_genre_df = close_years_movies_df.sort_values("rating", ascending=False)[:600]
    else:
        chosen_movies_genre_df = close_years_movies_df

    # Compute genre similarity
    chosen_movies_genre_df["genre_similarity"] = chosen_movies_genre_df["genres"].apply(
        lambda x: compute_similarity(x, genres, n_movie_genres)
    )
    gsim_column = chosen_movies_genre_df["genre_similarity"]
    # Fill the empty rows with mean genre similarity
    chosen_movies_genre_df["genre_similarity"] = gsim_column.fillna(gsim_column.mean())

    if metric == "keyword":
        # convert string to python list
        keywords = ast.literal_eval(target_movie.tmdb_keywords)
        n_keywords = len(keywords)
        # similar to genre similarity computation
        chosen_movies_genre_df["keyword_similarity"] = chosen_movies_genre_df["tmdb_keywords"].apply(
            lambda x: compute_similarity(x, keywords, n_keywords))
        ksim_column = chosen_movies_genre_df["keyword_similarity"]
        chosen_movies_genre_df["keyword_similarity"] = ksim_column.fillna(ksim_column.mean())
        # combine genre and actors similarity
        chosen_movies_genre_df["rating"] = chosen_movies_genre_df["keyword_similarity"] * chosen_movies_genre_df["genre_similarity"]

    elif metric == "actors":
        # convert string to python list
        actors = ast.literal_eval(target_movie.actors)
        # similar to other similarity computation
        chosen_movies_genre_df["actors_similarity"] = chosen_movies_genre_df["actors"].apply(
            lambda x: compute_similarity_actors(x, actors))
        asim_column = chosen_movies_genre_df["actors_similarity"]
        chosen_movies_genre_df["actors_similarity"] = asim_column.fillna(asim_column.mean())
        # combine genre and actors similarity
        chosen_movies_genre_df["rating"] = chosen_movies_genre_df["actors_similarity"] * chosen_movies_genre_df["genre_similarity"]

    recommended_movies = format_movie_recommendations(chosen_movies_genre_df.sort_values("rating", ascending=False),
                                                      round_to=2, top_n=top_n)
    return recommended_movies.to_dict("records")


@posters_decorator
@timing_decorator
def neighbours_recommend(movie_id: int, top_n: int = 5, pg: str = None, auto_pg: bool = True):
    """
    Given a movie, recommends a list of movies based on the average ratings of users that have rated the target movie
    5 stars, or 4 if not enough 5-star ratings exist and so on.
    :param movie_id: The movie ID to get recommendations for.
    :param top_n: The number of recommendations to return.
    :param pg: The minimum PG rating to filter recommendations on.
    :param auto_pg: Whether to automatically detect the pg rating to filter on based on the one from the target movie.
    """
    # - TODO: Precompute this for each movie because right now it's way too slow.
    # - TODO: Improve popularity penalties parameters with data analysis.
    target_movie = Movie.objects.get(movie_id=movie_id)
    if not pg and auto_pg:
        pg = target_movie.age_rating if target_movie.age_rating else SAFE_AGE_RATING.name
    best_star_ratings = None
    for val in sorted(list(Rating.RATINGS.keys()), reverse=True):
        best_star_ratings = best_star_ratings.union(Rating.objects.filter(movie_id=movie_id, rating=val)) \
            if best_star_ratings else Rating.objects.filter(movie_id=movie_id, rating=val)
        if best_star_ratings.count() >= BEST_STAR_RATINGS:
            break
    else:
        print("Not enough ratings available for target movie.")
        print(f"Available ratings: {best_star_ratings.count()}")
        # TODO: DONT FORGET TO CHANGE FILIP
        return  # render(request, "error.html", {"error": "Not enough ratings available for movie."})
    neighbours = best_star_ratings.values_list("user_id", flat=True)
    minimum_ratings = int(neighbours.count() * MINIMUM_RATINGS_PERCENT)
    neighbours_ratings = Rating.objects.filter(user_id__in=neighbours).exclude(movie_id=movie_id)
    if pg:
        neighbours_ratings = neighbours_ratings.filter(movie__age_rating__in=get_ratings_up_to(pg))
    neighbours_ratings = (neighbours_ratings.values("movie_id")
                          .annotate(avg_rating=Avg("rating"),
                                    ratings_count=Count("movie_id"))
                          .filter(ratings_count__gte=minimum_ratings).order_by("-avg_rating"))
    movie_ids = [movie["movie_id"] for movie in neighbours_ratings]
    movie_ratings = [movie["avg_rating"] for movie in neighbours_ratings]
    movie_ratings_count = [movie["ratings_count"] for movie in neighbours_ratings]
    recommended_movies = read_frame(
        Movie.objects.filter(movie_id__in=movie_ids)
    ).set_index("movie_id")
    max_ratings = Movie.objects.aggregate(max_ratings=Max("num_ratings"))["max_ratings"]
    recommended_movies["popularity"] = recommended_movies.num_ratings / max_ratings
    recommended_movies.loc[movie_ids, "pre_rating"] = movie_ratings
    recommended_movies["rating"] = (
            recommended_movies.pre_rating -
            POPULARITY_PENALTY * (POPULARITY_PENALTY_CENTRE - recommended_movies.popularity) ** 2
    )
    recommended_movies.loc[movie_ids, "ratings_count"] = movie_ratings_count
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    recommended_movies = format_movie_recommendations(recommended_movies.sort_values("rating", ascending=False),
                                                      round_to=2, top_n=top_n)
    return recommended_movies.to_dict("records")


@posters_decorator
@timing_decorator
def semantic_recommend(movie_id: int = 0,
                       metric: Literal["cosine", "euclidean"] = "cosine",
                       genres: List[str] = None,
                       pg: str = None,
                       popularity_penalty: bool = False,
                       auto_genres: bool = False,
                       auto_pg: bool = True,
                       top_n: int = 5):
    """
    Provides a list of movie recommendations based on semantic similarity between the movies synopsis' descriptions.
    :param movie_id: The movie ID to get recommendations for.
    :param metric: The similarity metric to use, options are "cosine" or "euclidean". Cosine is recommended.
    :param genres: A list of genres to filter on to make recommendations more focused.
    :param pg: The minimum PG rating to filter recommendations on.
    This avoids situations like recommending horror movies about possessed toys
    when looking for recommendations for Toy Story, for example. Strongly recommended.
    Options are: "G", "PG", "PG-13", "R", "NC-17"
    :param popularity_penalty: Whether to apply a penalty to similarity based on how popular the similar movie is.
    :param auto_genres: Whether to automatically detect the genres to filter on based on the ones from the target movie.
    :param auto_pg: Whether to automatically detect the pg rating to filter on based on the one from the target movie.
    :param top_n: The number of recommendations to return.
    """
    # - TODO: Add popularity penalty? -> Seems worse for now, re-evaluate after popularity penalty has been optimized
    # - TODO: Experiment with different embedding models.
    target_movie = Movie.objects.get(movie_id=movie_id)
    if not genres and auto_genres:
        genres = set(ast.literal_eval(target_movie.genres))
    if not pg and auto_pg:
        pg = target_movie.age_rating if target_movie.age_rating else SAFE_AGE_RATING.name
    synopsis_vec = pickle.loads(bytes.fromhex(target_movie.synopsis_vec))
    all_movies_with_vecs = Movie.objects.filter(synopsis_vec__isnull=False).exclude(movie_id=movie_id)
    top_scores = SortedList(key=lambda x: -x[0]) if metric == "cosine" else SortedList(key=lambda x: x[0])
    # If the target movie's synopsis vector hasn't been computed yet, compute it and add it to the DB immediately.
    if synopsis_vec is None:
        synopsis = target_movie.synopsis
        if synopsis is None:
            # TODO: @Filip FIX THIS!!!!
            return  # render(request, "error.html", {"error": "No synopsis available for movie"})
        synopsis_vec = compute_synopsis_vec(synopsis)
        target_movie.synopsis_vec = synopsis_vec.dumps()
        target_movie.save()
    # If an age rating has been provided for filtering, filter on that (first).
    if pg:
        all_movies_with_vecs = all_movies_with_vecs.filter(age_rating__in=get_ratings_up_to(pg))
    # Compute scores for movies based on chosen similarity metric.
    max_ratings = Movie.objects.aggregate(max_ratings=Max("num_ratings"))["max_ratings"]
    for movie in all_movies_with_vecs:
        # If a list of genres has been provided for filtering, filter on those genres.
        # This is equivalent to an OR filter.
        # Currently, if a movie doesn't have any genres recorded, we always include it.
        if genres and movie.genres and len(genres.intersection(set(ast.literal_eval(movie.genres)))) < 1:
            continue
        other_vec = pickle.loads(bytes.fromhex(movie.synopsis_vec))
        if metric == "cosine":
            similarity = round(1 - spatial.distance.cosine(synopsis_vec, other_vec), 4)
        else:
            similarity = round(spatial.distance.euclidean(synopsis_vec, other_vec), 4)
        if popularity_penalty:
            similarity -= (POPULARITY_PENALTY * (POPULARITY_PENALTY_CENTRE - (movie.num_ratings / max_ratings)) ** 2) / 5
        # We only care about keeping the first 5 entries so this is more memory efficient.
        if top_scores.bisect_right((similarity, movie.movie_id)) < top_n:
            top_scores.add((similarity, movie.movie_id))
    top_scores = top_scores[:top_n]
    recommended_movies = read_frame(
        Movie.objects.filter(movie_id__in=[movie_id for _, movie_id in top_scores])
    ).set_index("movie_id")
    # Multiply score by 5 to make it compatible with the stars in the frontend.
    # This only really works with cosine similarity.
    recommended_movies.loc[[movie_id for _, movie_id in top_scores], "rating"] = [score * 5 for score, _ in top_scores]
    recommended_movies = format_movie_recommendations(
        recommended_movies.sort_values("rating", ascending=False if metric == "cosine" else True), round_to=2
    )
    return recommended_movies.to_dict("records")
