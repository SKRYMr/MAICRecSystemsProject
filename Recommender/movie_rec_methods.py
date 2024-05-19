from django_pandas.io import read_frame
from .models import Movie
import random
import ast
from .utils import format_movie_recommendations, compute_similarity, compute_similarity_actors


def tqdm_recommendations(movie_id: int):
    target_movie = Movie.objects.get(movie_id=movie_id)
    rec_ids = target_movie.tmdb_recommendations.replace("]", "").replace("[", "").split(", ")
    rec_ids = [int(i) for i in rec_ids]
    rec_movies = Movie.objects.filter(tmdb_id__in=rec_ids)
    df_movies = read_frame(rec_movies)
    return df_movies.to_dict("records")


# noinspection PyPackageRequirements
def year_genre_recommend(movie_id: int,type: str = "keyword", parental_control: bool = True, year_proximity:int = 5,top_n: int = 5):
    """
    Year-Genre recommendations
    Given a movie, recommends a list of movies based on similar release_year, popularity, genre count and the final
    criterion which is specified in the parameter "type".
    :param movie_id: The movie ID to get recommendations for.
    :param type: Final parameter to consider in the recommendation. Can be one of: "keyword", "actors", "popularity"
    :param parental_control: Toggles a filter to not recommend movies for adult audiences if the target movie is for children
    :param year_proximity: The range of years from the reference movie to get recommendations for.
    :param top_n: The number of recommendations to return.
    """
    # get required target movie characteristics
    target_movie = Movie.objects.get(movie_id=movie_id)
    movie_year = target_movie.release_year
    # convert string to python list
    genres = ast.literal_eval(target_movie.genres)
    n_movie_genres = len(genres)

    #exclude the target movie from recommendations
    all_movies = Movie.objects.exclude(movie_id=movie_id)

    if parental_control:
        # Do not include movies of higher age_ratings (except PG if the movie is G)
        if target_movie.age_rating in ["G","PG"]:
            all_movies = all_movies.filter(age_rating__in=["G","PG"])
        elif target_movie.age_rating == "PG-13":
            all_movies = all_movies.filter(age_rating__in=["G","PG","PG-13"])
    # filter based on release date close to target movie
    close_years_movies = all_movies.filter(release_year__gte=movie_year-year_proximity, release_year__lte=movie_year+year_proximity)
    # convert to pandas dataframe
    close_years_movies_df = read_frame(close_years_movies)
    # compute the popularity of the movies
    close_years_movies_df['rating'] = close_years_movies_df['avg_ratings'] * (
                close_years_movies_df.num_ratings / close_years_movies_df.num_ratings.max())
    #include only first 600 most popular movies if there are more suitable
    if close_years_movies_df.shape[0] > 600:
        chosen_movies_genre_df = close_years_movies_df.sort_values("rating", ascending=False)[:600]
    else:
        chosen_movies_genre_df = close_years_movies_df

    # Compute genre similarity
    chosen_movies_genre_df["genre_similarity"] = chosen_movies_genre_df["genres"].apply(lambda x:compute_similarity(x,genres,n_movie_genres))
    gsim_column = chosen_movies_genre_df["genre_similarity"]
    # Fill the empty rows with mean genre similarity
    chosen_movies_genre_df["genre_similarity"]= gsim_column.fillna(gsim_column.mean())

    if type == "keyword":
        print("keyword recommendations")
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

    elif type == "actors":
        print("actors recommendations")
        # convert string to python list
        actors = ast.literal_eval(target_movie.actors)
        n_actors = len(actors)
        # similar to other similarity computation
        chosen_movies_genre_df["actors_similarity"] = chosen_movies_genre_df["actors"].apply(
            lambda x: compute_similarity_actors(x, actors))
        asim_column = chosen_movies_genre_df["actors_similarity"]
        chosen_movies_genre_df["actors_similarity"] = asim_column.fillna(asim_column.mean())
        # combine genre and actors similarity
        chosen_movies_genre_df["rating"] = chosen_movies_genre_df["actors_similarity"] * chosen_movies_genre_df["genre_similarity"]

    recommended_movies = format_movie_recommendations(chosen_movies_genre_df.sort_values("rating", ascending=False),round_to=2, top_n=top_n)
    return recommended_movies.to_dict("records")

