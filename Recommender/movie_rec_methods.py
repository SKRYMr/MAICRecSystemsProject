from django_pandas.io import read_frame
from .models import Movie

def tqdm_recommendations(movie_id: int):
    target_movie = Movie.objects.get(movie_id=movie_id)
    rec_ids = target_movie.tmdb_recommendations.replace("]", "").replace("[", "").split(", ")
    rec_ids = [int(i) for i in rec_ids]
    rec_movies = Movie.objects.filter(tmdb_id__in=rec_ids)
    df_movies = read_frame(rec_movies)
    return df_movies.to_dict("records")