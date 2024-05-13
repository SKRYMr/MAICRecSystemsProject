import pandas as pd
import pickle

from scipy import spatial
from sortedcontainers import SortedList
from django_pandas.io import read_frame
from django.contrib.auth.decorators import user_passes_test
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render
from django.forms.models import model_to_dict
from .extract_data import extract_data, GOOGLE_DRIVE_ROOT
from .utils import get_movies_recommendations, compute_synopsis_vec, format_movie_recommendations, compare_age_rating
from .models import User, Movie, Rating
from .movie_rec_methods import tqdm_recommendations
from typing import Literal, List


def index(request):
    return render(request, "index.html")


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def compute_synopsis_vecs(request):
    force = request.GET.get("force", True)
    movies = Movie.objects.all() if force else Movie.objects.filter(synopsis_vec__isnull=True)
    count = 0
    total = len(movies)
    for movie in movies:
        synopsis = movie.synopsis
        if synopsis is None:
            continue
        movie.synopsis_vec = compute_synopsis_vec(synopsis)
        if movie.synopsis_vec is not None:
            movie.synopsis_vec = movie.synopsis_vec.dumps().hex()
        movie.save()
        count += 1
    return render(request, "success.html", {"context": {"total": total, "count": count}})


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def extract_drive_data(request):
    total, created, updated = extract_data(GOOGLE_DRIVE_ROOT)
    print(f"Total: {total}, Created: {created}, Updated: {updated}")
    return render(request, "success.html", {"context": {"total": total, "created": created, "updated": updated}})


def recommend_item(request, movie_id: int = 0,
                   metric: Literal["cosine", "euclidean"] = "cosine",
                   genres: List[str] = None,
                   pg: str = None,
                   top_n: int = 5):
    if request.method == "POST":
        try:
            movie_id = request.POST.get("movie_id")
            genres = request.POST.get("genres", None)
            pg = request.POST.get("pg", None)
            top_n = request.POST.get("top_n", 5)
        except KeyError:
            return render(request, "error.html", {"error": "Movie ID is required"})
        metric = request.POST.get("metric", "cosine")
    if request.method == "GET":
        metric = request.GET.get("metric", "cosine")
        genres = request.GET.getlist("genres", None)
        pg = request.GET.get("pg", None)
        top_n = request.GET.get("top_n", 5)
    target_movie = Movie.objects.get(movie_id=movie_id)
    movie_title = target_movie.title
    synopsis_vec = pickle.loads(bytes.fromhex(target_movie.synopsis_vec))
    all_movies_with_vecs = Movie.objects.filter(synopsis_vec__isnull=False).exclude(movie_id=movie_id)
    top_scores = SortedList(key=lambda x: -x[0]) if metric == "cosine" else SortedList(key=lambda x: x[0])
    # If the target movie's synopsis vector hasn't been computed yet, compute it and add it to the DB immediately.
    if synopsis_vec is None:
        synopsis = target_movie.synopsis
        if synopsis is None:
            return render(request, "error.html", {"error": "No synopsis available for movie"})
        synopsis_vec = compute_synopsis_vec(synopsis)
        target_movie.synopsis_vec = synopsis_vec.dumps()
        target_movie.save()
    # If an age rating has been provided for filtering, filter on that (first).
    if pg:
        exclusion_ids = set()
        for movie in all_movies_with_vecs:
            if not compare_age_rating(movie.age_rating, pg):
                exclusion_ids.add(movie.movie_id)
        all_movies_with_vecs = all_movies_with_vecs.exclude(movie_id__in=exclusion_ids)
    # If a list of genres has been provided for filtering, filter on those genres.
    if genres:
        for genre in genres:
            all_movies_with_vecs = all_movies_with_vecs.filter(genres__icontains=genre)
    # Compute scores for movies based on chosen similarity metric.
    for movie in all_movies_with_vecs:
        other_vec = pickle.loads(bytes.fromhex(movie.synopsis_vec))
        if metric == "cosine":
            similarity = round(1 - spatial.distance.cosine(synopsis_vec, other_vec), 4)
        else:
            similarity = round(spatial.distance.euclidean(synopsis_vec, other_vec), 4)
        # We only care about keeping the first 5 entries so this is more memory efficient.
        if top_scores.bisect_right((similarity, movie.movie_id)) < top_n:
            top_scores.add((similarity, movie.movie_id))
    top_scores = top_scores[:top_n]
    recommended_movies = read_frame(
        Movie.objects.filter(movie_id__in=[movie_id for _, movie_id in top_scores])).set_index("movie_id")
    # Multiply score by 5 to make it compatible with the stars in the frontend.
    # This only really works with cosine similarity.
    recommended_movies.loc[[movie_id for _, movie_id in top_scores], "rating"] = [score * 5 for score, _ in top_scores]
    recommended_movies = format_movie_recommendations(recommended_movies.sort_values("rating", ascending=False if metric == "cosine" else True), round_to=4)
    return render(request, "recommendations.html",
                  {"recommendations": recommended_movies.to_dict("records"),
                   "movie_id": movie_id,
                   "movie_title": movie_title,
                   "metric": "similarity"})


def recommend_user(request):
    users = set(User.objects.values_list("user_id", flat=True))
    movies = read_frame(Movie.objects.all())
    ratings = pd.DataFrame(list(Rating.objects.values_list("movie_id", "user_id", "rating")),
                           columns=["movie_id", "user_id", "rating"])

    try:
        # Check if user_id is in users
        if int(request.POST["user_id"]) not in users:
            context = {"error": f"User: {request.POST['user_id']} not found"}
            return render(request, "error.html", context)
    except ValueError:
        # user_id is not a number
        context = {"error": f"User id must be a number"}
        return render(request, "error.html", context)
    except Exception as e:
        print(e)
        context = {"error": f"An unexpected error has occurred"}
        return render(request, "error.html", context)

    try:
        user_predictions = get_movies_recommendations(request.POST["user_id"], users, ratings, movies,
                                                      minimum_ratings=3,
                                                      metric="neighbours_average")

        user_predictions = format_movie_recommendations(user_predictions, 20)

        context = {"recommendations": user_predictions.to_dict('records'), "user_id": request.POST['user_id']}
        return render(request, "recommendations.html", context)

    except Exception as e:
        print(e)
        context = {"error": f"An unexpected error has occurred"}
        return render(request, "error.html", context)


def search(request):
    return render(request, "search.html")


def search_db(request):
    # Columns to keep
    items = ["movie_id", "title", "poster"]

    search_term = request.GET["query"]
    offset = int(request.GET["offset"])

    if search_term:
        movies = Movie.objects.filter(title__icontains=search_term)
    else:
        movies = Movie.objects.all()

    movies = movies[offset:offset + 50]
    movies = read_frame(movies)
    movies = movies[items]

    return JsonResponse({"movies": movies.to_dict('records')})


# For 5 different methods of recommendations
# Add methods in movie_rec_methods.py
def movie_recommendations(request):
    if request.method == "POST":
        movie_id = request.POST["movie_id"]

        # Convert this to dict and add to context
        target_movie = Movie.objects.get(movie_id=movie_id)

        context = {
            # "target_movie": "",
            "recommendations": {
                "TQDM Recommendations": tqdm_recommendations(movie_id)
            }
        }

        return render(request, "movie_recommendations.html", context)
    else:
        context = {"error": f"Not a post request"}
        return render(request, "error.html", context)