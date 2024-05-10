import pandas as pd
import pickle
from scipy import spatial
from sortedcontainers import SortedList
from django_pandas.io import read_frame
from django.contrib.auth.decorators import user_passes_test
from django.db import transaction
from django.shortcuts import render
from .extract_data import extract_data, GOOGLE_DRIVE_ROOT
from .utils import get_movies_recommendations, compute_synopsis_vec, format_movie_recommendations
from .models import User, Movie, Rating


def index(request):
    return render(request, "index.html")


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def compute_synopsis_vecs(request):
    movies = Movie.objects.all()
    count = 0
    total = len(movies)
    for movie in movies:
        synopsis = movie.synopsis
        if synopsis is None:
            continue
        movie.synopsis_vec = compute_synopsis_vec(synopsis).dumps().hex()
        movie.save()
        count += 1
    return render(request, "success.html", {"context": {"total": total, "count": count}})


@user_passes_test(lambda u: u.is_superuser)
@transaction.atomic
def extract_drive_data(request):
    total, created, updated = extract_data(GOOGLE_DRIVE_ROOT)
    print(f"Total: {total}, Created: {created}, Updated: {updated}")
    return render(request, "success.html", {"context": {"total": total, "created": created, "updated": updated}})


def recommend_item(request, movie_id: int = 0):
    if request.method == "POST":
        movie_id = request.POST.get("movie_id")
    target_movie = Movie.objects.get(movie_id=movie_id)
    movie_title = target_movie.title
    synopsis_vec = pickle.loads(bytes.fromhex(target_movie.synopsis_vec))
    all_movies_with_vecs = Movie.objects.filter(synopsis_vec__isnull=False)
    top_scores = SortedList(key=lambda x: -x[0])
    if synopsis_vec is None:
        synopsis = target_movie.synopsis
        if synopsis is None:
            return render(request, "error.html", {"error": "No synopsis available for movie"})
        synopsis_vec = compute_synopsis_vec(synopsis)
        target_movie.synopsis_vec = synopsis_vec.dumps()
        target_movie.save()
    for movie in all_movies_with_vecs:
        other_vec = pickle.loads(bytes.fromhex(movie.synopsis_vec))
        similarity = round(1 - spatial.distance.cosine(synopsis_vec, other_vec), 4)
        if len(top_scores) < 5 or similarity > top_scores[-1][0]:
            top_scores.add((similarity, movie.movie_id))
    top_scores = top_scores[:5]
    recommended_movies = read_frame(Movie.objects.filter(movie_id__in=[movie_id for _, movie_id in top_scores])).set_index("movie_id")
    recommended_movies.loc[[movie_id for _, movie_id in top_scores], "rating"] = [score * 5 for score, _ in top_scores]
    recommended_movies = format_movie_recommendations(recommended_movies.sort_values("rating", ascending=False))
    return render(request, "recommendations.html",
                  {"recommendations": recommended_movies.to_dict("records"),
                   "movie_id": movie_id,
                   "movie_title": movie_title})


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

