import pandas as pd
from django.http import Http404
from django_pandas.io import read_frame
from django.shortcuts import render
from .main import get_movies_recommendations
from .models import User, Movie, Rating
# Create your views here.


def index(request):
    return render(request, "index.html")


def recommend(request):
    users = set(User.objects.values_list("user_id", flat=True))
    movies = read_frame(Movie.objects.all())
    ratings = pd.DataFrame(list(Rating.objects.values_list("movie_id", "user_id", "rating")),
                           columns=["movie_id", "user_id", "rating"])
    try:
        user_predictions = get_movies_recommendations(request.POST["user_id"], users, ratings, movies,
                                                      minimum_ratings=3,
                                                      metric="neighbours_average")
    except FileNotFoundError as e:  # TODO: Replace with the correct error
        raise Http404("User does not exist")
    context = {"recommendations": list(user_predictions["title"])}
    return render(request, "recommendations.html", context)

