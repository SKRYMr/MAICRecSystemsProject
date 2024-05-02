import pickle
import os

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
    ratings = read_frame(Rating.objects.all())
    try:
        user_predictions = get_movies_recommendations(request.POST["user_id"], users, ratings, movies,
                                                      minimum_ratings=3,
                                                      metric="neighbours_average")
    except FileNotFoundError as e:  # TODO: Replace with the correct error
        raise Http404("User does not exist")
    print(user_predictions)
    context = {"recommendations": list(user_predictions["title"])}
    print(context)
    return render(request, "recommendations.html", context)

