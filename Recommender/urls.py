from django.urls import path
from . import views

app_name = "Recommender"
urlpatterns = [
    path("", views.index, name="index"),
    path("recommend/user/", views.recommend_user, name="recommend_user"),
    path("recommend/movie/<int:movie_id>/", views.recommend_item, name="recommend_item"),
    path("recommend/movie/", views.recommend_item, name="recommend_item"),
    path("compute_vecs", views.compute_synopsis_vecs, name="compute_synopsis_vecs"),
]
