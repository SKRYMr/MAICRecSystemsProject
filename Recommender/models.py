from django.db import models

# Create your models here.


class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=100)
    director = models.CharField(max_length=100)
    year = models.IntegerField()
    actors = models.JSONField(default=list)     # Needs to be a function that returns a fresh object each time
    synopsis = models.TextField(max_length=5000)
    genres = models.JSONField(default=list)


class User(models.Model):
    user_id = models.IntegerField(primary_key=True)


class Rating(models.Model):
    RATINGS = {
        1: "Very Bad",
        2: "Bad",
        3: "Neutral",
        4: "Good",
        5: "Very Good"
    }
    movie_id = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=RATINGS)
