from django.db import models

# Create your models here.


class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=500)
    director = models.CharField(max_length=100, null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    actors = models.TextField(null=True, blank=True)
    synopsis = models.TextField(max_length=5000, null=True, blank=True)
    genres = models.TextField()
    poster = models.CharField(max_length=100, null=True, blank=True)

    def get_base_url(self):
        """
        Returns the base url for the movie posters, just append whatever is in the "poster" field.
        """
        return "https://image.tmdb.org/t/p/original"


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
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=RATINGS)
