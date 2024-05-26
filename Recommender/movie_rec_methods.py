from django_pandas.io import read_frame
from .models import Movie
import openai
from .utils import format_gpt_response

def tqdm_recommendations(movie_id: int):
    target_movie = Movie.objects.get(movie_id=movie_id)
    rec_ids = target_movie.tmdb_recommendations.replace("]", "").replace("[", "").split(", ")
    rec_ids = [int(i) for i in rec_ids]
    rec_movies = Movie.objects.filter(tmdb_id__in=rec_ids)
    df_movies = read_frame(rec_movies)
    return df_movies.to_dict("records")

def gpt_recommendations(movie_id: int):
    target_movie = Movie.objects.get(movie_id=movie_id)
    print(target_movie.title)
    
    client = openai.OpenAI()
    prompt = (
    f"Based on this movie: '{target_movie.title}', "
    "please provide 20 other movie recommendations."
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
    
    target_movie = Movie.objects.get(movie_id=movie_id)
    resp = format_gpt_response(recommendations)
    print(recommendations)
    for movie in resp:
        print(movie)
    return resp