import pandas as pd
import requests

# Get the poster images urls. SInce the old datasets was outdated
def change_poster_url():
    movies2 = pd.read_csv("./data/movies_ext.csv", usecols=["movie_id","title","director","actors","synopsis","genres","poster","year"])
    movies = pd.read_csv("./data/metadata_filtered.csv", usecols=["movie_id",'tmdbId', "poster","synopsis"])
    for index, row in movies.iterrows():
        tmdb_id = row['tmdbId']
        api_key = '347f057cc85997cb119a516c59c66063'
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&append_to_response=images"
        headers = {"accept": "application/json"}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Check if the request was successful and if there are any images available
        if response.status_code == 200 and 'images' in data:
            posters = data['images']['posters']
            
            if posters:
                poster_url = f"https://image.tmdb.org/t/p/original{posters[0]['file_path']}"
                
                movies.at[index, "poster"] = posters[0]['file_path']
                movies2.at[index, "poster"] = posters[0]['file_path']
                
                row["poster"] = posters[0]['file_path']
                
                print(f"Poster URL for movie_id {tmdb_id}: {poster_url}")
            else:
                print(f"No poster available for movie_id {tmdb_id}")
        else:
            print(f"Failed to fetch data for movie_id {tmdb_id}. Status code: {response.status_code}")
    
    movies.to_csv("./data/metadata_filtered.csv", index=False)
    movies2.to_csv("./data/movies_ext.csv", index=False)
        
change_poster_url()