import sqlite3

con = sqlite3.connect("db.sqlite3")
cur = con.cursor()

# Create table
# cur.execute("CREATE TABLE googledrivemovies (movie_id, tmdb_id, title, genres, age_ratings, avg_ratings, num_ratings, languages, actors, directors, writers, release_year, release_date, runtime, imdb_link, poster, youtube_trailer_video_ids, synopsis, tmdb_recommendations)")


def add_to_googledrivemovies(item):
    """
    Adds a new row to the googledrivemovies table.

    Parameters:
        item (tuple): A tuple containing all fields in the order they appear in the table schema.
    """
    query = """
    INSERT INTO googledrivemovies (
        movie_id, tmdb_id, title, genres, age_ratings, avg_ratings, num_ratings, languages, 
        actors, directors, writers, release_year, release_date, runtime, imdb_link, 
        poster, youtube_trailer_video_ids, synopsis, tmdb_recommendations
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        cur.execute(query, item)
        con.commit()
    except sqlite3.IntegrityError as e:
        print(f"An error occurred: {e}")
    except sqlite3.OperationalError as e:
        print(f"Operational error: {e}")


print(cur.execute("SELECT title FROM googledrivemovies").fetchall())