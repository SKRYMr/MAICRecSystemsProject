from django.apps import AppConfig


class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Recommender'
    FASTTEXT_MODEL_LANGUAGE = "en"
    FASTTEXT_MODEL_FILE = "cc.en.300.bin"
    first_startup = True

    def ready(self):
        """
        This functions is called exactly once when the app is ready,
        before it starts serving requests.
        """
        if not RecommenderConfig.first_startup:
            return
        import fasttext.util
        import nltk
        import os
        import sqlite3
        con = sqlite3.connect("db.sqlite3")
        print(con.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
        print(con.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 250000))
        quiet = False
        if not os.path.isfile(RecommenderConfig.FASTTEXT_MODEL_FILE):
            fasttext.util.download_model("en", if_exists="ignore")
        else:
            print("FastText model file already up-to-date.")
            quiet = True
        nltk.download("stopwords", quiet=quiet)
        nltk.download("punkt", quiet=quiet)
        nltk.download("omw-1.4", quiet=quiet)
        nltk.download("names", quiet=quiet)
        RecommenderConfig.first_startup = False
