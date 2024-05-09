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
        if not os.path.isfile(RecommenderConfig.FASTTEXT_MODEL_FILE):
            fasttext.util.download_model("en", if_exists="ignore")
        else:
            print("FastText model file already up-to-date.")
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("omw-1.4")
        RecommenderConfig.first_startup = False
