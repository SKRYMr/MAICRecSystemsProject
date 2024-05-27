import fasttext
import numpy as np
import re
from .apps import RecommenderConfig as config
from nltk.corpus import stopwords, names
from nltk.tokenize import word_tokenize
from typing import Union

# Remove common english stopwords and also names.
# Having names in the vectors make it so that movies with similarly-named protagonists
# get recommended together.
stopwords = set(stopwords.words("english")).union(set(names.words()))


# Do not initialize the model until the first model call for memory constraints.
# Using Singletons to guarantee that the model is only initialized once.
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FastText(metaclass=Singleton):

    def __init__(self):
        self.model = fasttext.load_model(config.FASTTEXT_MODEL_FILE)

    def __getitem__(self, item):
        return self.model[item]


def rm_link(text):
    return re.sub(r'http\S+', '', text)


# handle case like "shut up okay?Im only 10 years old"
# become "shut up okay Im only 10 years old"
def rm_punct2(text):
    # return re.sub(r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
    return re.sub(r'[\"#$%&\'()*+/:;<=>@\[\\\]^_`{|}~]', ' ', text)


def rm_html(text):
    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    # remove <br /> tags
    return re.sub(r'<br />', '', text)


def space_bt_punct(text):
    pattern = r'([.,!?-])'
    s = re.sub(pattern, r'', text)  # remove punctuation
    s = re.sub(r'\s{2,}', ' ', s)  # remove double whitespaces
    return s


def rm_number(text):
    return re.sub(r'\d+', '', text)


def rm_whitespaces(text):
    return re.sub(r'\s+', ' ', text)


def rm_nonascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)


def rm_emoji(text):
    emojis = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emojis.sub(r'', text)


def clean_pipeline(text):
    text = text.lower()
    no_link = rm_link(text)
    no_html = rm_html(no_link)
    space_punct = space_bt_punct(no_html)
    no_punct = rm_punct2(space_punct)
    no_number = rm_number(no_punct)
    no_whitespaces = rm_whitespaces(no_number)
    no_nonascii = rm_nonascii(no_whitespaces)
    no_emoji = rm_emoji(no_nonascii)
    return no_emoji


# preprocessing
def tokenize(text):
    return word_tokenize(text)


def rm_stopwords(text):
    return [i for i in text if i not in stopwords]


def preprocess_pipeline(text):
    tokens = tokenize(text)
    no_stopwords = rm_stopwords(tokens)
    return " ".join(no_stopwords)


def vectorize(text: str) -> Union[np.array, None]:
    ft = FastText()
    vec = [ft[word] for word in text.split()]
    vec = np.mean(vec, axis=0)
    if np.any(np.isnan(vec)):
        vec = None
    return vec
