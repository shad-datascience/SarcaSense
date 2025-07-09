import pandas as pd
import numpy as np
import string
import re
# from ast import literal_eval
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline,FeatureUnion
import keras

# from sentence_transformers import SentenceTransformer


def to_lower(text):
    """Converts text to lowercase."""
    return text.lower()

exclude = string.punctuation
def removePunctuation(text):
    return text.translate(str.maketrans('','',exclude))


file_Path = './stop_hinglish.txt'

# Download NLTK stop words (if not already downloaded)
nltk.download("stopwords")

# Load Hinglish stop words from file
def load_stop_words(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        stop_words = set(word.strip().lower() for word in file.readlines())  # Normalize words
    return stop_words


hinglish_stop_words = load_stop_words("./stop_hinglish.txt")

# Load English stop words from NLTK
english_stop_words = set(stopwords.words("english"))

# Combine both stop words lists
all_stop_words = hinglish_stop_words.union(english_stop_words)

# Function to remove stop words (Hinglish + English)
def remove_stop_words(text):
    if isinstance(text, str):  # Ensure input is a string
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in all_stop_words]
        return " ".join(filtered_words)
    return text  # Return original if not a string (handles NaN values)

slang_dict = {
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
    "afaik": "as far as I know",
    "bcoz": "because",
    "frnd": "friend",
    "yaar": "friend",
    "mast": "awesome",
    "jhakaas": "superb",
    "sahi": "great",
    "bhai": "brother",
    "bro": "brother",
    "dost": "friend",
    "paka": "sure",
    "nai": "no",
    "koi nahi": "no one",
    "jldi": "jaldi",
    "aalsi": "lazy",
    "pakka": "sure",
    "biryani": "amazing",
    "scene hai": "there is a situation",
    "tight": "intoxicated",
    "lag gaye": "we are in trouble",
    "fix hai": "it is certain",
    "chill maar": "relax",
    "rapchik": "cool",
    "fadu": "amazing",
    "senti": "emotional",
    "jhakkas": "amazing",
    "kadak": "strong",
    "bindaas": "carefree",
    "haanikarak": "dangerous",
    "kaand": "big trouble",
    "faltu": "useless",
    "bhasad": "mess",
    "mamu": "dude",
    "tera kya scene hai?": "what's your plan?",
    "lafda": "problem",
    "locha": "issue",
    "jumla": "false promise",
    "khopdi tod": "mind-blowing",
    "chep": "clingy person",
    "lukkha": "useless guy",
    "matlab": "meaning",
    "chalu": "smart",
    "bawaal": "chaotic",
    "att": "attitude",
    "op": "overpowered",
    "hatt": "move away",
    "sahi hai": "it's good",
    "lit": "amazing",
    "supari": "contract killing",
    "ragra": "beaten badly",
    "maal": "attractive person",
    "item": "hot girl",
    "pataka": "attractive girl",
    "set hai": "everything is fine",
    "chindi": "cheap",
    "beedu": "close friend",
    "kat gaya": "got tricked",
    "tatti": "bad",
    "bakwaas": "nonsense",
    "scene on hai": "things are happening",
    "scene off hai": "not happening",
    "fix hai": "certain",
    "trip maar": "enjoy",
    "chhapri": "wannabe",
    "bhaiya": "elder brother",
}

def expand_slang(text):
    words = text.split()
    expanded_words = [slang_dict.get(word.lower(), word) for word in words]  # Replace slang
    return " ".join(expanded_words)

def preprocessing(text):
    lower = to_lower(text)
    rem_punct = removePunctuation(lower)
    rem_stop = remove_stop_words(rem_punct)
    text = expand_slang(rem_stop)
    return text


xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_model = AutoModel.from_pretrained("xlm-roberta-base")

def get_xlmr_embedding(text):
    tokens = xlmr_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = xlmr_model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

#sentiment pipeline
sentiment_analyzer = VS()

def sentiments(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound']]
    #features = pandas.DataFrame(features)
    return features

def expand_ndarray_series(series):
    """Expands a pandas Series containing ndarray values into a DataFrame with separate columns."""
    array_data = np.vstack(series.values)
    expanded_columns = [f"feature_{i}" for i in range(array_data.shape[1])]
    return pd.DataFrame(array_data, columns=expanded_columns)



class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.apply(preprocessing)

class XLMREmbeddingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(get_xlmr_embedding)

class ExpandNDArrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return expand_ndarray_series(X)

pipeline1 = Pipeline([
    ('preprocessing', PreprocessingTransformer()),
    ('embedding', XLMREmbeddingTransformer()),
    ('expand', ExpandNDArrayTransformer())
])

import re
import nltk
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.apply(preprocessing)

class SentimentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(sentiments)

class ExpandNDArrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return expand_ndarray_series(X)

pipeline2 = Pipeline([
    ('preprocessing', PreprocessingTransformer()),
    ('sentiments',SentimentTransformer()),
    ('expand', ExpandNDArrayTransformer())
])


# Concatenation of both pipelines
class DataFrameConcatenator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.concat(X, axis=1)

# Defining the feature processing pipeline
feature_pipeline = FeatureUnion([
    ("pipeline1", pipeline1),
    ("pipeline2", pipeline2)
])

# res = feature_pipeline.transform("Hello!! Darling what are you doing!!")

# model = keras.models.load_model("./mlp_model.keras")

# print(model.predict(res))
