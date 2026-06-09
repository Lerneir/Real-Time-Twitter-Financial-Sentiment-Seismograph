import re
import nltk

class VADERClassifier:
    """
    Wrapper around NLTK VADER Sentiment Intensity Analyzer.
    Ensures the VADER lexicon is downloaded and initialized.
    """
    _sia = None

    @classmethod
    def get_sia(cls):
        if cls._sia is None:
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            cls._sia = SentimentIntensityAnalyzer()
        return cls._sia

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text by:
    1. Removing URLs (http/https/www)
    2. Removing User Handles (@username)
    3. Retaining letters, numbers, spaces, and dollar sign ($) for stock tickers
    4. Lowercasing
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove User Handles
    text = re.sub(r"@\w+", "", text)
    # Remove all except alphanumeric characters, spaces, and dollar signs ($)
    text = re.sub(r"[^a-zA-Z0-9\s\$]", "", text)
    return text.lower().strip()

def analyze_sentiment(text: str) -> dict:
    """
    Cleans text and scores sentiment using VADER.
    Returns a dictionary of scores: positive, negative, neutral, compound.
    """
    cleaned = clean_tweet_text(text)
    sia = VADERClassifier.get_sia()
    scores = sia.polarity_scores(cleaned)
    return {
        "positive": float(scores['pos']),
        "negative": float(scores['neg']),
        "neutral": float(scores['neu']),
        "compound": float(scores['compound'])
    }
