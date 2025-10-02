from transformers import pipeline

sentiment_pipeline = pipeline(
    'sentiment-analysis',
    model='cardiffnlp/twitter-roberta-base-sentiment-latest'
)

def analyze_sentiment(text: str) -> dict:
    """
    Test
    """
    return sentiment_pipeline(text)[0]
