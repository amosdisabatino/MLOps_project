from fastapi import FastAPI
from pydantic import BaseModel
from src.model import analyze_sentiment

app = FastAPI(title="Sentiment Analysis API")

class TextInput(BaseModel):
    text: str

@app.post('/predict')
def predict(review: TextInput):
    return analyze_sentiment(review.text)
