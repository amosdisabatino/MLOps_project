from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from src.model import analyze_sentiment
from datetime import datetime
from collections import Counter
from src.config import CURRENT_PATH

import csv
import os

app = FastAPI(title="Sentiment Analysis API")

os.makedirs("data", exist_ok=True)


class TextInput(BaseModel):
    text: str = Field(
        min_lenght=1,
        max_length=512,
    )

    @field_validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


@app.post("/predict")
def predict(review: TextInput):
    """
    This method to processes the data, sends it to the model and returns
    the classification of the review in a python dictionary, then the result
    is saved in a csv file.
    """
    result = analyze_sentiment(review.text)

    save_result_in_csv(result, review)

    return result


@app.get("/metrics")
def get_metrics():
    """
    This method compute the metrics for each category of the review ('positive'
    or 'negative'):
        - for each category compute the percentange of reviews on the total;
        - and the number of predictions;
    """
    if not os.path.exists(CURRENT_PATH):
        return {"message": "No Predictions Found."}

    labels = []
    with open(CURRENT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["Label"])

    counts = Counter(labels)
    total = sum(counts.values())

    metrics = {
        label: round(count / total * 100, 2) for label, count in counts.items()
    }

    return {
        "total_predictions": total,
        "distribution": metrics,
    }


@app.get("/data")
def read_csv():
    """
    This method displays all the predictions made by the model and saved in
    its `csv` file.
    """
    data = []
    if os.path.exists(CURRENT_PATH):
        with open(CURRENT_PATH, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data
    raise FileNotFoundError("No Predictions File Found.")


def save_result_in_csv(result, review):
    with open(CURRENT_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Timestamp", "Text", "Label", "Confidence"])
        writer.writerow(
            [
                datetime.now().isoformat(),
                review.text,
                result.get("label", ""),
                result.get("confidence", 0.0),
            ]
        )
