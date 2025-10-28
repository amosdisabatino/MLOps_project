from datetime import datetime
from src.model import analyze_sentiment
from src.api import main
from src.api.main import app, save_result_in_csv, TextInput
from fastapi.testclient import TestClient
import os
import threading
import csv
import time

client = TestClient(app)


def test_positive_sentence():
    review = (
        "Amazon UK MAN1 is a very well-run and efficient warehouse. The site "
        "is easy to reach with clear directions, and the check-in process is "
        "smooth and well organised. Security is handled professionally, which "
        "keeps everything running safely and on time."
        "Staff are friendly, approachable, and always willing to assist, "
        "which makes the experience much better. Parking and access are "
        "straightforward, and the facility is kept clean and well managed."
        "Overall, a professional and efficient site – great experience every "
        "time."
    )
    result = analyze_sentiment(review)
    assert "label" in result
    assert result["label"] == "positive"


def test_negative_sentence():
    review = (
        "Bunch of crooks at this place item arrived back there on the 28th "
        "March have proof of delivery with a signature yet Amazon customer "
        "service say the item hadn’t arrived back yet.Now chasing up a refund "
        "for an £899 google pixel while some tea leaf is still working there "
        "stealing other items."
    )
    result = analyze_sentiment(review)
    assert result["label"] == "negative"


def test_neutral_sentence():
    review = (
        "Who are going to this company for delivery tomorrow ?"
        "Who work here? Where is the warehouse located?"
    )
    result = analyze_sentiment(review)
    assert result["label"] == "neutral"


def test_empty_sentence():
    review = ""
    result = analyze_sentiment(review)
    assert result["label"] == "neutral"


def test_special_characters():
    review = (
        "This company is excellent and reliable, recommended to everyone "
        "👍​👍​👍​👍​👍​😃​😃​😃​!!!"
    )
    result = analyze_sentiment(review)
    assert result["label"] == "positive"


def test_api_endpoints(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    lock_path = f"{csv_path}.lock"
    main.CURRENT_PATH = str(csv_path)
    main.LOCK_PATH = str(lock_path)

    response = client.post("/predict", json={"text": "I love this company!"})
    assert response.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200

    response = client.get("/data")
    assert response.status_code == 200


def test_valid_prediction():
    results = analyze_sentiment("This company is great!")
    assert "label" in results
    assert results["label"] in ["negative", "neutral", "positive"]
    assert "confidence" in results
    assert isinstance(results["confidence"], float)
    assert 0.0 <= results["confidence"] <= 1.0


def test_invalid_input():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_file_locking(tmp_path):
    """
    This test simulates two concurrent writes and checks that the CSV is
    written correctly.
    """

    # Update the paths in the main module to use the temp test files

    csv_path = tmp_path / "predictions.csv"
    lock_path = f"{csv_path}.lock"
    main.CURRENT_PATH = str(csv_path)
    main.LOCK_PATH = str(lock_path)

    # Test results to be written
    result1 = {
        "Timestamp": datetime.now().isoformat(),
        "label": "positive",
        "confidence": 0.95,
    }
    result2 = {
        "Timestamp": datetime.now().isoformat(),
        "label": "negative",
        "confidence": 0.88,
    }

    review1 = TextInput(text="This company is good!")
    review2 = TextInput(text="This company is terrible!")

    # Creates two threads that write simultaneously
    t1 = threading.Thread(target=save_result_in_csv, args=(result1, review1))
    t2 = threading.Thread(target=save_result_in_csv, args=(result2, review2))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    time.sleep(0.3)

    assert os.path.exists(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 rows
        labels = [row[2] for row in rows[1:]]
        assert set(labels) == {"positive", "negative"}
