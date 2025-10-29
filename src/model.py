from src.config import HF_REPO_DIR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

try:
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_DIR)
except Exception as e:
    raise RuntimeError(
        f"Error loading model or tokenizer from {HF_REPO_DIR}: {e}"
    )


def analyze_sentiment(text: str) -> dict:
    """
    This method is used to predict the sentiment of the sentence received by
    the client in `main.py`.
    Returns a dictionary with the label of the sentiment predicted by the model
    and and the probability value for the prediction.

    :param: `text`: the sentence to classify.
    :type: `text`: `str`
    :return: `dict`
    """

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        labels = model.config.id2label
        label = labels[pred_id] if labels else str(pred_id)

        return {"label": label, "confidence": probs[0][pred_id].item()}

    except Exception as e:
        raise RuntimeError(f"Error during sentiment analysis: {e}")
