import os
import torch
import logging
from config import HF_REPO_DIR, DATASET_NAME, METRICS_PATH
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Evaluating the model...")

try:
    # Load tokenizer and model from HuggingFace
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_DIR)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_DIR)
    # Loading Test Dataset
    dataset = load_dataset(DATASET_NAME, "sentiment")
    test_data = dataset["test"]
except Exception:
    logger.info("Model or dataset not found.")
    exit()

os.makedirs(METRICS_PATH, exist_ok=True)

logger.info("Starting the evaluation of the model...")

texts = test_data["text"]
labels = test_data["label"]

# Evaluation the model
model.eval()
preds = []


with torch.no_grad():
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

# Compute Metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="weighted")
recall = recall_score(labels, preds, average="weighted")
f1 = f1_score(labels, preds, average="weighted")

# Display Metrics
logger.info("Metrics of the model:")
logger.info(f"Accuracy:  {accuracy:.3f}")
logger.info(f"Precision: {precision:.3f}")
logger.info(f"Recall:    {recall:.3f}")
logger.info(f"F1-score:  {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(labels, preds)
labels_text = list(model.config.id2label.values())

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_text,
    yticklabels=labels_text,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Save the confusion matrix plot
plt.tight_layout()
cf_matrix_path = os.path.join(METRICS_PATH, "confusion_matrix.png")
plt.savefig(cf_matrix_path)
logger.info(f"Confusion Metrics Saved In:{cf_matrix_path}")

# Save Results
results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
pd.DataFrame([results]).to_csv(
    f"{METRICS_PATH}/model_results.csv", index=False
)
logger.info(f"Results saved in {METRICS_PATH}/model_results.csv")
