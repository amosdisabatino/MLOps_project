LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

STR_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

NUM_RECS_TRAIN = 2000

NUM_RECS_TEST_VAL = 250

BATCH_SIZE = 8

TRAIN_EPOCHS = 20

DATASET_NAME = "tweet_eval"

BASE_PATH = "models/finetuned_model"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

HF_REPO_DIR = "DiSabatino/mlops-sentiment-model"

CURRENT_PATH = "data/predictions.csv"

LOCK_PATH = "data/predictions.lock"

REF_STATS_FILE = "monitoring/ref_stats.json"

OUTPUT_REPORT = "monitoring/no_label_monitoring_report.json"

METRICS_PATH = "metrics"
