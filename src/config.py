LABELS = {
    'negative': 0,
    'neutral': 1,
    'positive': 2,
}

NUM_RECS_TRAIN = 2000

NUM_RECS_TEST_VAL = 250

BATCH_SIZE = 8

TRAIN_EPOCHS = 20

DATASET_NAME = 'tweet_eval'

BASE_PATH = 'models/finetuned_model'

MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

HF_REPO_DIR = 'DiSabatino/mlops-sentiment-model'

REFERENCE_PATH = 'data/train.csv'

CURRENT_PATH = 'data/predictions.csv'

REPORT_PATH = 'reports/monitoring_report.html'

METRICS_JSON = 'reports/metrics_summary.json'

REF_STATS_FILE = 'monitoring/ref_stats.json'

OUTPUT_REPORT = 'monitoring/no_label_monitoring_report.json'
