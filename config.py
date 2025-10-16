# Training Params

LABELS = {
    'negative': 0,
    'neutral': 1,
    'positive': 2,
}

NUM_RECS_TRAIN = 1000

NUM_RECS_TEST_VAL = 250

BASE_PATH = 'models/finetuned_model'

MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

HF_REPO_DIR = 'DiSabatino/mlops-sentiment-model'
