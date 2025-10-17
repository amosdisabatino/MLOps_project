import json
import logging
import os
import torch
from src.config import (
    LABELS, NUM_RECS_TRAIN, NUM_RECS_TEST_VAL, MODEL_NAME, HF_REPO_DIR
)
from datasets import load_dataset, concatenate_datasets
from evaluate import load
from huggingface_hub import upload_folder
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Fine Tuning Process")

logger.info("Loading IMDB Dataset...")

dataset = load_dataset('tweet_eval', 'sentiment')


logger.info("Loading Model...")

# This part of code is used to download the pre-trainded model from
# `huggingface` and its `tokenizer`, that it is used to convert the words in
# numeric data, so that the model can work with them.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=config.num_labels,
    ignore_mismatched_sizes=True
)

train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']

def get_data_by_labels(datas, data_type, num_rec):
    """
    This method is used to get for each label (0, 1, 2) the right data from the
    original dataset.
    :param `datas`: the origin datasets (`train` or `test`)
    :param `data_type`: a int variable used to search the data that have the
    same value in the `label` field.
    :param `num_rec`: the number of records to return.
    """
    return datas.filter(
        lambda data: data['label'] == data_type
    ).select(range(num_rec))

train_negative = get_data_by_labels(
    train_data, LABELS.get('negative'), NUM_RECS_TRAIN
)
train_neutral = get_data_by_labels(
    train_data, LABELS.get('neutral'), NUM_RECS_TRAIN
)
train_positive = get_data_by_labels(
    train_data, LABELS.get('positive'), NUM_RECS_TRAIN
)
test_negative = get_data_by_labels(
    test_data, LABELS.get('negative'), NUM_RECS_TEST_VAL
)
test_neutral = get_data_by_labels(
    test_data, LABELS.get('neutral'), NUM_RECS_TEST_VAL
)
test_positive = get_data_by_labels(
    test_data, LABELS.get('positive'), NUM_RECS_TEST_VAL
)
val_negative = get_data_by_labels(
    val_data, LABELS.get('negative'), NUM_RECS_TEST_VAL
)
val_neutral = get_data_by_labels(
    val_data, LABELS.get('neutral'), NUM_RECS_TEST_VAL
)
val_positive = get_data_by_labels(
    val_data, LABELS.get('positive'), NUM_RECS_TEST_VAL
)


train_set = concatenate_datasets([
    train_negative, train_neutral, train_positive
])
test_set = concatenate_datasets([test_negative, test_neutral, test_positive])
val_set = concatenate_datasets([val_negative, val_neutral, val_positive])

train_set = train_set.shuffle(42)
test_set = test_set.shuffle(42)
val_set = val_set.shuffle(42)

def tokenize_function(datas):
    """
    This method converts the `text` of each record in the `datas` dataset in a
    numerical data, so that the model can work with them.
    """
    return tokenizer(
        datas['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

tokenized_train = train_set.map(tokenize_function, batched=True)
tokenized_test = test_set.map(tokenize_function, batched=True)
tokenized_val = val_set.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy='epoch',
)

accuracy = load('accuracy')

def compute_metrics(eval_pred):
    """
    This method returns the metrics (E.X: `eval_accuracy` or `eval_loss`)
    computed after the training/testing of the model.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
)

logger.info("Start The Training Of The Model...")
train_result = trainer.train(resume_from_checkpoint=True)
metrics = train_result.metrics
logger.info(f"Training Metrics: {metrics}")

logger.info("Evaluation Of The Model On Test Set...")
eval_metrics = trainer.evaluate(tokenized_test)
logger.info(f"Eval Metrics: {eval_metrics}")

base_path = 'models/finetuned_model'

os.makedirs(base_path, exist_ok=True)
model.save_pretrained(base_path)
tokenizer.save_pretrained(base_path)

with open(base_path+'/metrics.json', 'w') as f:
    json.dump({**metrics, **eval_metrics}, f, indent=4)

logger.info("Fine Tuning Completed, model saved successfully.")

logger.info("Upload The Model On HuggingFace...")

upload_folder(
    repo_id=HF_REPO_DIR,
    folder_path="models/finetuned_model",
    repo_type="model",
)

logger.info("Upload Complete.")
