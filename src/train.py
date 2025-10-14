import os
import json
import logging
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from evaluate import load
from huggingface_hub import upload_folder

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
dataset = load_dataset('imdb')

logger.info("Loading Model...")

model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=config.num_labels,
    ignore_mismatched_sizes=True
)

train_data = dataset['train']
test_data = dataset['test']

def get_data_by_labels(datas, data_type, num_rec):
    return datas.filter(
        lambda data: data['label'] == data_type
    ).select(range(num_rec))

train_negative = get_data_by_labels(train_data, 0, 1000)
train_positive = get_data_by_labels(train_data, 1, 1000)
test_negative = get_data_by_labels(test_data, 0, 250)
test_positive = get_data_by_labels(test_data, 1, 250)

train_set = concatenate_datasets([train_negative, train_positive])
test_set = concatenate_datasets([test_negative, test_positive])

train_set = train_set.shuffle(42)
test_set = test_set.shuffle(42)

def tokenize_function(datas):
    return tokenizer(
        datas['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

tokenized_train = train_set.map(tokenize_function, batched=True)
tokenized_test = test_set.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy='steps',
    logging_steps=50,
)

accuracy = load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logger.info("Start The Training Of The Model...")
train_result = trainer.train()
metrics = train_result.metrics
logger.info(f"Training Metrics: {metrics}")

logger.info("Evaluation Of The Model On Test Set...")
eval_metrics = trainer.evaluate()
logger.info(f"Eval Metrics: {eval_metrics}")

os.makedirs('models/finetuned_model', exist_ok=True)
model.save_pretrained('models/finetuned_model')
tokenizer.save_pretrained('models/finetuned_model')

with open('models/finetuned_model/metrics.json', 'w') as f:
    json.dump({**metrics, **eval_metrics}, f, indent=4)

logger.info("Fine Tuning Completed, model saved successfully.")

logger.info("Upload The Model On HuggingFace...")

upload_folder(
    repo_id="DiSabatino/mlops-sentiment-model",
    folder_path="models/finetuned_model",
    repo_type="model",
)

logger.info("Upload Complete.")
