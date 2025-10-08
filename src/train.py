import os
import json
import logging
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from evaluate import load

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
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=config.num_labels,
)

def tokenize_function(datas):
    return tokenizer(
        datas['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset['train'].shuffle(seed=42).select(range(2000))
test_dataset = tokenized_dataset['test'].shuffle(seed=42).select(range(500))

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
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
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

with open('models/finetuned_model/metrics.json', 'w') as f:
    json.dump({**metrics, **eval_metrics}, f, indent=4)

logger.info("Fine Tuning Completed, model saved successfully.")
