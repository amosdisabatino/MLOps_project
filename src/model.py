from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = 'DiSabatino/mlops-sentiment-model'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text: str) -> dict:
    """
    TODO
    """
    
    model.config.id2label = {0: 'negative', 1: 'positive'}
    model.config.label2id = {'negative': 0, 'positive': 1}

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)

    pred_id = torch.argmax(probs, dim=-1).item()
    labels = model.config.id2label
    label = labels[pred_id] if labels else str(pred_id)

    return {
        'label': label,
        'confidence': probs[0][pred_id].item()
    }
