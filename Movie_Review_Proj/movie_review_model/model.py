import torch
from transformers import BertForSequenceClassification

def build_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Positive/Negative
    )
    return model

def get_optimizer(model, learning_rate=2e-5):
    # Use PyTorch's AdamW optimizer instead of the deprecated transformers' AdamW
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)
