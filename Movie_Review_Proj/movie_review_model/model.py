from transformers import BertForSequenceClassification, AdamW

def build_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Positive/Negative
    )
    return model

def get_optimizer(model, learning_rate=2e-5):
    return AdamW(model.parameters(), lr=learning_rate)
