"""prediction pipeline"""
from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from movie_review_model.pipeline import build_pipeline
from movie_review_model.processing.preprocess import TokenizerTransformer
import torch
from transformers import BertForSequenceClassification, BertTokenizer

def predict(text):
    """Predict the sentiment of a text input.

    Args:
        text (str): The text to be classified.

    Returns:
        str: "Positive" or "Negative" based on the prediction.
    """

    # Load the trained model and tokenizer (if not already loaded)
    model = BertForSequenceClassification.from_pretrained(
        "./movie_review_model/trained_models/model"
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare the input data
    input_ids = tokenizer(text, return_tensors="pt", max_length=128).input_ids.to(device)

    # Use the pipeline to predict the sentiment
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

    sentiment = "Positive" if predicted_label == 1 else "Negative"
    return sentiment

if __name__ == "__main__":
    # Example input data
    _review = "The movie was really good and I enjoyed it a lot."
    # Predict sentiment
    _result = predict(_review)
    print(f"Predicted sentiment: {_result}")
