"""Module for model training."""
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from movie_review_model.config.core import config
from movie_review_model.processing.data_manager import load_custom_dataset
from movie_review_model.pipeline import build_pipeline




def train_pipeline():
    """training pipeline"""
    # Load dataset
    texts, labels = load_custom_dataset(config.data.train_data_path, split_percentage=5)
    # Ensure texts are in the correct format
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Texts must be a list of strings.")

    # Build pipeline
    pipeline = build_pipeline()

    # Train the model using the pipeline
    pipeline.fit(texts, labels)

    # Save the pipeline (including the trained model)
    pipeline.named_steps['classifier'].model.save_pretrained(config.output.output_model_path)
    print("Model and pipeline saved successfully.")

if __name__ == "__main__":
    train_pipeline()
