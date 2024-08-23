"""Module for model training."""
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from movie_review_model.config.core import config
from movie_review_model.processing.data_manager import load_custom_dataset, load_imdb_dataset
from movie_review_model.pipeline import build_pipeline


import pickle


def train_pipeline():
    """training pipeline"""
    # Load dataset
    # texts, labels = load_custom_dataset(config.data.train_data_path, split_percentage=25)

    # Load a small sample from the IMDB dataset
    texts, labels = load_imdb_dataset(sample_size=50) #for fast training
    
    # Ensure texts are in the correct format
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Texts must be a list of strings.")

    # Build pipeline
    pipeline = build_pipeline()

    # Train the model using the pipeline
    pipeline.fit(texts, labels)

    # Save the pipeline (including the trained model)
    # pipeline.named_steps['classifier'].model.save_pretrained(config.output.output_model_path)
    
    #     # Save the pipeline as a .pkl file
    model_dir = Path(config.output.output_model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "trained_pipeline.pkl"
    print(f"Model path: {model_path}")
    pipeline.named_steps['classifier'].model.save_pretrained(model_path)

    # save_pipeline(pipeline_to_persist=pipeline)
    if model_path.exists():
        print(f"Model saved successfully at: {model_path}")
    else:
        print("Model save failed.")
    print("Model and pipeline saved successfully.")

if __name__ == "__main__":
    train_pipeline()
