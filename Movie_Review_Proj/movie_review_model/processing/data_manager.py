"""data load fromsource

Returns:
    _type_: _description_
"""
from pathlib import Path
import sys
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# from movie_review_model.config.core import config
# from movie_review_model.config import core
from datasets import load_dataset

# import typing as t
# import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def load_custom_dataset(filepath, split_percentage=10):
    """
    Load a custom dataset and return a specified percentage for training.

    Args:
        filepath (str): Path to the CSV file containing the dataset.
        split_percentage (int): Percentage of the dataset to load (e.g., 10 for 10%).

    Returns:
        texts (pd.Series): The text data.
        labels (pd.Series): The labels corresponding to the text data.
    """
    df = pd.read_csv(filepath)
    texts = df['text']
    labels = df['label'].values

    # Use train_test_split to select a small subset of the data
    texts_subset, _, labels_subset, _ = train_test_split(
    texts.tolist(), labels, test_size=split_percentage / 100, random_state=42   
    )

    return texts_subset, labels_subset


def load_imdb_dataset(sample_size=1000):
    """Load a small sample from the IMDB dataset."""
    dataset = load_dataset("imdb")
    # Take a smaller sample for faster training
    train_data = dataset["train"].shuffle(seed=42).select(range(sample_size))
    
    # Extract texts and labels
    texts = train_data["text"]
    labels = train_data["label"]

    return texts, labels

# def savePipeline(*, pipeline_to_persist: Pipeline) -> None:
#     # Prepare versioned save file name
#     save_file_name = f"{config.output.output_model_path}{_version}.pkl"
#     save_path = TRAINED_MODEL_DIR / save_file_name
    
#     remove_old_pipelines(files_to_keep=[save_file_name])
#     joblib.dump(pipeline_to_persist, save_path)


# def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
#     """Persist the pipeline.
#     Saves the versioned model, and overwrites any previous saved models.
#     This ensures that when the package is published, there is only one trained model that
#     can be called, and we know exactly how it was built.
#     """

#     # Prepare versioned save file name
#     save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#     save_path = TRAINED_MODEL_DIR / save_file_name

#     remove_old_pipelines(files_to_keep=[save_file_name])
#     joblib.dump(pipeline_to_persist, save_path)


# def load_pipeline(*, file_name: str) -> Pipeline:
#     """Load a persisted pipeline."""

#     file_path = TRAINED_MODEL_DIR / file_name
#     trained_model = joblib.load(filename=file_path)
#     return trained_model


# def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
#     """
#     Remove old model pipelines.
#     This is to ensure there is a simple one-to-one mapping between the package version and
#     the model version to be imported and used by other applications.
#     """
#     do_not_delete = files_to_keep + ["__init__.py"]
#     for model_file in TRAINED_MODEL_DIR.iterdir():
#         if model_file.name not in do_not_delete:
#             model_file.unlink()