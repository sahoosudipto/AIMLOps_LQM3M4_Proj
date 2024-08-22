"""data load fromsource

Returns:
    _type_: _description_
"""
from datasets import load_dataset
import pandas as pd
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