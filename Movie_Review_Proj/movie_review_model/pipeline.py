
""" Define Pipleine for SentimentClassifier Model"""
from sklearn.pipeline import Pipeline
import torch

from movie_review_model.processing.preprocess import TokenizerTransformer, SentimentClassifier
from movie_review_model import model as movie_model


def build_pipeline():
    """
    Build  the pipeline.
    """
    model = movie_model.build_model()
    optimizer = movie_model.get_optimizer(model, 1e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return Pipeline([
        ("tokenizer", TokenizerTransformer()),
        ("classifier", SentimentClassifier(model=model, optimizer=optimizer, device=device)),
    ])
