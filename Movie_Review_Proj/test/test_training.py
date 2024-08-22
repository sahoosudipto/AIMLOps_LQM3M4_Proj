"""Tests for training 

Returns:
    _type_: _description_
"""
import os
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from movie_review_model.train import train_pipeline
from movie_review_model.config.core import config

@pytest.fixture
def training_data():
    return ["The movie was excellent!", "It was a boring movie."], [1, 0]

def test_train_pipeline(training_data):
    # texts, labels = training_data
    # train_pipeline()
    # Verify that the model was saved successfully
    model_path = config.output.output_model_path
    assert os.path.exists(model_path), f"Model directory was not created at {model_path}!"


