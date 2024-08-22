"""Tests for pipeline 

Returns:
    _type_: _description_
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from movie_review_model.pipeline import build_pipeline

@pytest.fixture
def sample_pipeline():
    return build_pipeline()

def test_pipeline_structure(sample_pipeline):
    assert 'tokenizer' in sample_pipeline.named_steps
    assert 'classifier' in sample_pipeline.named_steps

def test_pipeline_fitting(sample_pipeline, sample_text):
    labels = [1]  # Assume positive sentiment
    sample_pipeline.fit([sample_text], labels)
    assert sample_pipeline.named_steps['classifier'].model is not None

