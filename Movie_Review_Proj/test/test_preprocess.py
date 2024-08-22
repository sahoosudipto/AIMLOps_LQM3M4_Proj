"""Tests for preprocessing 

Returns:
    _type_: _description_
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from movie_review_model.processing.preprocess import TokenizerTransformer

@pytest.fixture
def sample_text():
    return "The movie was fantastic!"

def test_preprocessing(sample_text):
    tokenizer_transformer = TokenizerTransformer()
    transformed_output = tokenizer_transformer.transform([sample_text])
    assert 'input_ids' in transformed_output
    assert 'attention_mask' in transformed_output
    assert len(transformed_output['input_ids']) > 0
