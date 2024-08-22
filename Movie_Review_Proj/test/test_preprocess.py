"""Tests for preprocessing 

Returns:
    _type_: _description_
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from movie_review_model.processing.preprocess import TokenizerTransformer
import pytest
import numpy as np

@pytest.fixture
def sample_text():
    return "The movie was fantastic!"

def test_preprocessing(sample_text):
    tokenizer_transformer = TokenizerTransformer()
    transformed_output = tokenizer_transformer.transform([sample_text])
    
    # Ensure the output is a NumPy array
    assert isinstance(transformed_output, np.ndarray), "Expected output to be a NumPy array."
    
    # Ensure the output has the correct shape (1 sequence of token IDs)
    assert transformed_output.shape[0] == 1, "Expected output to contain a single sequence."
    
    # Ensure the sequence contains token IDs
    assert len(transformed_output[0]) > 0, "The tokenized sequence should not be empty."
    
    # Optional: Print the token IDs for debugging
    print("Token IDs:", transformed_output[0])