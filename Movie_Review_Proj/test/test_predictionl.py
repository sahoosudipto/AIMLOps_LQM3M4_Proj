import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from movie_review_model.predict import predict

@pytest.fixture
def positive_review():
  return "This movie was fantastic!"

@pytest.fixture
def negative_review():
  return "This movie was terrible!"

def test_predict_negative_sentiment(negative_review):
  prediction = predict(negative_review)
  print(prediction)
  assert prediction['label'] == 0
  #   assert prediction['confidence'] > 0.5

def test_predict_positive_sentiment(positive_review):
  prediction = predict(positive_review)
  print(prediction)
  assert prediction['label'] == 1
#   assert prediction['confidence'] > 0.5

def test_predict_invalid():
  prediction = predict(None)
  assert prediction is None

